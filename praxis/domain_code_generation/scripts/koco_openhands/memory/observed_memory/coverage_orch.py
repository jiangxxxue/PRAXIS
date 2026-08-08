"""Coverage measurement orchestrator — runs on host, coordinates Docker execution.

Measures how well generated test inputs cover the GT function's code paths.
Uses Python coverage.py inside Docker containers (same images as evaluation).
"""

import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from memory.config import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    candidates_path,
    code_dir,
    coverage_result_path,
    feedback_status_path,
    test_input_path,
)
from memory.observed_memory.quality import coverage_has_execution_failure

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import get_docker_image

# Module-level cache for GPU availability test (run once per CLI invocation)
_gpu_cache = {}  # {image: bool}


def _coverage_docker_runtime() -> str | None:
    """Return the Docker runtime to use for coverage containers.

    Some shared servers configure Docker's default runtime as ``nvidia``. That
    can make CPU-only coverage fail before the container starts when the NVIDIA
    container runtime cannot inspect host GPUs. Force plain runc by default and
    allow callers to override it when needed.
    """
    value = os.environ.get("PRAXIS_COVERAGE_DOCKER_RUNTIME", "runc").strip()
    return value or None


def _coverage_docker_gpus_enabled() -> bool:
    return os.environ.get("PRAXIS_COVERAGE_DOCKER_GPUS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _gpu_ready(image: str) -> bool:
    """Check if GPU passthrough works for a Docker image. Cached per session."""
    if image in _gpu_cache:
        return _gpu_cache[image]
    try:
        _test = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", image, "echo", "ok"],
            capture_output=True, timeout=30
        )
        _gpu_cache[image] = _test.returncode == 0
    except Exception:
        _gpu_cache[image] = False
    return _gpu_cache[image]


def _parse_location(location: str) -> tuple[str, int, int]:
    """Parse 'code/recipe/prime/algo.py:line 21-24' or 'code/.../file.py:21-24'."""
    rel = location.removeprefix("code/")
    if ":line " in rel:
        path_part, range_part = rel.split(":line ", 1)
    elif ":" in rel:
        path_part, range_part = rel.rsplit(":", 1)
    else:
        return rel, 0, 0
    try:
        start_s, end_s = range_part.split("-")
        return path_part, int(start_s), int(end_s)
    except ValueError:
        return path_part, 0, 0


def _existing_test_input_path(
    framework: str,
    example: str,
    function_name: str,
) -> Path | None:
    path = test_input_path(framework, example, function_name)
    if path.exists():
        return path
    if "." in function_name:
        leaf_path = test_input_path(
            framework,
            example,
            function_name.rsplit(".", 1)[-1],
        )
        if leaf_path.exists() and _test_input_matches_function(
            leaf_path,
            function_name,
        ):
            return leaf_path
    return None


def _test_input_matches_function(path: Path, function_name: str) -> bool:
    """Return whether a legacy leaf-name input declares the requested target."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "FUNCTION_NAME"
            for target in targets
        ):
            continue
        value = node.value
        return (
            isinstance(value, ast.Constant)
            and value.value == function_name
        )
    return False


def _coverage_fingerprint(
    *,
    test_input: Path,
    source_file: Path,
    runner_file: Path,
    implementation_location: str,
    per_test: bool,
    capture_output: bool,
    execution_context: str,
) -> str:
    digest = sha256()
    for path in (test_input, source_file, runner_file):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(implementation_location.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(per_test).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(capture_output).encode("ascii"))
    digest.update(b"\0")
    digest.update(execution_context.encode("utf-8"))
    return digest.hexdigest()


def _coverage_python_overlay(framework: str, example: str) -> Path | None:
    suffix = "".join(
        char if char.isalnum() else "_"
        for char in f"{framework}_{example}"
    ).upper()
    value = os.environ.get(
        f"PRAXIS_COVERAGE_PYTHONPATH_{suffix}",
        "",
    ).strip()
    if not value:
        return None

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"Coverage Python overlay does not exist: {path}")
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "Coverage Python overlays must be inside PROJECT_ROOT so Docker "
            f"can access them: {path}"
        ) from exc
    return path


def _coverage_overlay_signature(path: Path) -> str:
    digest = sha256()
    files = (
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )
    for child in sorted(files):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _coverage_execution_context(
    framework: str,
    native: bool,
    example: str = "",
) -> str:
    overlay = _coverage_python_overlay(framework, example)
    overlay_context = ""
    if overlay is not None:
        overlay_context = (
            f":pythonpath={overlay}:"
            f"signature={_coverage_overlay_signature(overlay)}"
        )

    if native:
        return f"native:{sys.executable}{overlay_context}"

    return (
        f"docker:{get_docker_image(framework)}:"
        f"runtime={_coverage_docker_runtime()}:"
        f"gpus={_coverage_docker_gpus_enabled()}"
        f"{overlay_context}"
    )


def _load_reusable_coverage(
    output_path: Path,
    *,
    dependencies: list[Path],
    fingerprint: str,
) -> dict | None:
    if not output_path.is_file():
        return None
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or coverage_has_execution_failure(data):
        return None
    if not isinstance(data.get("line_coverage"), (int, float)):
        return None

    metadata = data.get("_praxis_coverage")
    if isinstance(metadata, dict) and metadata.get("fingerprint"):
        return data if metadata["fingerprint"] == fingerprint else None

    try:
        output_mtime = output_path.stat().st_mtime_ns
        if any(
            not dependency.is_file()
            or dependency.stat().st_mtime_ns > output_mtime
            for dependency in dependencies
        ):
            return None
    except OSError:
        return None
    return data


def _coverage_metadata(
    *,
    framework: str,
    example: str,
    test_input: Path,
    source_file: Path,
    target_signature: str,
    per_test: bool,
    capture_output: bool,
    native: bool,
) -> dict:
    return {
        "fingerprint": _coverage_fingerprint(
            test_input=test_input,
            source_file=source_file,
            runner_file=Path(__file__).with_name("coverage_runner.py"),
            implementation_location=target_signature,
            per_test=per_test,
            capture_output=capture_output,
            execution_context=_coverage_execution_context(
                framework,
                native,
                example=example,
            ),
        ),
        "per_test": per_test,
        "capture_output": capture_output,
    }


def _save_coverage_data(
    framework: str,
    example: str,
    function_name: str,
    data: dict,
    *,
    test_input: Path,
    source_file: Path,
    target_signature: str,
    per_test: bool,
    capture_output: bool,
    native: bool,
) -> dict:
    data["_praxis_coverage"] = _coverage_metadata(
        framework=framework,
        example=example,
        test_input=test_input,
        source_file=source_file,
        target_signature=target_signature,
        per_test=per_test,
        capture_output=capture_output,
        native=native,
    )
    out = coverage_result_path(framework, example, function_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _feedback_accepts_failed_coverage(
    framework: str,
    example: str,
    function_name: str,
    coverage_data: dict,
) -> bool:
    metadata = coverage_data.get("_praxis_coverage")
    if not isinstance(metadata, dict) or not metadata.get("fingerprint"):
        return False
    try:
        feedback_status = json.loads(
            feedback_status_path(
                framework,
                example,
                function_name,
            ).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if feedback_status.get("status") not in {"exhausted", "unrunnable"}:
        return False
    details = feedback_status.get("details")
    return (
        isinstance(details, dict)
        and details.get("coverage_fingerprint") == metadata["fingerprint"]
    )


def _load_gt_locations_from_jsonl(framework: str, example: str) -> dict[str, tuple[str, int, int]]:
    """Load correct line ranges from the parsed JSONL data.

    Returns dict mapping function_name -> (source_file, line_start, line_end).
    Keys include both full names (e.g. 'RayPRIMETrainer.filter_and_downsample')
    and short names (e.g. 'filter_and_downsample') for flexible lookup.
    """
    from runner import load_jsonl, _parse_impl_location

    data_file = SCRIPTS_DIR / "data" / framework / f"algorithm_methods_data_{example}.jsonl"
    if not data_file.exists():
        return {}

    result = {}
    for r in load_jsonl(str(data_file)):
        fn = r.get("function_name", "")
        loc = r.get("implementation_location", "")
        if not fn or not loc:
            continue
        rel_path, start, end = _parse_impl_location(loc)
        if rel_path and start and end:
            entry = (rel_path, start, end, fn)
            result[fn] = entry
            # Also register short name for class methods (e.g. 'Class.method' -> 'method')
            if "." in fn:
                short = fn.rsplit(".", 1)[1]
                result[short] = entry
    return result


def run_coverage_single(
    framework: str,
    example: str,
    function_name: str,
    per_test: bool = False,
    capture_output: bool = True,
    gt_locations: dict | None = None,
    timeout: int = 120,
    native: bool = False,
) -> dict | None:
    """Run coverage measurement for a single function.

    By default runs inside Docker (isolated). When *native* is True, runs
    coverage_runner.py directly on the host — useful when Docker is unavailable
    or unnecessary (e.g. autodl GPU instances).
    """
    # Find candidate metadata
    cands_path = candidates_path(framework, example)
    if not cands_path.exists():
        print(f"  SKIP {function_name}: candidates.json not found")
        return None

    candidates = json.loads(cands_path.read_text(encoding="utf-8"))
    target = next((c for c in candidates if c["function_name"] == function_name), None)
    if not target:
        print(f"  SKIP {function_name}: not in candidates.json")
        return None

    # Verify test_input exists (try full name, then leaf name fallback)
    ti_path = _existing_test_input_path(framework, example, function_name)
    if ti_path is None:
        print(f"  SKIP {function_name}: test_input.py not found")
        return None

    # Use correct line ranges from JSONL (candidates.json line ranges may be wrong)
    resolved_name = function_name
    if gt_locations and function_name in gt_locations:
        entry = gt_locations[function_name]
        source_file, line_start, line_end = entry[0], entry[1], entry[2]
        # Use the full qualified name (e.g. 'RayPRIMETrainer.filter_and_downsample')
        if len(entry) > 3:
            resolved_name = entry[3]
    else:
        source_file, line_start, line_end = _parse_location(target["implementation_location"])
    source_path = code_dir(framework, example) / source_file
    target_signature = (
        f"{source_file}:{line_start}-{line_end}:{resolved_name}"
    )

    # Docker paths
    image = get_docker_image(framework)
    host_root = str(PROJECT_ROOT)
    mnt = "/workspace/project"

    ctr_source_dir = f"{mnt}/{framework}/test_examples/{example}/code"
    ctr_runner = f"{mnt}/scripts/koco_openhands/memory/observed_memory/coverage_runner.py"

    # Map the actual run-scoped artifact paths into the project mount.
    output_path = coverage_result_path(framework, example, function_name)
    ctr_test_input = f"{mnt}/{ti_path.relative_to(PROJECT_ROOT)}"
    ctr_output = f"{mnt}/{output_path.relative_to(PROJECT_ROOT)}"

    # Build docker command
    container_name = f"praxis-coverage-{os.getpid()}-{uuid4().hex[:12]}"
    docker_cmd = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{host_root}:{mnt}",
    ]
    overlay = _coverage_python_overlay(framework, example)
    if overlay is not None:
        ctr_overlay = f"{mnt}/{overlay.relative_to(PROJECT_ROOT.resolve())}"
        docker_cmd[3:3] = ["-e", f"PYTHONPATH={ctr_overlay}"]
    runtime = _coverage_docker_runtime()
    if runtime:
        docker_cmd[3:3] = ["--runtime", runtime]

    # GPU passthrough for CUDA-based images (verl / open-r1) — optional.
    # Checks once per image. When GPU is unavailable, force CPU mode to avoid
    # CUDA initialization hangs during torch import.
    if framework in ("verl", "open-r1"):
        if _coverage_docker_gpus_enabled() and _gpu_ready(image):
            docker_cmd.extend(["--gpus", "all"])
        else:
            # Force CPU — prevents torch from trying to init CUDA (slow/hang)
            docker_cmd[3:3] = ["-e", "CUDA_VISIBLE_DEVICES="]

    if platform.system() == "Linux":
        uid, gid = os.getuid(), os.getgid()
        docker_cmd[3:3] = [
            "--user", f"{uid}:{gid}",
            "-e", "HOME=/tmp",
            "-e", "USER=benchuser",
        ]

    docker_cmd.extend([
        image,
        "python3", ctr_runner,
        "--source-dir", ctr_source_dir,
        "--source-file", source_file,
        "--function-name", resolved_name,
        "--test-input-file", ctr_test_input,
        "--line-start", str(line_start),
        "--line-end", str(line_end),
        "--output-file", ctr_output,
    ])
    if per_test:
        docker_cmd.append("--per-test")
    if capture_output:
        docker_cmd.append("--capture-output")

    # Native execution: bypass Docker, run coverage_runner.py directly on host
    exec_env = None
    if native:
        import sys as _sys
        runner_path = SCRIPTS_DIR / "koco_openhands" / "memory" / "observed_memory" / "coverage_runner.py"
        source_dir_real = PROJECT_ROOT / framework / "test_examples" / example / "code"
        ti_path_real = test_input_path(framework, example, function_name)
        out_path_real = coverage_result_path(framework, example, function_name)
        exec_cmd = [
            _sys.executable, str(runner_path),
            "--source-dir", str(source_dir_real),
            "--source-file", source_file,
            "--function-name", resolved_name,
            "--test-input-file", str(ti_path_real),
            "--line-start", str(line_start),
            "--line-end", str(line_end),
            "--output-file", str(out_path_real),
        ]
        if per_test:
            exec_cmd.append("--per-test")
        if capture_output:
            exec_cmd.append("--capture-output")
        if overlay is not None:
            exec_env = os.environ.copy()
            existing_pythonpath = exec_env.get("PYTHONPATH", "")
            exec_env["PYTHONPATH"] = str(overlay) + (
                os.pathsep + existing_pythonpath
                if existing_pythonpath
                else ""
            )
    else:
        exec_cmd = docker_cmd

    print(f"  Running coverage for {function_name} (timeout={timeout}s)...")
    try:
        try:
            proc = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=exec_env,
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT {function_name}: exceeded {timeout}s, skipping")
            data = {
                "function_name": function_name,
                "line_coverage": 0.0,
                "num_covered_lines": 0,
                "total_executable_lines": 0,
                "covered_lines": [],
                "missing_lines": [],
                "num_execution_errors": 1,
                "execution_errors": [{"error": f"Timed out after {timeout}s", "category": "timeout"}],
                "is_execution_failure": True,
            }
            return _save_coverage_data(
                framework,
                example,
                function_name,
                data,
                test_input=ti_path,
                source_file=source_path,
                target_signature=target_signature,
                per_test=per_test,
                capture_output=capture_output,
                native=native,
            )
    finally:
        if not native:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    if proc.stdout:
        for line in proc.stdout.strip().splitlines():
            print(f"    {line}")

    if proc.returncode != 0:
        print(f"  FAIL {function_name}: exit code {proc.returncode}")
        stderr_tail = ""
        if proc.stderr:
            for line in proc.stderr.strip().splitlines()[-20:]:
                print(f"    {line}")
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
        # Save error-only coverage JSON so failures remain visible in results.
        data = {
            "function_name": function_name,
            "line_coverage": 0.0,
            "num_covered_lines": 0,
            "total_executable_lines": 0,
            "covered_lines": [],
            "missing_lines": [],
            "num_execution_errors": 1,
            "execution_errors": [{"error": stderr_tail, "category": "execution_failure"}],
            "is_execution_failure": True,
        }
        return _save_coverage_data(
            framework,
            example,
            function_name,
            data,
            test_input=ti_path,
            source_file=source_path,
            target_signature=target_signature,
            per_test=per_test,
            capture_output=capture_output,
            native=native,
        )

    # Read result
    out_path = coverage_result_path(framework, example, function_name)
    if not out_path.exists():
        print(f"  FAIL {function_name}: output not created")
        data = {
            "function_name": function_name,
            "line_coverage": 0.0,
            "num_covered_lines": 0,
            "total_executable_lines": 0,
            "covered_lines": [],
            "missing_lines": [],
            "num_execution_errors": 1,
            "execution_errors": [{
                "error": "Coverage runner did not create output",
                "category": "execution_failure",
            }],
            "is_execution_failure": True,
        }
        return _save_coverage_data(
            framework,
            example,
            function_name,
            data,
            test_input=ti_path,
            source_file=source_path,
            target_signature=target_signature,
            per_test=per_test,
            capture_output=capture_output,
            native=native,
        )

    data = json.loads(out_path.read_text(encoding="utf-8"))
    data = _save_coverage_data(
        framework,
        example,
        function_name,
        data,
        test_input=ti_path,
        source_file=source_path,
        target_signature=target_signature,
        per_test=per_test,
        capture_output=capture_output,
        native=native,
    )
    lc = data.get("line_coverage", 0)
    n_cov = data.get("num_covered_lines", "?")
    n_tot = data.get("total_executable_lines", "?")
    n_err = data.get("num_execution_errors", 0)
    results = data.get("results")
    output_info = ""
    if results is not None:
        ok = sum(1 for r in results if r["success"])
        output_info = f", outputs: {ok}/{len(results)}"
    print(f"  OK   {function_name}: {lc:.1%} ({n_cov}/{n_tot} lines, {n_err} errors{output_info})")
    return data


def run_coverage(
    framework: str,
    example: str,
    function_name: str | None = None,
    per_test: bool = False,
    capture_output: bool = True,
    timeout: int = 120,
    native: bool = False,
    resume: bool = False,
    concurrency: int = 1,
) -> list[dict]:
    """Run coverage for all (or one) candidate functions in an example.

    Args:
        native: If True, run coverage_runner.py directly on host instead of
            inside Docker. Useful when Docker is unavailable or when running
            on a GPU instance that already has all dependencies installed.
        resume: Reuse successful coverage whose inputs have not changed.
        concurrency: Maximum number of independent functions to measure at once.
    """
    cands_path = candidates_path(framework, example)
    if not cands_path.exists():
        print(f"candidates.json not found: {cands_path}")
        return []

    candidates = json.loads(cands_path.read_text(encoding="utf-8"))

    if function_name:
        candidates = [c for c in candidates if c["function_name"] == function_name]
        if not candidates:
            print(f"Function '{function_name}' not found in candidates.json")
            return []
    else:
        runnable = []
        missing = []
        for c in candidates:
            fn = c["function_name"]
            ti_path = _existing_test_input_path(framework, example, fn)
            if ti_path is not None:
                runnable.append(c)
            else:
                missing.append(fn)
        if missing:
            print(
                "  WARN: coverage will skip "
                f"{len(missing)}/{len(candidates)} candidate(s) without test_input.py: "
                f"{', '.join(missing[:10])}"
            )
        candidates = runnable

    # Load correct line ranges from JSONL once
    gt_locations = _load_gt_locations_from_jsonl(framework, example)
    if gt_locations:
        print(f"  Loaded {len(gt_locations)} GT locations from JSONL")

    reusable = {}
    pending = []
    for c in candidates:
        fn = c["function_name"]
        if not resume or per_test:
            pending.append(c)
            continue

        ti_path = _existing_test_input_path(framework, example, fn)
        entry = gt_locations.get(fn)
        if entry:
            source_file, line_start, line_end = entry[0], entry[1], entry[2]
            resolved_name = entry[3] if len(entry) > 3 else fn
        else:
            source_file, line_start, line_end = _parse_location(
                c["implementation_location"]
            )
            resolved_name = fn
        source_path = code_dir(framework, example) / source_file
        runner_path = Path(__file__).with_name("coverage_runner.py")
        if (
            ti_path is None
            or not source_path.is_file()
            or not runner_path.is_file()
        ):
            pending.append(c)
            continue

        target_signature = (
            f"{source_file}:{line_start}-{line_end}:{resolved_name}"
        )
        fingerprint = _coverage_fingerprint(
            test_input=ti_path,
            source_file=source_path,
            runner_file=runner_path,
            implementation_location=target_signature,
            per_test=per_test,
            capture_output=capture_output,
            execution_context=_coverage_execution_context(
                framework,
                native,
                example=example,
            ),
        )
        data = _load_reusable_coverage(
            coverage_result_path(framework, example, fn),
            dependencies=[cands_path, ti_path, source_path, runner_path],
            fingerprint=fingerprint,
        )
        if (
            data is None
            and coverage_result_path(framework, example, fn).is_file()
        ):
            try:
                failed_data = json.loads(
                    coverage_result_path(
                        framework,
                        example,
                        fn,
                    ).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                failed_data = None
            if (
                isinstance(failed_data, dict)
                and failed_data.get("is_execution_failure")
                and failed_data.get("_praxis_coverage", {}).get("fingerprint")
                == fingerprint
                and _feedback_accepts_failed_coverage(
                    framework,
                    example,
                    fn,
                    failed_data,
                )
            ):
                data = failed_data
        if data is None:
            pending.append(c)
            continue
        reusable[fn] = data
        label = "terminal failed" if data.get("is_execution_failure") else "successful"
        print(f"  SKIP {fn}: reusable {label} coverage")

    def _run(candidate):
        fn = candidate["function_name"]
        return fn, run_coverage_single(
            framework, example, fn, per_test, capture_output,
            gt_locations=gt_locations, timeout=timeout,
            native=native,
        )

    measured = {}
    workers = max(1, concurrency)
    if workers == 1:
        for candidate in pending:
            fn, data = _run(candidate)
            if data:
                measured[fn] = data
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, candidate): candidate for candidate in pending}
            for future in as_completed(futures):
                fn, data = future.result()
                if data:
                    measured[fn] = data

    results = []
    for candidate in candidates:
        fn = candidate["function_name"]
        data = reusable.get(fn) or measured.get(fn)
        if data:
            results.append(data)

    # Summary table
    if results:
        _print_summary(results)

    return results


def _save_error_coverage(
    framework: str, example: str, function_name: str, error_text: str,
):
    """Save a minimal coverage JSON with error info for functions that failed to execute.

    This keeps failed coverage attempts visible to downstream analysis.
    """
    out = coverage_result_path(framework, example, function_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "function_name": function_name,
        "line_coverage": 0.0,
        "num_covered_lines": 0,
        "total_executable_lines": 0,
        "covered_lines": [],
        "missing_lines": [],
        "num_execution_errors": 1,
        "execution_errors": [
            {"error": error_text, "category": "execution_failure"}
        ],
        "is_execution_failure": True,
    }
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _print_summary(results: list[dict]):
    has_outputs = any("results" in r for r in results)
    if has_outputs:
        print(f"\n{'Function':<40} {'Line Cov':>10} {'Errors':>8} {'Outputs':>10} {'Cases (n/e/err)':>16}")
        print("-" * 88)
    else:
        print(f"\n{'Function':<40} {'Line Cov':>10} {'Errors':>8} {'Cases (n/e/err)':>16}")
        print("-" * 76)
    for r in results:
        fn = r["function_name"]
        lc = r.get("line_coverage") or 0
        err = r.get("num_execution_errors") or 0
        stub_tag = " [stub]" if r.get("is_abstract_stub") else ""
        ntc = r.get("num_test_cases") or {}
        cases = f"{ntc.get('normal', 0)}/{ntc.get('edge', 0)}/{ntc.get('error', 0)}"
        fn_display = fn + stub_tag
        if has_outputs:
            outputs = r.get("results", [])
            ok = sum(1 for o in outputs if o.get("success"))
            print(f"{fn_display:<40} {lc:>9.1%} {err:>8} {ok:>4}/{len(outputs):<5} {cases:>16}")
        else:
            print(f"{fn_display:<40} {lc:>9.1%} {err:>8} {cases:>16}")

    # Average excluding abstract stubs
    concrete = [r for r in results if not r.get("is_abstract_stub")]
    stubs = [r for r in results if r.get("is_abstract_stub")]
    if concrete:
        avg_lc = sum(r.get("line_coverage") or 0 for r in concrete) / len(concrete)
        tot_err = sum(r.get("num_execution_errors") or 0 for r in concrete)
    else:
        avg_lc = 0.0
        tot_err = 0
    if has_outputs:
        print("-" * 88)
    else:
        print("-" * 76)
    print(f"{'Average (excl. stubs)':<40} {avg_lc:>9.1%} {tot_err:>8}")
    if stubs:
        print(f"  ({len(stubs)} abstract stub(s) excluded from average)")
