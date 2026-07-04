"""Coverage measurement orchestrator — runs on host, coordinates Docker execution.

Measures how well generated test inputs cover the GT function's code paths.
Uses Python coverage.py inside Docker containers (same images as evaluation).
"""

import json
import os
import platform
import subprocess
from pathlib import Path

from memory.config import (
    DERIVED_DIR,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    candidates_path,
    code_dir,
    coverage_result_path,
    test_input_path,
)

# Module-level cache for GPU availability test (run once per CLI invocation)
_gpu_cache = {}  # {image: bool}


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
    from config import get_docker_image

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
    ti_path = test_input_path(framework, example, function_name)
    if not ti_path.exists():
        leaf = function_name.rsplit(".", 1)[-1] if "." in function_name else ""
        if leaf and leaf != function_name:
            ti_path = test_input_path(framework, example, leaf)
    if not ti_path.exists():
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

    # Docker paths
    image = get_docker_image(framework)
    host_root = str(PROJECT_ROOT)
    mnt = "/workspace/project"

    ctr_source_dir = f"{mnt}/{framework}/test_examples/{example}/code"
    ctr_runner = f"{mnt}/scripts/koco_openhands/memory/observed_memory/coverage_runner.py"

    # Derive container paths from the host DERIVED_DIR relative to PROJECT_ROOT
    derived_rel = DERIVED_DIR.relative_to(PROJECT_ROOT)
    ctr_test_input = f"{mnt}/{derived_rel}/observed_knowledge/{framework}/{example}/{function_name}_test_input.py"
    ctr_output = f"{mnt}/{derived_rel}/observed_knowledge/{framework}/{example}/{function_name}_coverage.json"

    # Build docker command
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_root}:{mnt}",
    ]

    # GPU passthrough for CUDA-based images (verl / open-r1) — optional.
    # Checks once per image. When GPU is unavailable, force CPU mode to avoid
    # CUDA initialization hangs during torch import.
    if framework in ("verl", "open-r1"):
        if _gpu_ready(image):
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
    else:
        exec_cmd = docker_cmd

    print(f"  Running coverage for {function_name} (timeout={timeout}s)...")
    try:
        proc = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {function_name}: exceeded {timeout}s, skipping")
        _save_error_coverage(framework, example, function_name,
                             f"Coverage measurement timed out after {timeout}s")
        return {
            "function_name": function_name,
            "line_coverage": None,
            "num_execution_errors": 1,
            "execution_errors": [{"error": f"Timed out after {timeout}s", "category": "timeout"}],
            "is_execution_failure": True,
        }

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
        _save_error_coverage(framework, example, function_name, stderr_tail)
        return {
            "function_name": function_name,
            "line_coverage": None,
            "num_execution_errors": 1,
            "execution_errors": [{"error": stderr_tail, "category": "execution_failure"}],
            "is_execution_failure": True,
        }

    # Read result
    out_path = coverage_result_path(framework, example, function_name)
    if not out_path.exists():
        print(f"  FAIL {function_name}: output not created")
        return None

    data = json.loads(out_path.read_text(encoding="utf-8"))
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
) -> list[dict]:
    """Run coverage for all (or one) candidate functions in an example.

    Args:
        native: If True, run coverage_runner.py directly on host instead of
            inside Docker. Useful when Docker is unavailable or when running
            on a GPU instance that already has all dependencies installed.
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

    # Load correct line ranges from JSONL once
    gt_locations = _load_gt_locations_from_jsonl(framework, example)
    if gt_locations:
        print(f"  Loaded {len(gt_locations)} GT locations from JSONL")

    results = []
    for c in candidates:
        fn = c["function_name"]
        data = run_coverage_single(
            framework, example, fn, per_test, capture_output,
            gt_locations=gt_locations, timeout=timeout,
            native=native,
        )
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
