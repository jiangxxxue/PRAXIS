"""OpenHands agent logic for KOCO-bench code generation.

Uses the OpenHands SDK to run an agent that explores a repository and
implements a function.  Each invocation gets an isolated workspace copy
so agents cannot interfere with each other or pollute the source tree.

Provides: prompt construction, single-instance agent execution, JSONL I/O,
and resume helpers.
"""

import ast
import json
import os
import re
import shutil
import tempfile

from agent.sdk import (
    SDK_AVAILABLE as _SDK_AVAILABLE,
    ConversationExecutionStatus,
    ConversationRunError,
    run_sdk_agent,
    _resolve_llm_model,
)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    """Load records from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: str) -> None:
    """Save records to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_ids(path: str) -> set:
    """Load set of completed function names from a progress file."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_ids", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_completed_ids(ids: set, path: str) -> None:
    """Save set of completed function names to a progress file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"completed_ids": sorted(ids)}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _collect_gt_locations(records):
    """Collect GT function line ranges from all records for the example.

    Returns: dict mapping file paths (relative to code/) to lists of
             (start_line, end_line) tuples (1-indexed, inclusive).
    """
    locations = {}
    for r in records:
        impl_loc = r.get("implementation_location", "")
        if not impl_loc:
            continue
        # Format: "code/path/to/file.py:line 86-87"
        # or with backslashes: "code\\path\\to\\file.py:line 86-87"
        parts = impl_loc.split(":line ")
        if len(parts) != 2:
            continue
        file_part = parts[0].replace("\\", "/")
        if file_part.startswith("code/"):
            file_part = file_part[len("code/"):]
        try:
            start_s, end_s = parts[1].split("-")
            locations.setdefault(file_part, []).append((int(start_s), int(end_s)))
        except ValueError:
            continue
    return locations


def _stub_one_function(lines, start, end):
    """Replace a single function body with a stub, keeping signature + docstring.

    ``start`` and ``end`` are 1-indexed inclusive line numbers covering the
    entire function (signature through last body line).  Returns a new list
    of lines with the body replaced by ``raise NotImplementedError``.
    """
    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # AST parse failed — fall back to regex-based stubbing
        return _stub_one_function_regex(lines, start, end)

    # Walk AST to find the FunctionDef/AsyncFunctionDef whose lineno is in [start, end]
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if start <= node.lineno <= end:
                target_node = node
                # Don't break — a nested def closer to ``start`` may exist,
                # but the outermost one matching is what we want.  Actually
                # we want the one whose lineno is closest to ``start``.
                if node.lineno == start:
                    break

    if target_node is None:
        # No matching function found — fall back to regex
        return _stub_one_function_regex(lines, start, end)

    node = target_node
    body = node.body
    replacement_end = max(end, int(node.end_lineno or end))

    # Determine where the stub should start (after signature / docstring)
    has_docstring = (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )

    if has_docstring:
        # Keep the docstring; stub starts after it
        stub_start = body[0].end_lineno  # 1-indexed, inclusive
        # Use indentation of the second body element if available, else infer
        if len(body) > 1:
            indent = body[1].col_offset
        else:
            indent = body[0].col_offset
    else:
        # Stub replaces entire body
        stub_start = body[0].lineno  # 1-indexed
        indent = body[0].col_offset

    # Build a line-count-preserving stub so graph locations and every function
    # after the target retain their original source coordinates.
    indent_str = " " * indent
    if has_docstring:
        replace_start = stub_start
    else:
        replace_start = stub_start - 1
    replace_count = max(1, replacement_end - replace_start)
    if replace_count == 1:
        stub_lines = [f"{indent_str}raise NotImplementedError\n"]
    else:
        stub_lines = [
            f"{indent_str}# TODO: implement this function\n",
            f"{indent_str}raise NotImplementedError\n",
        ]
        stub_lines.extend("\n" for _ in range(replace_count - 2))
    return lines[:replace_start] + stub_lines + lines[replacement_end:]


def _stub_one_function_regex(lines, start, end):
    """Regex fallback for _stub_one_function when AST parsing fails."""
    # Find the def line within [start, end]
    def_idx = None
    for i in range(start - 1, min(end, len(lines))):
        if re.match(r'\s*(async\s+)?def\s+', lines[i]):
            def_idx = i
            break

    if def_idx is None:
        # Can't find def — leave unchanged
        return lines

    # Infer body indent = def indent + 4
    def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())
    body_indent = def_indent + 4
    indent_str = " " * body_indent

    signature_end = def_idx
    paren_depth = 0
    for index in range(def_idx, min(end, len(lines))):
        line = lines[index]
        paren_depth += line.count("(") + line.count("[") + line.count("{")
        paren_depth -= line.count(")") + line.count("]") + line.count("}")
        signature_end = index
        if paren_depth <= 0 and line.rstrip().endswith(":"):
            break

    body_start = signature_end + 1
    replace_count = max(1, end - body_start)
    if replace_count == 1:
        stub_lines = [f"{indent_str}raise NotImplementedError\n"]
    else:
        stub_lines = [
            f"{indent_str}# TODO: implement this function\n",
            f"{indent_str}raise NotImplementedError\n",
        ]
        stub_lines.extend("\n" for _ in range(replace_count - 2))
    return lines[:body_start] + stub_lines + lines[end:]


def _stub_gt_functions(code_dst, gt_locations):
    """Replace GT function bodies with stubs, keeping signature + docstring."""
    for rel_path, ranges in gt_locations.items():
        file_path = os.path.join(code_dst, rel_path)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Process in reverse order so line numbers stay valid
        for start, end in sorted(ranges, reverse=True):
            lines = _stub_one_function(lines, start, end)
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def _prepare_workspace(workspace_root, knowledge_corpus_root, gt_locations, tmp_dir,
                       observed_knowledge_root=None, framework=None, example=None):
    """Copy workspace + knowledge_corpus, strip GT function bodies and test_code.

    Both directories are placed under ``tmp_dir/workspace/`` so the agent's
    cwd can be set there and ``ls .`` shows exactly ``code/`` and
    ``knowledge_corpus/`` plus optional single-example observed knowledge.

    Returns a dict with absolute paths for workspace, knowledge_corpus, code,
    and observed_knowledge/observed_knowledge_file when available.
    """
    ws_dir = os.path.join(tmp_dir, "workspace")
    os.makedirs(ws_dir, exist_ok=True)

    # Copy knowledge_corpus
    kc_dst = os.path.join(ws_dir, "knowledge_corpus")
    shutil.copytree(knowledge_corpus_root, kc_dst, symlinks=True)

    # Copy code/ excluding test_code and caches
    code_dst = os.path.join(ws_dir, "code")
    def _ignore(_dir, contents):
        return {c for c in contents if c in ("test_code", "__pycache__", ".pytest_cache")}
    shutil.copytree(workspace_root, code_dst, symlinks=True, ignore=_ignore)

    # Replace GT function bodies with stubs (signature + docstring kept)
    _stub_gt_functions(code_dst, gt_locations)

    paths = {"workspace": ws_dir, "knowledge_corpus": kc_dst, "code": code_dst}

    if observed_knowledge_root and os.path.isdir(observed_knowledge_root) and framework and example:
        source_file = os.path.join(observed_knowledge_root, framework, f"{example}.md")
        if not os.path.exists(source_file):
            return paths

        ok_dst = os.path.join(ws_dir, "observed_knowledge")
        ok_framework_dir = os.path.join(ok_dst, framework)
        os.makedirs(ok_framework_dir, exist_ok=True)
        ok_file = os.path.join(ok_framework_dir, f"{example}.md")
        shutil.copy2(source_file, ok_file)
        paths["observed_knowledge"] = ok_dst
        paths["observed_knowledge_file"] = ok_file

    return paths


def _parse_impl_location(impl_loc: str):
    """Parse 'code/path/to/file.py:line 58-133' or 'code/path/file.py:21-24'.

    ``rel_path`` is relative to the code/ directory (the ``code/`` prefix is
    stripped).  Returns ``(None, 0, 0)`` on parse failure.
    """
    if ":line " in impl_loc:
        file_part, range_part = impl_loc.split(":line ", 1)
    elif ":" in impl_loc:
        file_part, range_part = impl_loc.rsplit(":", 1)
    else:
        return None, 0, 0
    file_part = file_part.replace("\\", "/")
    if file_part.startswith("code/"):
        file_part = file_part[len("code/"):]
    try:
        start_s, end_s = range_part.split("-")
        return file_part, int(start_s), int(end_s)
    except ValueError:
        return None, 0, 0


from memory.config import MEMORY_RUN_ID, OBSERVED_KNOWLEDGE_DIR

_OBSERVED_KNOWLEDGE_ROOT = str(
    OBSERVED_KNOWLEDGE_DIR / MEMORY_RUN_ID
    if MEMORY_RUN_ID
    else OBSERVED_KNOWLEDGE_DIR
)

_GRAPH_KNOWLEDGE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "memory", "derived", "graph_knowledge",
)


def _default_graph_knowledge_path(
    framework: str,
    example: str,
    graph_knowledge_profile: str = "",
    graph_knowledge_artifact: str = "auto",
) -> str:
    """Return the optimized graph knowledge path for an example."""

    graph_dir = (
        os.path.join(_GRAPH_KNOWLEDGE_ROOT, graph_knowledge_profile, framework, example)
        if graph_knowledge_profile
        else os.path.join(_GRAPH_KNOWLEDGE_ROOT, framework, example)
    )
    optimized = os.path.join(graph_dir, "dep_graph.with_knowledge.optimized.json")
    mounted = os.path.join(graph_dir, "dep_graph.with_knowledge.json")
    if graph_knowledge_artifact == "optimized":
        return optimized
    if graph_knowledge_artifact == "mounted":
        return mounted
    if os.path.exists(optimized):
        return optimized
    return mounted


def _build_initial_graph_knowledge_context(
    graph_knowledge_path: str,
    function_name: str,
    min_confidence: float | None = None,
) -> str:
    """Build auto-injected one-hop caller knowledge context, if available."""

    if not graph_knowledge_path or not os.path.exists(graph_knowledge_path):
        return ""
    try:
        from memory.graph_knowledge_retriever import build_initial_caller_knowledge_context

        return build_initial_caller_knowledge_context(
            graph_knowledge_path=graph_knowledge_path,
            function_name=function_name,
            min_confidence=min_confidence,
        )
    except Exception as exc:
        print(f"    [{function_name}] Warning: graph knowledge context failed: {exc}")
        return ""


def _observed_knowledge_prompt(repo_paths):
    """Build the prompt bullet for observed knowledge when present."""

    observed_knowledge_file = repo_paths.get("observed_knowledge_file")
    if not observed_knowledge_file:
        return ""
    return (
        f"- Observed Knowledge Summary: {observed_knowledge_file}\n"
        "  This is precomputed project-level observed knowledge for this benchmark "
        "example: it summarizes repository structure, important APIs, and "
        "observed domain conventions. Use it for orientation, then verify "
        "details in the source code when implementing.\n"
    )


def build_prompt(record: dict, framework: str, repo_paths: dict,
                 stub_file: str = "", stub_line: int = 0,
                 graph_knowledge_context: str = "") -> str:
    """Build the task prompt for the OpenHands headless agent.

    The agent runs with cwd set to the workspace directory which contains
    ``code/`` and ``knowledge_corpus/`` as its only children.
    ``repo_paths`` has keys ``workspace``, ``knowledge_corpus``, and ``code``.
    """
    function_name = record["function_name"]

    # Extract system/user context from the pre-built prompt
    system_context = ""
    user_task = ""
    if record.get("prompt") and isinstance(record["prompt"], list):
        for msg in record["prompt"]:
            if msg.get("role") == "system":
                system_context = msg["content"]
            elif msg.get("role") == "user":
                user_task = msg["content"]

    # Build context section about the stub location
    stub_context = ""
    if stub_file:
        stub_context = f"""
IMPORTANT CONTEXT:
- The function stub is at: {stub_file} (near line {stub_line})
  It currently has `raise NotImplementedError` as a placeholder.
- The file already has imports for commonly used modules.
  Read the file header before adding any new imports.
"""

    observed_knowledge_hint = _observed_knowledge_prompt(repo_paths)
    graph_context = f"\n{graph_knowledge_context}\n" if graph_knowledge_context else ""

    return f"""You are working in a repository for the {framework} framework.
{system_context}

TASK: Implement the function `{function_name}`.

{user_task}

You can freely explore the following known repositories to obtain the required information:
- Framework Knowledge Base: {repo_paths["knowledge_corpus"]}
- Development Repository: {repo_paths["code"]}
{observed_knowledge_hint}

Please use the code in these repositories to implement the required functionality.
{stub_context}
{graph_context}
INSTRUCTIONS:
1. Explore the repositories to understand the codebase, domain knowledge, and callable functions.
2. Read {stub_file or "the source file"} to see the existing imports, class structure, and function signature.
3. Replace the `raise NotImplementedError` with your implementation using the file_editor tool.

RULES:
- Do NOT modify any other functions or code outside the target function.
- Do NOT run tests. Do NOT create helper scripts. Do NOT debug.
"""


# ---------------------------------------------------------------------------
# Single-instance agent execution
# ---------------------------------------------------------------------------

def _extract_from_events(events, function_name: str) -> str:
    """Fallback: scan SDK conversation events for file_editor create actions
    and extract the function body for ``function_name``.

    ``events`` is a list of SDK event objects (from ``conversation.state.events``).
    """
    import re as _re

    # Collect file content from file_editor create actions (newest last)
    created_files = {}
    for evt in events:
        try:
            # SDK events are pydantic models; check for file_editor actions
            if not hasattr(evt, "tool_name") or evt.tool_name != "file_editor":
                continue
            action = getattr(evt, "action", None)
            if action is None:
                continue
            # action may be a pydantic model or dict
            cmd = action.get("command") if isinstance(action, dict) else getattr(action, "command", None)
            path = action.get("path") if isinstance(action, dict) else getattr(action, "path", None)
            text = action.get("file_text") if isinstance(action, dict) else getattr(action, "file_text", None)
            if cmd == "create" and text and path:
                created_files[path] = text
        except Exception:
            continue

    if not created_files:
        return ""

    # Look for a file containing the target function
    for path, content in reversed(list(created_files.items())):
        if function_name in content:
            leaf_name = function_name.rsplit(".", 1)[-1]
            pattern = _re.compile(
                rf'^((?:async\s+)?def\s+{_re.escape(leaf_name)}\s*\(.*?\)\s*.*?:\s*\n)'
                r'((?:(?:[ \t]+.+|[ \t]*#.+|[ \t]*)\n)*)',
                _re.MULTILINE,
            )
            m = pattern.search(content)
            if m:
                signature = m.group(1)
                body = m.group(2)
                lines = body.rstrip('\n').split('\n')
                if lines:
                    indent = len(lines[0]) - len(lines[0].lstrip())
                    body = '\n'.join(l[indent:] for l in lines)
                print(f"    [{function_name}] Fallback: extracted from SDK events ({os.path.basename(path)})")
                return signature + body

    return ""


def _sanitize_completion(code: str, function_name: str) -> str:
    """Clean up agent output to plain Python, enforced in code rather than prompt.

    Handles common agent output issues:
    - Double-escaped newlines (literal \\n instead of real newlines)
    - JSON wrapping ({"implementation": "..."})
    - Markdown fences (```python ... ```)
    """
    if not code or not code.strip():
        return code

    # 1. Unwrap JSON (agent wrote {"implementation": "..."} or similar)
    #    Must run before escape-fixing so json.loads handles escapes correctly.
    stripped = code.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                for key in ("implementation", "code", "function", function_name):
                    if key in obj and isinstance(obj[key], str):
                        code = obj[key]
                        print(f"    [{function_name}] Sanitize: unwrapped JSON key '{key}'")
                        break
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Fix double-escaped newlines (literal \n instead of real newlines)
    if '\n' not in code and '\\n' in code:
        code = (code
                .replace('\\n', '\n')
                .replace('\\t', '\t')
                .replace('\\"', '"'))
        print(f"    [{function_name}] Sanitize: fixed escaped newlines")

    # 3. Strip markdown fences
    stripped = code.strip()
    for fence in ('```python', '```py', '```'):
        if stripped.startswith(fence):
            code = stripped[len(fence):]
            end = code.rfind('```')
            if end != -1:
                code = code[:end]
            code = code.strip()
            print(f"    [{function_name}] Sanitize: stripped markdown fences")
            break

    return code


def _record_callable_name(record: dict) -> str:
    """Return the source callable name, which may differ from the benchmark label."""

    benchmark_name = str(record.get("function_name") or "")
    signature = str(record.get("function_signature") or "")
    match = re.search(r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", signature)
    if not match:
        return benchmark_name

    signature_name = match.group(1)
    if benchmark_name.rsplit(".", 1)[-1] == signature_name:
        return benchmark_name
    return signature_name


def _extract_function_from_file(file_path, function_name):
    """Extract implemented function from a modified source file.

    Parses ``file_path`` with AST, locates the function (or method) named
    ``function_name``, and returns its full source text.  For dotted names
    like ``ClassName.method``, walks ``ClassDef`` → ``FunctionDef``.

    Returns an empty string if the file cannot be parsed, the function is
    not found, or the body is still the ``raise NotImplementedError`` stub.
    """
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines(keepends=True)
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ""

    # Support dotted names: "ClassName.method_name"
    parts = function_name.split(".")
    if len(parts) == 2:
        class_name, method_name = parts
    else:
        class_name, method_name = None, function_name

    target = None
    for node in ast.walk(tree):
        if class_name:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            target = item
                            break
        else:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    target = node
                    break

    # Fallback: dotted name may be "module.function" rather than
    # "Class.method".  If no class was found, retry as a top-level function.
    if target is None and class_name:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    target = node
                    break

    if target is None:
        return ""

    # Check if body is still the stub
    body = target.body
    # Skip docstring if present
    real_body = body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        real_body = body[1:]
    if (len(real_body) == 1
            and isinstance(real_body[0], ast.Raise)
            and isinstance(real_body[0].exc, ast.Name)
            and real_body[0].exc.id == "NotImplementedError"):
        return ""

    # Extract full function text (from first decorator or def line to end_lineno)
    start = target.lineno
    if target.decorator_list:
        start = target.decorator_list[0].lineno
    end = target.end_lineno
    func_text = "".join(lines[start - 1 : end])

    # Dedent to remove any class-level indentation
    import textwrap
    func_text = textwrap.dedent(func_text)
    return func_text.strip()



def _preserve_debug_artifacts(tmp_dir, function_name, framework, example,
                              sdk_events=None):
    """Copy agent logs and workspace snapshot on failure for post-mortem.

    Saved to ``scripts/data/{framework}/openhands/debug/{example}/``.
    """
    debug_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        "data", framework, "openhands", "debug", example, function_name,
    )
    debug_dir = os.path.normpath(debug_dir)
    try:
        os.makedirs(debug_dir, exist_ok=True)

        # 1. task prompt
        prompt_src = os.path.join(tmp_dir, "task_prompt.txt")
        if os.path.exists(prompt_src):
            shutil.copy2(prompt_src, os.path.join(debug_dir, "task_prompt.txt"))

        # 2. modified stub file (the code/ tree after agent ran)
        code_dir = os.path.join(tmp_dir, "workspace", "code")
        if os.path.isdir(code_dir):
            dst = os.path.join(debug_dir, "code_snapshot")
            if os.path.exists(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(code_dir, dst, symlinks=True,
                            ignore=lambda d, c: {"__pycache__", ".pytest_cache"} & set(c))

        # 3. SDK conversation events (replaces openhands.log / oh_events)
        if sdk_events:
            events_file = os.path.join(debug_dir, "sdk_events.json")
            try:
                serialized = []
                for evt in sdk_events:
                    if hasattr(evt, "model_dump"):
                        serialized.append(evt.model_dump(mode="json"))
                    else:
                        serialized.append(str(evt))
                with open(events_file, "w", encoding="utf-8") as f:
                    json.dump(serialized, f, indent=2, default=str)
            except Exception as exc:
                print(f"    [{function_name}] Warning: failed to serialize SDK events: {exc}")

        print(f"    [{function_name}] Debug artifacts saved to {debug_dir}")
    except Exception as exc:
        print(f"    [{function_name}] Warning: failed to save debug artifacts: {exc}")


def _safe_log_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value).strip()).strip(" .")
    return text or "function"


def _event_to_json(event):
    if hasattr(event, "model_dump"):
        try:
            return event.model_dump(mode="json")
        except TypeError:
            return event.model_dump()
    if isinstance(event, dict):
        return event
    return getattr(event, "__dict__", {"raw": str(event)})


def _event_field(event, field: str, default=None):
    if isinstance(event, dict):
        return event.get(field, default)
    return getattr(event, field, default)


def _action_payload(action):
    if action is None:
        return {}
    if isinstance(action, dict):
        return action
    if hasattr(action, "model_dump"):
        try:
            return action.model_dump(mode="json")
        except TypeError:
            return action.model_dump()
    return getattr(action, "__dict__", {"raw": str(action)})


def _observation_text(observation) -> str:
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        for key in ("content", "text", "stdout", "stderr", "output"):
            value = observation.get(key)
            if value:
                return str(value)
        return json.dumps(observation, ensure_ascii=False, default=str)
    for key in ("content", "text", "stdout", "stderr", "output"):
        value = getattr(observation, key, None)
        if value:
            return str(value)
    return str(observation)


def _thought_text(event) -> str:
    for field in ("thought", "thought_text", "content"):
        value = _event_field(event, field, "")
        if value:
            return str(value)
    return ""


def _extract_knowledge_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    markers = (
        "GRAPH KNOWLEDGE",
        "Graph knowledge",
        "Relevant graph knowledge",
        "Practice memory",
        "Observed Knowledge",
    )
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            return True, text[idx: idx + 4000]
    return False, ""


_KNOWLEDGE_ID_RE = re.compile(r"Knowledge-ID:\s*([^\]\s]+)")


def _normalize_knowledge_id(value) -> str:
    knowledge_id = str(value or "").strip()
    while knowledge_id.endswith(("\\n", "\\r", "\\t")):
        knowledge_id = knowledge_id[:-2].rstrip()
    return knowledge_id


def _tool_trace_from_events(events) -> list[dict]:
    rows = []
    for idx, event in enumerate(events or []):
        tool_name = _event_field(event, "tool_name", "")
        action = _action_payload(_event_field(event, "action", None))
        observation = _event_field(event, "observation", None)
        observation_text = _observation_text(observation)
        if not tool_name and not action and not observation_text:
            continue
        has_knowledge, knowledge_text = _extract_knowledge_text(observation_text)
        knowledge_ids = sorted({
            knowledge_id
            for raw_id in _KNOWLEDGE_ID_RE.findall(observation_text)
            if (knowledge_id := _normalize_knowledge_id(raw_id))
        })
        rows.append({
            "event_index": idx,
            "tool_name": str(tool_name or ""),
            "thought_text": _thought_text(event),
            "action": action,
            "observation_text": observation_text[:12000],
            "has_graph_knowledge": bool(has_knowledge and "graph" in knowledge_text.lower()),
            "has_practice_knowledge": bool(has_knowledge and "practice" in knowledge_text.lower()),
            "has_observed_knowledge": bool(has_knowledge and "observed" in knowledge_text.lower()),
            "knowledge_text": knowledge_text,
            "knowledge_ids": knowledge_ids,
            "is_error": "error" in observation_text.lower() or "traceback" in observation_text.lower(),
        })
    return rows


def _write_agent_logs(output_dir, example, function_name, prompt_file, sdk_events):
    """Write target-practice agent logs for every infer run."""

    if not output_dir:
        return
    log_dir = os.path.join(
        str(output_dir),
        "agent_logs",
        example,
        _safe_log_component(function_name),
    )
    os.makedirs(log_dir, exist_ok=True)
    try:
        if prompt_file and os.path.exists(prompt_file):
            shutil.copy2(prompt_file, os.path.join(log_dir, "task_prompt.txt"))
        with open(os.path.join(log_dir, "sdk_events.json"), "w", encoding="utf-8") as f:
            json.dump([_event_to_json(evt) for evt in (sdk_events or [])], f, indent=2, default=str)
        with open(os.path.join(log_dir, "tool_trace.jsonl"), "w", encoding="utf-8") as f:
            for row in _tool_trace_from_events(sdk_events or []):
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"    [{function_name}] Warning: failed to save agent logs: {exc}")


def run_single_instance(
    record: dict,
    framework: str,
    example: str,
    workspace_root: str,
    knowledge_corpus_root: str,
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_iterations: int = 50,
    debug: bool = False,
    graph_knowledge_profile: str = "",
    graph_knowledge_artifact: str = "auto",
    graph_knowledge_min_confidence: float | None = None,
    output_dir: str = "",
) -> dict:
    """Run the OpenHands SDK agent for one function.

    Steps:
      1. copy workspace to a temp directory for isolation
      2. build the task prompt
      3. run the SDK agent (LLM + tools) in the workspace
      4. extract the implementation from the modified source file
      5. clean up

    Returns the *record* dict augmented with ``completions`` and ``status``.
    The ``completions`` field is always a list with exactly one string:
    the implementation code on success, or an empty string on failure.
    This ensures failed attempts are counted in the evaluation denominator.
    """
    function_name = record["function_name"]
    print(f"    [{function_name}] Starting agent...")

    if not _SDK_AVAILABLE:
        print(f"    [{function_name}] Error: openhands-sdk not installed.")
        print("      Install with: uv pip install openhands-sdk openhands-tools --python 3.12")
        record["completions"] = [""]
        record["status"] = "error"
        record["error"] = "openhands-sdk not installed"
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
        return record

    tmp_dir = tempfile.mkdtemp(prefix=f"oh_{function_name}_")
    conv_events = []

    try:
        # Hide every benchmark target before the agent explores the repository.
        # This prevents cross-target answer leakage; signatures and docstrings
        # remain available for structural reasoning.
        impl_loc = record.get("implementation_location", "")
        my_rel_path, my_start, my_end = _parse_impl_location(impl_loc)
        from memory.observed_memory.workspace import benchmark_target_locations

        my_gt_locations = benchmark_target_locations(framework, example)
        repo_paths = _prepare_workspace(
            workspace_root,
            knowledge_corpus_root,
            my_gt_locations,
            tmp_dir,
            observed_knowledge_root=_OBSERVED_KNOWLEDGE_ROOT,
            framework=framework,
            example=example,
        )
        work_dir = repo_paths["workspace"]

        # Compute stub file path and line for the prompt (reuse parsed location)
        stub_file = os.path.join(repo_paths["code"], my_rel_path) if my_rel_path else ""
        stub_start = my_start

        graph_knowledge_path = _default_graph_knowledge_path(
            framework,
            example,
            graph_knowledge_profile=graph_knowledge_profile,
            graph_knowledge_artifact=graph_knowledge_artifact,
        )
        graph_knowledge_enabled = os.path.exists(graph_knowledge_path)
        if graph_knowledge_profile and not graph_knowledge_enabled:
            raise FileNotFoundError(
                "profile-scoped graph knowledge file not found: "
                f"{graph_knowledge_path}"
            )
        if graph_knowledge_artifact != "auto" and not graph_knowledge_enabled:
            raise FileNotFoundError(
                f"requested graph knowledge artifact not found: {graph_knowledge_path}"
            )
        graph_knowledge_context = _build_initial_graph_knowledge_context(
            graph_knowledge_path=graph_knowledge_path,
            function_name=function_name,
            min_confidence=graph_knowledge_min_confidence,
        )
        if graph_knowledge_context:
            print(f"    [{function_name}] Injected graph caller knowledge into prompt")

        prompt = build_prompt(
            record,
            framework,
            repo_paths,
            stub_file=stub_file,
            stub_line=stub_start,
            graph_knowledge_context=graph_knowledge_context,
        )

        # Save prompt for debug artifacts
        prompt_file = os.path.join(tmp_dir, "task_prompt.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        # --- Run SDK agent ---
        print(f"    [{function_name}] Running SDK agent (model={_resolve_llm_model(model, base_url)}) ...")

        conv_events, conv_status = run_sdk_agent(
            prompt=prompt,
            workspace=work_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
            # Stage 4 target inference keeps knowledge_search disabled.
            # Stage 1/2 practice agents still pass corpus_dirs explicitly.
            corpus_dirs=None,
            graph_knowledge_path=graph_knowledge_path if graph_knowledge_enabled else None,
            graph_knowledge_min_confidence=graph_knowledge_min_confidence,
        )

        print(f"    [{function_name}] SDK status: {conv_status.value}")

        # --- Extract result ---
        implementation = ""
        callable_name = _record_callable_name(record)

        # Primary: extract the target function from the modified source file
        if stub_file and os.path.exists(stub_file):
            implementation = _extract_function_from_file(stub_file, callable_name)
            if implementation:
                print(f"    [{function_name}] Extracted from modified source file")

        # Fallback 1: read implementation_result.py (agent may still create it)
        if not implementation:
            result_py = os.path.join(repo_paths["code"], "implementation_result.py")
            if os.path.exists(result_py):
                with open(result_py, "r", encoding="utf-8") as f:
                    implementation = f.read().strip()

        # Fallback 2: scan SDK conversation events for file_editor actions
        if not implementation:
            implementation = _extract_from_events(conv_events, callable_name)

        # Sanitize: fix escaping, unwrap JSON/markdown
        implementation = _sanitize_completion(implementation, function_name)

        if implementation:
            record["completions"] = [implementation]
            record["status"] = "success"
            print(f"    [{function_name}] Success ({len(implementation)} chars)")
        elif conv_status == ConversationExecutionStatus.STUCK:
            record["completions"] = [""]
            record["status"] = "stuck"
            record["results"] = [False]
            record["pass_ratios"] = [0.0]
            print(f"    [{function_name}] Agent stuck")
        elif conv_status == ConversationExecutionStatus.ERROR:
            record["completions"] = [""]
            record["status"] = "max_iterations"
            record["results"] = [False]
            record["pass_ratios"] = [0.0]
            print(f"    [{function_name}] Reached max iterations ({max_iterations})")
        else:
            record["completions"] = [""]
            record["status"] = "no_result"
            record["results"] = [False]
            record["pass_ratios"] = [0.0]
            print(f"    [{function_name}] No implementation found")

    except ConversationRunError as e:
        print(f"    [{function_name}] SDK error: {e}")
        record["completions"] = [""]
        record["status"] = "error"
        record["error"] = str(e.original_exception)
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
    except Exception as e:
        print(f"    [{function_name}] Error: {e}")
        record["completions"] = [""]
        record["status"] = "error"
        record["error"] = str(e)
        record["results"] = [False]
        record["pass_ratios"] = [0.0]
    finally:
        _write_agent_logs(
            output_dir,
            example,
            function_name,
            os.path.join(tmp_dir, "task_prompt.txt"),
            conv_events,
        )
        status = record.get("status", "")
        if debug or status in ("no_result", "max_iterations", "stuck", "error"):
            _preserve_debug_artifacts(
                tmp_dir, function_name, framework, example,
                sdk_events=conv_events,
            )
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return record
