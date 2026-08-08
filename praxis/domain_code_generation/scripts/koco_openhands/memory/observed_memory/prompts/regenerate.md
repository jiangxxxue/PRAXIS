You are fixing or improving test inputs for a Python function.

## Target Function: {function_name}
Location: {implementation_location}

{iteration_info}
{previous_error_feedback}
## Current Status: {coverage_stats}

## Source Code (with line numbers)
Lines marked with `[MISSING]` are NOT covered by current test inputs.
Lines marked with `[COVERED]` are already covered.
If no coverage markers appear (EXECUTION FAILURE), the test_input crashed before any lines could be traced — fix the crash first.

```
{annotated_source}
```

## Uncovered Lines
{missing_lines_summary}

## Execution Errors
{execution_errors_summary}

## Current Test Input (for reference)
```python
{current_test_input}
```

---

Your task: write a corrected `{function_name}_test_input.py`.

**If the Current Status says "EXECUTION FAILURE"** — the test_input crashed. Your PRIMARY goal is to fix the crash:

1. Read the Execution Errors section. Identify the root cause: syntax error, missing mock, wrong mock target, missing import, etc.
2. Fix the error. Common causes and fixes:
   - `mock.patch("X.Y")` failed → the target "X.Y" does not exist. Search the code for the real function/class name and use the correct path.
   - `ModuleNotFoundError: No module named 'X'` → X is not installed in the test environment. Replace with mock or stdlib alternative.
   - `No module named 'utils...'` or another guessed project path → do NOT invent local imports. Search code/ for the real module path, or remove the import and use mocks.
   - `NameError: __file__ is not defined` or path-discovery failures → remove repository-discovery logic. The coverage runner imports the target function; your test input should define data, mocks, `setup_environment()`, and optional `setup_self()` only.
   - `SyntaxError` → fix the Python syntax.
   - `AttributeError: 'X' object has no attribute 'Y'` → the mock is missing an attribute, add it to setup_self or the mock target.
3. Once execution succeeds, also try to cover the function's logic.

**If the Current Status shows coverage stats** — improve coverage:

1. Fix remaining execution errors first
2. Cover MISSING lines — add test cases for uncovered paths
3. Keep working cases — preserve passing test cases

**Coverage improvement tactics:**
- **Closures / inner functions**: if the function defines callbacks (e.g. `def llm_model_func(...)`) that are passed to other objects, your mock MUST call them to cover their bodies
- **Conditional branches**: `if/else` where only one path is covered — add cases for the other path
- **Exception handlers**: `except` blocks — add error cases that trigger them
- **Async code paths**: `await` calls whose results affect subsequent logic

## Mock Verification (CRITICAL)

When writing `mock.patch("module.path.symbol", ...)` in `setup_environment()`:

1. **Verify the target exists BEFORE writing it.** Use `terminal` to grep for the function/class name in `code/`:
   ```
   grep -rn "def symbol_name\|class symbol_name" code/
   ```
2. Only write `mock.patch(...)` when you are certain the module path AND the symbol both exist.
3. If the error traceback says a mock target doesn't exist, the module path is wrong — search for the correct one.

## Output Format

A single Python file `{function_name}_test_input.py`:
- `FUNCTION_NAME = "{function_name}"` string constant
- `t(data)` helper for `torch.float32` tensors
- `case(**kwargs)` helper for normal/edge cases
- `test_cases` dict with `"normal"`, `"edge"`, `"error"` keys
- Each `case()` must pass ALL function parameters as keyword arguments
- Do not pass helper state such as `scenario`, `mode`, or mock configuration through `case()`; class methods must keep this state in `setup_self()`
- For error cases, use dict literal with `"inputs"`, `"expected_error"`, `"note"` keys
- `expected_error` must be an exception class such as `ValueError`, never a string

## Mock Infrastructure

`MagicMock`, `AsyncMock`, and `mock` are available without imports.

For class methods, define `setup_self(config=None)`:
```python
def setup_self(config=None):
    mock_self = MagicMock()
    mock_self.config = config
    # Add attributes the function accesses (check error traceback for missing ones)
    mock_self.working_dir = "/tmp/test"
    return mock_self
```

For module-level patching, define `setup_environment()`:
```python
def setup_environment():
    from unittest import mock
    mock.patch("correct_module.ClassName", MagicMock).start()
```

Do not use `__file__`, parent-directory walking, or guessed project imports in
the corrected test input. If a module path is not verifiably present under
`code/`, mock the dependency instead of importing it.

Read the surrounding code. Use `file_editor` to write the file.
