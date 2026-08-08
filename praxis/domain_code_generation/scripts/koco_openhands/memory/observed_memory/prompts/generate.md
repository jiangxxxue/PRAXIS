You are writing task specifications for a software engineering agent.

You have access to the `code/` directory. The target function `{function_name}` is stubbed at `{implementation_location}` — only this function is stubbed; all other functions are complete.

Your task: produce two files.

**File 1**: `{function_name}_requirement.md` — a requirement document with these sections:

# FUNCTION: {function_name}

## Function Overview
Brief description of what this function does.

## Function Signature
The complete Python function signature.

## Input Parameters
For each parameter: name, type, shape/structure, and meaning.

## Detailed Description
Behavior of the function: what it computes, data transformations, edge cases, and numerical conventions.

## Output
Return value type, shape, and meaning.

---

Do NOT include implementation location or test code paths. Do NOT describe internal algorithm steps or name helper variables. Describe only the observable interface behavior.

**File 2**: `{function_name}_test_input.py` — test inputs as a Python file with `torch.Tensor` inputs:

```python
"""Test inputs for {function_name}."""
import torch

FUNCTION_NAME = "{function_name}"

f32 = torch.float32

def t(data, dtype=f32):
    return torch.tensor(data, dtype=dtype)

def case(**kwargs):
    """Build a test case dict with inputs, note, and expected_error."""
    inputs = {}
    meta = {}
    for k, v in kwargs.items():
        if k in ("note", "expected_error"):
            meta[k] = v
        else:
            inputs[k] = v
    result = {"inputs": inputs}
    result.update(meta)
    return result

test_cases = {{
    "normal": [
        case(param1=t([[0.1, 0.2]]), param2=t([1.0]), param3=t([[1, 1]]), param4=0.05),
        # ... at least 5 cases
    ],
    "edge": [
        case(param1=t([[0.0]]), param2=t([0.0]), param3=t([[0]]), param4=0.0, note="all zeros"),
        # boundary conditions: empty, zero-dim, extreme values
    ],
    "error": [
        {{
            "inputs": {{"param1": t([[0.1]]), "param2": t([1.0, 0.0])}},
            "expected_error": RuntimeError,
            "note": "batch size mismatch",
        }},
    ],
}}
```

Format rules:
- `FUNCTION_NAME`: the target function name as a string constant
- Use `t(data)` helper to create `torch.float32` tensors from nested lists — this preserves type information that JSON cannot represent
- Use `case(**kwargs)` helper for normal/edge cases: parameter names are keyword args, `note=` and `expected_error=` are optional metadata
- For error cases, use a dict literal with `"inputs"`, `"expected_error"` (exception class), and `"note"` keys
- `normal`: at least 5 representative inputs with valid parameter combinations, varying batch sizes and sequence lengths
- `edge`: boundary conditions (all-zeros, zero beta, large beta, asymmetric masks, minimum shape, etc.)
- `error`: inputs that should raise exceptions (shape mismatches, out-of-range values, wrong types)
- Each `case()` call must pass ALL function parameters as keyword arguments, matching the function signature exactly
- Do not pass helper state such as `scenario`, `mode`, or mock configuration through `case()`
- For class methods with no parameters beyond `self`, configure per-case behavior through `setup_self(config=None)` or fixed mock state, not `case()` inputs
- Add `note=` to document the purpose of each test case
- `expected_error` should be the exception **class** (e.g., `RuntimeError`, `ValueError`), not a string

## Mock Infrastructure for Class Methods

If the target function is a class method (takes `self`), you can define a `setup_self(config=None)` function to construct a mock self object with the attributes and methods the function accesses:

```python
from unittest.mock import MagicMock, AsyncMock

def setup_self(config=None):
    """Construct a mock self with attributes the target method needs."""
    mock_self = MagicMock()
    mock_self.config = config
    mock_self.working_dir = "/tmp/test_workspace"
    # For async methods accessed via self:
    mock_self.rag = MagicMock()
    mock_self.rag.aquery = AsyncMock(return_value="test result")
    # For regular methods:
    mock_self._detect_existing_database = MagicMock(return_value=False)
    return mock_self
```

Available mock utilities (pre-injected, no import needed):
- `MagicMock` — auto-creates child attributes on access, records calls
- `AsyncMock` — async version, returns awaitable coroutines
- `mock` — the full `unittest.mock` module (for `mock.patch`, `mock.PropertyMock`, etc.)

When `setup_self()` is defined, the coverage runner will call it to construct `self` instead of the default bare mock. This allows class methods that access `self.xxx` attributes to execute past the first attribute access.

**When to use `setup_self()`**: Only when the target is a class method that accesses `self` attributes beyond `config`. For standalone functions, do not define `setup_self()`.

For module-level patching (e.g., mocking external API clients), define `setup_environment()`:

```python
def setup_environment():
    """Patch module-level dependencies before the function is imported."""
    from unittest import mock
    mock.patch("some_module.Client", MagicMock).start()
```

## Mock Verification (CRITICAL)

When writing `mock.patch("module.path.symbol", ...)` targets in `setup_environment()`:

1. **Verify the target exists BEFORE writing it.** Use the `terminal` tool to grep for the function/class name in `code/`:
   ```
   grep -rn "def symbol_name\|class symbol_name" code/
   ```
2. Only write `mock.patch(...)` when you are certain the module path AND the symbol both exist.
3. Wrong mock targets are the most common cause of test failure — the module path must be exactly correct.

## Import Restrictions

The test_input.py will be executed in a Docker environment. Only use:
- Python standard library modules
- Packages known to be installed in the evaluation image for this framework
- NEVER import from local project modules that are not part of the installed package (e.g., `from bookworm.library import ...` will fail because bookworm is not a pip-installed package)
- NEVER invent project-local imports such as `utils.model_wrapper`. If you need a local project module, first verify the exact path exists under `code/`; otherwise use mocks or stdlib helpers.
- Avoid repository discovery code based on `__file__`, parent directory walks, or guessed relative paths. The coverage runner already imports the target function; your test input should define data, mocks, `setup_environment()`, and optional `setup_self()` only.

Read the surrounding code to understand parameter types and expected behavior. Use `file_editor` to write both files.
