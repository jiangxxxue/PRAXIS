You are a software engineer analyzing a codebase to find functions suitable for **practice** — functions that an agent can learn from by studying their inputs and outputs.

You have access to the `code/` directory.

## Important: Instance Code vs Framework Code

The `code/` directory may contain both **instance code** (the specific project/example you should focus on) and **framework code** (a larger library that the instance depends on). Your focus must be on the **instance code**. Framework-internal functions are NOT candidates — they belong to a general-purpose library, not this specific project.

## Important: Skip Abstract Stubs

Some functions have a body that consists ONLY of `raise NotImplementedError` (possibly preceded by a docstring). These are abstract stubs or interface definitions — they have no real implementation. **Do NOT select these functions.** They cannot be practiced because there is no concrete input→output behavior to learn from. Only select functions that have an actual implementation body.

## Selection Criteria

Select functions that are suitable for practice — meaning they have clear, observable input→output behavior:

- **Well-defined contract**: The function has clear parameter types and return values visible from its signature, docstring, and call sites
- **Testable logic**: The function performs a discrete, understandable computation — given specific inputs, you can determine expected outputs
- **Domain-relevant**: The function encodes logic specific to this project's domain, not generic boilerplate (config loading, logging, type conversion)
- **Non-trivial**: The function requires some understanding to implement correctly — not a one-line wrapper or simple delegation

## Output

Write a file named `candidate_functions.json` in the workspace root:

```json
[
  {
    "function_name": "function_name",
    "implementation_location": "code/path/to/file.py:line START-END",
    "rationale": "One sentence describing what this function does and why it is good for practice"
  }
]
```

**Rules:**
- Read actual source files to determine accurate line numbers
- Only include functions from the instance code, NOT framework-internal functions
- Only include functions with a real implementation (NOT abstract stubs with `raise NotImplementedError` body)
- Order by practice value: most suitable for learning first
