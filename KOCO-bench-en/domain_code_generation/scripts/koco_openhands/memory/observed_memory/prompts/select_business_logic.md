You are a senior software engineer tasked with identifying functions that carry **domain-specific business logic** and are suitable for **practice** — learning through input→output behavior.

You have access to the `code/` directory.

## Important: Instance Code vs Framework Code

The `code/` directory may contain both **instance code** (the specific project/example you should focus on) and **framework code** (a larger library that the instance depends on). Your focus must be on the **instance code**. Framework-internal functions are NOT candidates — they belong to a general-purpose library, not this specific project.

## Important: Skip Abstract Stubs

Some functions have a body that consists ONLY of `raise NotImplementedError` (possibly preceded by a docstring). These are abstract stubs or interface definitions — they have no real implementation. **Do NOT select these functions.** They cannot be practiced because there is no concrete input→output behavior to learn from.

## Your Task

1. Explore the instance code and identify functions with a real implementation.
2. For each function, determine whether it primarily implements **domain-specific business logic** or is **generic/utility code**.
3. Select ONLY functions that carry domain-specific business logic AND are suitable for practice (clear input→output contract).

### Domain Business Logic — SELECT these

A function carries domain business logic if it:

- **Encodes domain rules or algorithms**: Contains formulas, heuristics, or computational procedures specific to this application's domain — rules that would be unfamiliar to someone outside the domain.
- **Non-trivial domain-dependent control flow**: Branching logic whose conditions depend on domain concepts and their relationships (not just null checks or type dispatch).
- **Core domain computation**: Implements a central computational or decision-making process that defines what this project does differently from generic frameworks.
- **Domain correctness**: Verifying correctness requires domain expertise, not just programming skill — the "why" behind the logic is tied to domain-specific requirements.
- **Encodes implicit domain knowledge**: Captures constraints, thresholds, or relationships that are documented in research papers or domain literature, not obvious from the code itself.

### Generic Code — DO NOT select these

A function is generic if it primarily:

- **Wraps or dispatches**: Routes calls to other functions without adding meaningful logic (thin wrappers, proxy methods, simple dispatchers).
- **Data plumbing**: Formats data, serializes/deserializes, performs type conversion, or reshapes tensors without domain-specific rules.
- **Infrastructure**: Loads configuration, initializes objects, sets up logging, collects metrics, manages connections.
- **Standard patterns**: Implements factory, builder, singleton, or other GoF patterns without embedding domain-specific decision logic.
- **Simple CRUD / accessors**: Basic data access, property getters/setters, simple filtering or sorting.
- **Glue code**: Integration layers (API endpoints, CLI handlers, event listeners) that delegate all logic to domain functions.

## Gray Area Guidance

- If a function mixes generic scaffolding with domain logic, **include it** if the domain logic is the non-trivial, error-prone part.
- If a function is a medium-complexity algorithm but uses only standard CS techniques (no domain-specific tuning), **exclude it** — it's a general algorithm, not domain logic.
- If unsure, ask: "Would a competent programmer unfamiliar with this domain be able to implement this correctly from a description?" If yes, it's likely generic. If no, it carries domain knowledge.

## Output

Write a file named `candidate_functions.json` in the workspace root:

```json
[
  {
    "function_name": "function_name",
    "implementation_location": "code/path/to/file.py:line START-END",
    "rationale": "One sentence describing what domain business logic this function implements and why it requires domain expertise"
  }
]
```

**Rules:**
- Read actual source files to determine accurate line numbers
- Only include functions from the instance code, NOT framework-internal functions
- Only include functions with a real implementation (NOT abstract stubs with `raise NotImplementedError` body)
- Only include functions that **primarily** carry domain-specific business logic
- Order by domain-specificity: most domain-dependent functions first
