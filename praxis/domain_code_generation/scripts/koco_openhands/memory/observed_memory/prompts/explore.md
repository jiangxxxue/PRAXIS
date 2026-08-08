You are a software engineer studying a specific project within a larger codebase. The `code/` directory contains both the project's own source files and the underlying framework it depends on.

Your task is to understand **this project** — its purpose, how it works, and what someone would need to know to work on it. The framework code is available for reference, but your focus should be on the project's own code.

## What is "Observed Knowledge"

The document you produce is called **observed knowledge** — knowledge gained purely by *observing* the codebase, without being told which parts are important or will be tested later. You decide what matters by reading the code and identifying what is central to this project's functionality.

Write a document named `observed_knowledge.md` in the workspace root with the following sections:

## 1. Project Overview
What this project does, its role within the larger framework, and its core design. If the project is a recipe/example/extension, explain what problem it solves and how it relates to the framework.

## 2. Key Modules and Their Roles
The project's own directory structure. For each module: what it contains, its responsibility, and how it connects to other modules. Focus on the project's code — only mention framework modules when the project directly depends on them.

## 3. Core Functions and Their Relationships
The important functions and methods within this project. For each: its signature, what it computes, and which other functions it calls or is called by. Pay attention to the call chain — how functions pass data to each other. This is the most important section: a reader should understand the project's computational graph.

## 4. Data Flow
How data enters, moves through, and exits this project. Include concrete data structures (tensors, dicts, dataclasses), their shapes and types when visible from the code, and how they are transformed along the way.

## 5. Patterns and Conventions
Recurring code patterns within this project — how similar functions are structured, common parameter conventions, error handling style, and any project-specific idioms. Also note the coding conventions (naming, type annotations, imports) used in the project's files.

## Exploration Strategy (CRITICAL — follow this order)

You must explore the codebase in two passes. Do NOT skip pass 1.

**Pass 1 — Full structural scan (do this first, before reading any file in detail):**
1. Run `find code/ -type f -name "*.py" | sort` to list every Python file.
2. Run `find code/ -type d | sort` to list every directory.
3. From the file list, identify ALL logical modules/subsystems — every distinct
   area of functionality, no matter how deeply nested. A module at
   `code/a/b/c/d/module.py` is just as important as `code/top.py`.
4. Build a mental map: for each subsystem, note its purpose in one line.

**Pass 2 — Per-module documentation:**
For each subsystem identified in Pass 1:
1. Read key files in that subsystem (prioritize files with function/class definitions).
2. Extract a high-level summary: what this subsystem does, its core abstractions,
   key function signatures, and how it connects to other subsystems.
3. Move on to the next subsystem. Do NOT spend excessive iterations on any single
   subsystem — breadth of coverage is more important than depth of any one part.

**Do NOT:**
- Spend more than ~15% of your iterations on any single directory or subsystem.
- Get stuck reading repetitive boilerplate (e.g., many similar config/launch scripts).
- Skip a deeply nested directory because it "looks similar" to something you've already seen.

## Guidelines
- **Hidden benchmark targets**: Some function bodies are intentionally replaced by `raise NotImplementedError`. Treat those functions as opaque interfaces. You may document their signatures, docstrings, callers, and callees, but do not infer or reconstruct their hidden implementations.
- **Project first, framework second**: Describe the framework only when it helps understand the project. Don't document the framework's general API — document how this project *uses* it.
- **Concrete over abstract**: Specific function signatures, parameter names, and data shapes are more useful than abstract descriptions. "Takes a DataProto with `input_ids`, `attention_mask`" is better than "takes a DataProto."
- **Relationships matter**: Which function calls which, what data passes between them — these connections are as important as individual function descriptions.
- **Reference sources**: Mention specific file paths and function/class names so readers can locate the source.
- **No length limit**: Write as much as needed to cover every subsystem. A thorough document is always better than one that omits parts of the codebase.

Use `file_editor` to write the document. Do not guess or fabricate content.
