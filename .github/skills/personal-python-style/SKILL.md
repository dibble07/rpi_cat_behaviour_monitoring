---
name: personal-python-style
description: Apply the user's Python coding style when writing, editing, refactoring, or reviewing Python code. Use for Python implementations and style feedback, including requests to follow the user's conventions.
---

# Personal Python Style

Apply these preferences when working with Python. First inspect applicable project instructions and nearby code. Follow those conventions where they conflict with these preferences, and mention meaningful conflicts rather than silently imposing a different style.

## Rules

- Use clear British English in authored prose, comments, docstrings, and identifiers. Use American spellings only when required by external names (for example, library APIs or data schemas), not merely because an internal name already uses them.
- Keep code concise and clear: use f-strings, avoid one-line function bodies and duplicated logic, and split long comprehensions into intermediate variables when clearer.
- Name variables and functions in `snake_case`, constants in `UPPER_CASE`, and private names with a leading underscore. Use single-letter names for temporary comprehension variables.
- Follow nearby code for ordering; do not reorder unrelated code. Keep imports on one line when they fit the formatter's limit, and avoid trailing commas that needlessly wrap them.
- Add parameter and return type hints to functions and methods, but not to local or module-level variables. Prefer modern built-in generics, `X | None`, and `collections.abc` types; limit nested type hints to one level.
- Avoid unnecessary exception handling and casts. Use narrow type-checker ignores rather than adding them just to silence diagnostics.
- Keep useful abstractions, public APIs, callbacks, and lifecycle boundaries even when they have one call site; inline other single-use functions when clearer.
- Write brief, one-line comments for logical blocks and one-line docstrings that describe a function's purpose. Comments should be lower case; neither comments nor docstrings need a full stop. Skip comments in very short functions.

## Applying the Style

1. Check project instructions, formatter and linter configuration, and nearby code before editing.
2. Apply these rules to new or directly modified code. Don't reformat unrelated code just to make it match.
3. In reviews, give a few concrete examples and distinguish clear violations from preferences not specified here.
4. Run the narrowest relevant formatter, linter, type checker, or tests when available. Don't claim a check passed unless it was run.
