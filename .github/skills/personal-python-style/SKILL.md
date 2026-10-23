---
name: personal-python-style
description: Apply the user's Python coding style when writing, editing, refactoring, or reviewing Python code. Use for Python implementations and style feedback, including requests to follow the user's conventions.
---

# Personal Python Style

Apply these preferences when working with Python. First inspect applicable project instructions and nearby code. Follow those conventions where they conflict with these preferences, and mention meaningful conflicts rather than silently imposing a different style.

## Rules

- Keep code concise without sacrificing clarity. Avoid one-line function bodies. If a comprehension becomes too long after linting, use an intermediate variable.
- Use f-strings for string interpolation.
- Use `snake_case` for variables and functions, `UPPER_CASE` for module-level constants, and a leading underscore for private or hidden names.
- Follow the project's established ordering for variables, functions, classes, and methods. When no convention is clear, follow the closest related code rather than reordering code arbitrarily.
- Avoid duplicating logic, constants, or configuration. Reuse an existing helper or constant, or extract a shared one when the same code appears in more than one place.
- Write one-line comments that explain the next group of lines. Comment each logical block, even when its code is obvious, so blocks stay visibly separated and unrelated code isn't merged into one block. Skip comments only in very short functions. No trailing full stop. All lower case.
- Write one-line docstrings for functions and methods that describe purpose, not inputs or outputs. No trailing full stop.
- Keep each import statement on a single line when it fits within the formatter's line length. Let the formatter wrap longer imports, and don't leave trailing commas that keep imports wrapped when they would fit on one line.
- Avoid functions with only one call site when inlining keeps the code clear. Preserve meaningful public APIs, callbacks, lifecycle boundaries, and independently useful abstractions even if they have a single call site.
- Add type hints to function and method parameters and return types. Do not annotate local or module-level variables. Keep type hints to at most one nested level, such as `list[dict]`, not `list[dict[str, int]]`.
- Use modern typing syntax (PEP 604/585): `X | None` and `A | B` instead of `Optional` and `Union`, built-in generics such as `list`, `tuple`, `dict`, and `set` instead of `typing.List` and similar, and `collections.abc` for abstract types such as `Callable` and `Iterable`.
- Avoid unnecessary `try`/`except` blocks and type casts. Use narrow type-checker ignores when needed; don't add casts or exception handling merely to silence a diagnostic.
- Use single-letter names for temporary variables in comprehensions.

## Applying the Style

1. Check project instructions, formatter and linter configuration, and nearby code before editing.
2. Apply these rules to new or directly modified code. Don't reformat unrelated code just to make it match.
3. In reviews, give a few concrete examples and distinguish clear violations from preferences not specified here.
4. Run the narrowest relevant formatter, linter, type checker, or tests when available. Don't claim a check passed unless it was run.
