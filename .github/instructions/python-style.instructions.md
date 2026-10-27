---
description: "Apply the personal Python style skill when writing, editing, refactoring, or reviewing Python code, including notebook code cells."
applyTo: "**/*.py,**/*.ipynb"
---

# Python Style

- Before writing or editing Python code, read and apply [personal-python-style](../skills/personal-python-style/SKILL.md), even when the user has not explicitly requested the skill.
- Apply the same rules to Python notebook code cells. Preserve the notebook's existing content and outputs outside the requested changes.
- Treat the skill as the source of truth for personal style; do not infer conflicting style rules from existing code. Follow explicit project instructions first, as the skill requires.
- Before finishing, check new and modified code against the skill and correct any deviations. Keep unrelated code unchanged.
- For Python style reviews, load the skill before assessing the code and distinguish violations from unspecified preferences.