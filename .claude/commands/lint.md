---
description: Auto-fix linting issues and resolve remaining errors
allowed-tools: Bash(pre-commit run), Bash(pre-commit run --all-files), Bash(eslint)
model: haiku
---

# Lint and Fix

I have already run the linter with the fix flag. Here are the results:
Run the pre-commit hooks defined in `.pre-commit-config.yaml` only on files that were changed.

<lint_results>
!`pnpm run lint:fix || true`
</lint_results>

## Instructions
1. **Analyze the `<lint_results>`** above.
   - If the output shows "No errors" or is empty, respond with "No lint errors found." and stop - no further action needed.
   - If errors remain, they require manual intervention.
2. **Fix remaining errors** by editing the referenced files.
3. **Verify** by running the lint command one last time.
