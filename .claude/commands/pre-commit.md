---
description: Auto-fix linting issues and resolve remaining errors
argument-hint: [--all]
allowed-tools: Bash(pre-commit run), Bash(pre-commit run --all-files)
model: haiku
---

# Lint and Fix

Run the pre-commit hooks defined in `.pre-commit-config.yaml`. 

<pre_commit_staged>
!`pre-commit run`
</pre_commit_staged>

<pre_commit_all>
!`pre-commit run --all-files`
</pre_commit_all>

## Instructions
1. If the optional flag `--all` is provided in the $ARGUMENTS, then run `<pre_commit_all>` else run `<pre_commit_staged>`. If there changes due to formatting, then there would be unstaged files which need to be staged before committing.

## Conventions
* Always ensure `.pre-commit-config.yaml` is up to date.
* Run `pre-commit install` if not already installed.
