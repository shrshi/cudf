# CLAUDE.md

## Coding Style Preferences
Every line of code in a diff that isn't immediately and obviously necessary for achieving the stated goal destroys the reviewer's ability to reason about the change. This is not hyperbole—it's the fundamental constraint that determines whether code review enables or blocks progress.

### Core Principles
- *Simplicity first* - Always choose the simplest solution that works
- *Obviously correct* - Code should be self-evidently correct upon reading
- *DRY (Don't Repeat Yourself)* - Eliminate code duplication through abstraction
- *KISS (Keep It Simple, Stupid)* - Prefer simple solutions over complex ones; complexity should be justified by real needs
- *YAGNI (You Aren't Gonna Need It)* - Don't implement functionality until it's actually needed; avoid premature generalization
- *Diff discipline* - A tight, parsimonious diff builds trust and enables rapid review. Minimize diff noise by avoiding unrelated whitespace or stylistic changes; ensure every line in the diff serves the stated goal. **If you don't know what the stated goal is clearly, stop and ask**. Every modification, 100% of the time, is for the purpose of satisfying a stated, falsifiable goal.
- *Atomic Commits* - Our goal is not just to produce beautiful code, but beautiful commits as well. **Before committing, review your own diff**. For EVERY changed line ask: "Would a reviewer immediately understand this is required for [the stated goal]?" If the answer is anything but "obviously yes," that line must be removed or moved to a separate branch.

### Implementation Guidelines
- Write code that a new team member can understand immediately
- Prefer explicit over implicit behavior
- Use descriptive variable and function names
- Extract common patterns into reusable functions or modules. However, for one time operations, do not create helper functions.
- Don't optimize prematurely - simple, clear code is more valuable than fast code
- When in doubt, choose the approach that's easier to read and reason about
- *Be judicious with comments* - Only add comments when the code's purpose isn't obvious to a generalist reader

### Debugging Approach
Use systematic, hypothesis-driven debugging (not trial-and-error):
- *Read* the relevant code first - understand what it's supposed to do
- *Theorize* - come up with 2-3 theories about what might be wrong; present them to the user. If it's a particularly complicated problem, stop and ask for feedback before proceeding.
- *Verify* - test your theories with targeted investigation
- *Fix* - implement the solution once you've confirmed the root cause
- *Ask for help* if you're stuck after trying this process

Avoid jumping to conclusions or applying common fixes without understanding the specific problem. This is especially important for end-to-end test failures - if stuck, propose theories to the user along with ideas for information that could make the test easier to debug.

### Branch Cleanliness Protocol
- **Do exploratory work** - Try approaches, add debug logging, experiment freely
- **Identify the minimal solution** - Once working, determine exactly what's required
- **Create a clean branch** - Check out from main, apply ONLY the essential changes
- **Review your own diff** - Read it as if you know nothing about the problem
- **Verify independently** - Run tests to prove the minimal changes work

### Workslop
"Workslop" is AI-generated debris - plausible output that misses the mark. A 50-line diff when 5 would do destroys the reviewer's ability to reason about the change.
Don't introduce workslop.
If you're unsure what the goal is, stop and ask.
