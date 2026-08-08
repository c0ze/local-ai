# CLAUDE.md

These rules apply to every task in this repository unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

This file is generic and identical across projects. **It contains no project-specific
information.** `AGENTS.md` is a symlink to this file, so Claude Code, Codex, Kimi, Cursor
and anything else reading either name get the same rules.

## Where knowledge lives

Read in this order. Stop as soon as you have what you need.

| Layer | File | Contains |
|---|---|---|
| 1. Rules | `CLAUDE.md` = `AGENTS.md` (this file) | How to work. Generic, never project-specific. |
| 2. Project | [PROJECT.md](PROJECT.md) | This project: architecture, constraints, key files, how to run and validate. |
| 3. Wiki | `llm-wiki/index.md` (if present) | Compounding knowledge: decisions, entities, sources. |

**Before changing project behavior, read `PROJECT.md`.** It is the handoff document and
takes precedence over anything you infer from the code.

If a project has an `llm-wiki/`, it follows the Karpathy LLM-wiki pattern: raw sources are
compiled once into interlinked pages, and you query the wiki rather than re-deriving from
sources. Start at `index.md` and `agent-rules.md`. Append to `log.md` when you change it.
Treat archived pages as historical only — never cite them as current.

Keep the layers honest: a fact that is true of every project belongs here; a fact true of
this project belongs in `PROJECT.md`; a fact that took real work to establish belongs in
the wiki. Duplicating across layers is how they drift.

## Rule 1 — Think before coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 — Simplicity first
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

## Rule 4 — Goal-driven execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls
Use the model for: classification, drafting, summarization, extraction.
Do NOT use it for: routing, retries, deterministic transforms.
If code can answer, code answers.

## Rule 6 — Token budgets are not advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
If approaching budget, summarize and start fresh.
Surface the breach. Do not silently overrun.

## Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

## Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

## Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

## Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Rule 13 — Keep these documents current
When you learn something that contradicts `PROJECT.md`, fix `PROJECT.md` in the same
change. A stale handoff document is worse than none — the next agent will trust it.
Never edit `AGENTS.md`: it is a symlink to this file.
