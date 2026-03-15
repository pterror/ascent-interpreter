# CLAUDE.md

Behavioral rules for Claude Code in the ascent-interpreter repository.

## Project Overview

Interpreter and JIT compiler for [Ascent](https://github.com/s-arash/ascent) Datalog programs.

## Architecture

<!-- Project-specific architecture notes -->

## Development

```bash
nix develop        # Enter dev shell
cargo test         # Run tests
cargo clippy       # Lint
cd docs && bun dev # Local docs
```

## Core Rules

**Note things down immediately — no deferral:**
- Problems, tech debt, issues → TODO.md now, in the same response
- Design decisions, key insights → docs/ or CLAUDE.md
- Future/deferred scope → TODO.md **before** writing any code, not after
- **Every observed problem → TODO.md. No exceptions.** Code comments and conversation mentions are not tracked items. If you write a TODO comment in source, the next action is to open TODO.md and write the entry.

**Conversation is not memory.** Anything said in chat evaporates at session end. If it implies future behavior change, write it to CLAUDE.md or a memory file immediately — or it will not happen.

**Warning — these phrases mean something needs to be written down right now:**
- "I won't do X again" / "I'll remember to..." / "I've learned that..."
- "Next time I'll..." / "From now on I'll..."
- Any acknowledgement of a recurring error without a corresponding CLAUDE.md or memory edit

**Triggers:** User corrects you, 2+ failed attempts, "aha" moment, framework quirk discovered → document before proceeding.

**When the user corrects you:** Ask what rule would have prevented this, and write it before proceeding. **"The rule exists, I just didn't follow it" is never the diagnosis** — a rule that doesn't prevent the failure it describes is incomplete; fix the rule, not your behavior.

**Something unexpected is a signal, not noise.** Surprising output, anomalous numbers, files containing what they shouldn't — stop and ask why before continuing. Don't accept anomalies and move on.

**When explaining why something behaves a certain way, state what evidence supports it.** If the evidence is "profiling data," cite it. If it's "reasoning from known costs," show the reasoning. If there isn't any, say so rather than rationalizing.

**Do the work properly.** Don't leave workarounds or hacks undocumented. When asked to analyze X, actually read X — don't synthesize from conversation.

## Performance Goal

**The JIT target is wall-clock parity with `ascent_macro` (LLVM-compiled Datalog).** This means optimizing for actual runtime, not a proxy metric. Instruction count matters, but so do cache misses, stall cycles, and memory layout. A change that reduces instructions while increasing cache pressure can be a regression.

When evaluating a JIT optimization, profile with `perf stat -e instructions,cycles,cache-misses` and compare *per-iteration* numbers against the `ascent_macro` benchmark. The gap is closed when both instruction count and cache behavior converge, not just one.

Current state (2026-03-16): fibonacci jit_hot/20 = 11.9µs vs ascent_macro 8.0µs (**1.49×**); triangle jit_hot/20 = 245µs vs ascent_macro 34µs (**7.2×**). The EDB contiguous index is already in place, reducing triangle cache misses from 116× to ~13× (estimated). Primary remaining bottleneck for triangle is the `JitHashIndex` pointer-chased structure — see TODO.md § "New architecture" for the compact JitColIndex replacement plan.

**ABI scan callbacks (`packed_data_ptr`, `packed_count`, `packed_recent_ptr`) are NOT the bottleneck.** These functions access warm cache lines (offsets 24, 72, 112 in `PackedStorage`). Any scheme that replaces them with direct loads risks adding cache misses from `PackedScanInfo` fields at cold offsets, or disrupts the overall layout. Attempted 2026-03-16 as `PackedScanInfo` in `PackedStorage` — regressed fibonacci 1.49× → 1.56–1.73× in all configurations. Do not retry.

**Parallelism must also be designed for max throughput.** When implementing parallel evaluation: zero shared state in the hot path (thread-local dedup tables, thread-local head buffers, read-only shared indices), batch merge only after all threads finish their scan. A parallel design that introduces lock contention or false sharing in the inner loop is worse than single-threaded. The test is: does the parallel version scale linearly with thread count on a join-heavy benchmark?

## JIT Design Rule

**Before implementing any JIT stage, trace the full hot-path and count Rust callbacks.**

If the inner loop still calls back into Rust — for any reason (reading tuple data, inserting, counting) — **stop and fix the data access model first**. Do not build control-flow machinery on top of a callback-heavy foundation. Each layer of control-flow improvement (buffer flush → call_indirect → inlining) is wasted if the data ops underneath still round-trip through Rust.

The question to ask before writing any JIT code: *"After this change, what does one iteration of the innermost loop look like, and how many Rust calls does it make?"* If the answer isn't zero (or one for insertion), the architecture is wrong.

This rule exists because four stages of JIT work (Stages 1–4) were built without ever asking this question. The hash probe became inline, but `packed_data_ptr`, `packed_count`, `packed_recent_idx`, and `packed_try_insert` remained callbacks on every inner-loop iteration. The stages kept finding real overhead to eliminate at the control-flow level while leaving the data-access callbacks untouched throughout.

## Design Principles

**Unify, don't multiply.** One interface for multiple cases > separate interfaces. Plugin systems > hardcoded switches.

**Simplicity over cleverness.** HashMap > inventory crate. OnceLock > lazy_static. Functions > traits until you need the trait. Use ecosystem tooling over hand-rolling.

**Explicit over implicit.** Log when skipping. Show what's at stake before refusing.

**Separate niche from shared.** Don't bloat shared config with feature-specific data. Use separate files for specialized data.

## Workflow

**Batch cargo commands** to minimize round-trips:
```bash
cargo clippy --all-targets --all-features -- -D warnings && cargo test
```
After editing multiple files, run the full check once — not after each edit. Formatting is handled automatically by the pre-commit hook (`cargo fmt`).

**When making the same change across multiple crates**, edit all files first, then build once.

**Minimize file churn.** When editing a file, read it once, plan all changes, and apply them in one pass. Avoid read-edit-build-fail-read-fix cycles by thinking through the complete change before starting.

**Always commit completed work.** After tests pass, commit immediately — don't wait to be asked. When a plan has multiple phases, commit after each phase passes. Do not accumulate changes across phases. Uncommitted work is lost work.

## Context Management

**Use subagents to protect the main context window.** For broad exploration or mechanical multi-file work, delegate to an Explore or general-purpose subagent rather than running searches inline. The subagent returns a distilled summary; raw tool output stays out of the main context.

Rules of thumb:
- Research tasks (investigating a question, surveying patterns) → subagent; don't pollute main context with exploratory noise
- Searching >5 files or running >3 rounds of grep/read → use a subagent
- Codebase-wide analysis (architecture, patterns, cross-file survey) → always subagent
- Mechanical work across many files (applying the same change everywhere) → parallel subagents
- Single targeted lookup (one file, one symbol) → inline is fine

## Session Handoff

Use plan mode as a handoff mechanism when:
- A task is fully complete (committed, pushed, docs updated)
- The session has drifted from its original purpose
- Context has accumulated enough that a fresh start would help

**For handoffs:** enter plan mode, write one or two sentences that are literally just a pointer into TODO.md (e.g. "See TODO.md § JIT tuning — next item is X"), and ExitPlanMode. **Do not summarize, do not re-describe the work.** The fresh session reads TODO.md directly. Do NOT investigate first — the session is context-heavy and about to be discarded.

**For mid-session planning** on a different topic: investigating inside plan mode is fine — context isn't being thrown away.

**TODO.md is the lossless record. Memory files are lossy.** When preserving state across sessions, update TODO.md with the current plan/status and point the handoff there. Do not re-summarize plans or roadmaps into memory files — that introduces drift and the distorted version gets treated as authoritative. Memory files are for user preferences, workflow notes, and cross-cutting lessons — not for re-stating what TODO.md already tracks.

Before the handoff plan, flush any new items to TODO.md. Memory files only need updating if there is genuinely new user/workflow/feedback information that isn't in TODO.md.

## Commit Convention

Use conventional commits: `type(scope): message`

Types:
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code change that neither fixes a bug nor adds a feature
- `docs` - Documentation only
- `chore` - Maintenance (deps, CI, etc.)
- `test` - Adding or updating tests

Scope is optional but recommended for multi-crate repos.

## Negative Constraints

Do not:
- Announce actions ("I will now...") - just do them
- Leave work uncommitted
- Use interactive git commands (`git add -p`, `git add -i`, `git rebase -i`) — these block on stdin and hang in non-interactive shells; stage files by name instead
- Use path dependencies in Cargo.toml - causes clippy to stash changes across repos
- Use `--no-verify` - fix the issue or fix the hook
- Assume tools are missing - check if `nix develop` is available for the right environment
