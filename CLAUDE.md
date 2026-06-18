# CLAUDE.md

Behavioral rules for Claude Code in the ascent-interpreter repository.

## Project Overview

Interpreter and JIT compiler for [Ascent](https://github.com/s-arash/ascent) Datalog programs.

## Architecture

- Single-crate layout. Modules: `syntax/` (parse + desugar, `derive-syn-parse` + `syn`), `ir.rs` + `ir/` (lowered IR), `eval/` (semi-naive engine + expression evaluator), `main.rs` (REPL/CLI).
- Negation lowers directly to a `not` `Aggregation` in IR (`src/ir.rs` around `BodyItemNode::Negation`) — not via `::ascent::aggregators::not`.
- Stratification: simple 2-stratum (base rules → aggregation rules) handles common cases; full SCC-based stratification is needed for the general case. Aggregation gating must check both `Clause` and `Aggregation` body items.
- `syn::parse_str::<AscentProgram>(input)` works outside proc-macro context — that is how the interpreter front-ends the macro grammar.
- `#[peek_with(...)]` from `derive-syn-parse` requires standalone `fn`s, not closures.
- `PackedType::Interned(table)` (see `src/eval/specialized.rs`) hashes through an `InternTable` at pack time. String is just `Interned(string_table())` — there is no separate zero-cost `String` variant.

## Publishing

- Only the `ascent-interpreter` crate is published. The `ascent` namespace is not ours; do not try to publish `ascent-syntax`/`ascent-ir`/`ascent-eval` as separate crates. The binary is named `interpreter` (see `[[bin]]` in `Cargo.toml`).

## REPL Contract

- REPL prompts, status, errors, and `:command` feedback go to **stderr**. Relation/data output goes to **stdout**. This lets `ascent-interpreter foo.ascent | …` pipe cleanly. Preserve the split when adding new REPL output.

## Rust Edition

- `gen` is a reserved keyword in Rust 2024. Use `generator` (or another name) for any such identifier.

## Development

```bash
nix develop        # Enter dev shell
cargo test         # Run tests
cargo clippy       # Lint
cd docs && bun dev # Local docs
```

If a tool appears missing, you are outside `nix develop`. Do not assume the tool is unavailable to the project.

## Performance Goal

**The JIT target is wall-clock parity with `ascent_macro` (LLVM-compiled Datalog).** This means optimizing for actual runtime, not a proxy metric. Instruction count matters, but so do cache misses, stall cycles, and memory layout. A change that reduces instructions while increasing cache pressure can be a regression.

When evaluating a JIT optimization, profile with `perf stat -e instructions,cycles,cache-misses` and compare *per-iteration* numbers against the `ascent_macro` benchmark. The gap is closed when both instruction count and cache behavior converge, not just one.

Current state (2026-03-21): triangle jit_hot/20 ≈ 106–114µs vs ascent_macro ≈ 50–57µs (**~2.1×** total; high variance). Eval-only (jit_hot − jit_setup_only): ~59µs vs ~46µs = **~1.2–1.3× eval ratio** at n=20 — close to parity for the LSP target. **Benchmark environment has high variance** (always compare hot vs macro in the same run and check ratios, not absolutes). Key bookkeeping optimizations (2026-03-19): batch native flush (bd9df6b) eliminates 1140 per-tuple call overhead; sequential-index fast path (6b5c380) skips gather Vec for contiguous recent indices; EDB recent rebuild skip (8b2c8f5) avoids 3 allocs+frees per EDB relation per iteration when recent is already empty. Register allocation improvement (2026-03-20, `66107d5`): pointer-based outer loop frees r14 for variable `a` — saves 1 stack load per inner probe. JitSwissTable SIMD probe removed (2026-03-20, `f04eb17`): built and then removed — adds ~15 instructions overhead vs scalar for small hot L1/L2 tables; JitRelData C-visible region shrunk from 88 → 64 bytes (no swiss field). jit_dedup pre-reserve before batch flush (2026-03-20, `1d85347`): eliminates 8 rehash cascades for 1140 tuple inserts (was growing 16→32→64→...→2048); single upfront alloc saves ~8µs and ~330K instructions per benchmark call. AVX2 SIMD column scan (2026-03-20, `c5c023e`): `emit_col_value_simd_prefix` added — processes 8 u32 column values per iteration using vpcmpgtd, fires when `detect_simd_filter` finds a Lt/Gt condition with a stack-resident comparand. No measurable benefit for standard triangle query (comparand is register-resident at inner loop entry). JitColIndex linear scan for existence checks (2026-03-21, evaluated and reverted): replacing `JitTupleSet` probe with JitColIndex linear scan for fully-bound EDB clauses was consistently slower — O(n) scan depth (~99 iterations × 5 instr = ~495 instr/probe) vs JitTupleSet's ~43 instr/probe. JitTupleSet is L2-resident in hot benchmarks (not L3), so cache miss savings do not offset the 10× instruction overhead. The earlier "2.3× at n=100" measurement was noise. `emit_native_edb_full_probe` kept as dead code for a future binary-search variant (see TODO.md). Merge intersection (2026-03-21): `detect_merge_pattern` + `emit_merge_scan_exist` — fuses arity-2 EDB col-scan (level L) with fully-bound EDB existence check (level L+1) into a two-pointer sorted-merge loop. Instead of probing JitTupleSet 99×/pair (each 43 instr, L2-resident), walks both sorted JitColIndex vals arrays in O(|A|+|B|). Benchmark results (same-machine, separate runs): n=50: neutral (JitTupleSet 49KB still L2-resident, branch misprediction offsets instruction savings); n=100: ~2% improvement (noise); n=200: **eval-only 333ms → 270ms = 1.23×** (JitTupleSet 393KB spills to L3, merge walks L1-cached 800-byte slices). The sorted merge is neutral-or-better for all sizes. Large-n performance (n=200): jit_hot ~276ms vs ascent_macro ~109ms ≈ **2.5× total**.

**Interned ordering comparisons use a callback, not raw ID comparison (2026-03-25).** For Lt/Le/Gt/Ge on interned columns (strings, custom types), the JIT emits a `jit_cmp_interned` trampoline call that reconstructs the `&dyn InternTable` fat pointer and calls `cmp_ids`. Raw u32 ID comparison gives wrong results because intern IDs depend on insertion order, not semantic order. Eq/Ne on interned types is safe (same ID = same value). String triangle benchmark: JIT 114µs vs interpreter 1.5ms = 13× speedup despite the callback. The i32 path is unchanged (no overhead). See TODO.md for the specialization idea (monotonic ID tracking to skip the callback when safe).

**Fully-bound existence check paths must emit clause.conditions (fixed 2026-03-25).** `optimize_body` Phase 2 merges standalone `if` conditions into the preceding clause's `conditions` list. The `emit_native_edb_full_probe` and `emit_tuple_set_probe` fast paths for fully-bound clauses emitted `rule_conds` but skipped `clause.conditions`, silently dropping merged conditions like `if a < b`. The merge intersection path (`emit_merge_scan_exist`) was correct. Rule: any new fully-bound fast path MUST emit both `clause.conditions` AND `rule_conds`.

**Any change that increases live SSA values across a hot Cranelift loop causes regressions.** Observed twice: (1) dedup probe CSE (keep entry_ptr/entry_hash live from probe_loop through probe_check_found/probe_verify) — TC +57% despite saving ~5 instructions; (2) deferred stack stores (keep col_vals live until call_insert instead of storing early) — triangle +39%, TC +57%, fibonacci +18%. Root cause: early stores in Cranelift act as cheap register-pressure relief — the compiler writes values to a stack slot, frees their registers for the loop body, then reloads for verification. Removing early stores extends live ranges across the loop, causing spills of other values that cost more than the instructions saved. Rule: never extend a live range across a hot loop without first verifying there is register headroom (use CLIF dump to count live values at the back edge).

**In x86-64 ASM codegen, `rdi` and `edi` alias the same register.** When loading a 64-bit pointer into `rdi` (e.g., `mov rdi, [rbp + slot]`), the 32-bit `edi` is overwritten with the low bits of the pointer. Any args loaded into `edi`/`esi` before a pointer load will be silently corrupted. Rule: always extract pointer fields into scratch registers (`r8`, `r9`, `r10`, etc.) **before** loading args into `edi`/`esi`. Observed in `emit_tuple_set_probe` variants (2026-03-20) — caused wrong triangle results when JitRelData pointer was loaded after arg0/arg1 were already in edi/esi.

**ABI scan callbacks (`packed_data_ptr`, `packed_count`, `packed_recent_ptr`) are NOT the bottleneck.** These functions access warm cache lines (offsets 24, 72, 112 in `PackedStorage`). Any scheme that replaces them with direct loads risks adding cache misses from `PackedScanInfo` fields at cold offsets, or disrupts the overall layout. Attempted 2026-03-16 as `PackedScanInfo` in `PackedStorage` — regressed fibonacci 1.49× → 1.56–1.73× in all configurations. Do not retry.

**Parallelism must also be designed for max throughput.** When implementing parallel evaluation: zero shared state in the hot path (thread-local dedup tables, thread-local head buffers, read-only shared indices), batch merge only after all threads finish their scan. A parallel design that introduces lock contention or false sharing in the inner loop is worse than single-threaded. The test is: does the parallel version scale linearly with thread count on a join-heavy benchmark?

## JIT Architecture Goal

**The end state is zero Cranelift dependency.** The asm backend (`jit-asm`) must handle every rule shape; Cranelift is a temporary fallback to be deleted once asm coverage is complete. Any work that keeps rules falling through to Cranelift is unfinished work, not a solution.

Current state (2026-03-19): Cranelift deleted (commit `0258e88`). ASM backend (`jit-asm`) is the sole JIT path. Ineligible rules fall back to interpreted evaluation. TC at parity (verified). Step 3 complete: 3c (Div/Rem/bitops), 3d (count/sum/min/max aggregation), 3e (negation/anti-join, commit `8c7a3e1`) all implemented.

## JIT Design Rule

**Before implementing any JIT stage, trace the full hot-path and count Rust callbacks.**

If the inner loop still calls back into Rust — for any reason (reading tuple data, inserting, counting) — **stop and fix the data access model first**. Do not build control-flow machinery on top of a callback-heavy foundation. Each layer of control-flow improvement (buffer flush → call_indirect → inlining) is wasted if the data ops underneath still round-trip through Rust.

The question to ask before writing any JIT code: *"After this change, what does one iteration of the innermost loop look like, and how many Rust calls does it make?"* If the answer isn't zero (or one for insertion), the architecture is wrong.

This rule exists because four stages of JIT work (Stages 1–4) were built without ever asking this question. The hash probe became inline, but `packed_data_ptr`, `packed_count`, `packed_recent_idx`, and `packed_try_insert` remained callbacks on every inner-loop iteration. The stages kept finding real overhead to eliminate at the control-flow level while leaving the data-access callbacks untouched throughout.

## Workflow

Batch checks to minimize round-trips:
```bash
cargo clippy --all-targets --all-features -- -D warnings && cargo test
```

After editing multiple files, run the full check once. `cargo fmt` runs in the pre-commit hook.

When the same change spans multiple crates, edit all files first, then build once.

`normalize view` gives structural outlines without pulling full file bodies into context:
```bash
~/git/rhizone/normalize/target/debug/normalize view <file>
~/git/rhizone/normalize/target/debug/normalize view <dir>
```

## Commit Convention

Conventional commits: `type(scope): message`

Types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`. Scope is optional but recommended for multi-crate repos.

## Hard Constraints (repo-local additions)

- No interactive git (`git add -p`, `git add -i`, `git rebase -i`) — these block on stdin and hang.
- No assuming a tool is missing without checking `nix develop`.

<!-- BEGIN ECOSYSTEM RULES -->

## Ecosystem Design Principles

Cross-cutting principles distilled from the ecosystem's own decisions (synthesized in `docs/decisions/throughlines.md`). Apply them when building new repos and recording decisions. (Already-encoded principles — independent-tools / no-path-deps, the delegation model, CLAUDE.md-as-control-surface — live in their own sections and are not repeated here.)

- **Prefer data over code at a seam — where a faithful serialization is actually viable.** Serializable AST / struct / JSON over closures, embedded DSLs, or source text, so artifacts cache, replay, transport, and diff. The preference is conditional, not absolute: when a seam carries irreducibly heterogeneous, one-off glue whose only data form is a leaky lowest-common-denominator schema (or a "descriptor" that just wraps a closure), a code seam is the honest choice. Push to data where the representation stays faithful; don't force it where it doesn't.
- **Library-first; projection-from-one-definition.** The typed library is the source of truth; CLI / HTTP / MCP / WebSocket / JSON surfaces are generated projections, never hand-rolled per surface.
- **Capability security.** Hosts grant pre-opened handles; code only attenuates what it is given; nothing forges authority; allow-list over deny-list.
- **The LLM is an oracle at the leaves, never the control loop.** Determinism is a hard invariant: seeded RNG, event-log replay, build-time-only inference. Per-query LLM in the hot loop is a defect.
- **Trust comes from verifiable evidence, not authority.** Verbatim snippets, pinned-commit permalinks, claim→node citation — never a bare reference.
- **Retire, don't deprecate; collapse asymmetries to primitives.** Remove backward-compat aliases rather than carry them; reduce N special cases to their irreducible primitives.
- **Finish migrations before building on top; fence what you can't finish.** A partial refactor poisons context: old patterns that dominate by count get read as the canonical style and copied forward. Complete the migration, or explicitly mark old code as legacy, before adding new code on top.
- **Validate against reality; tests are the spec.** Load-bearing substrates are validated against real corpora; fixtures and tests define correctness, not aspirational specs.

### Relay discipline (blackboard protocol)

Reach for the blackboard when it earns its keep, not for every subagent. When a payload is large or evidence-heavy enough that passing it through the dispatcher's context would poison it — or when a downstream critic/step must read it by path so the dispatcher routes on a verdict without ingesting the evidence — the subagent writes its output to an artifact file and returns only a path + short digest. That is what stops conclusions being laundered in place of evidence. Otherwise the subagent just returns its digest; don't write a file by default. Persist to a tracked path only when the output is durable (in docs-shaped repos, `docs/artifacts/<session>/`); ephemeral relay scratch stays out of the tracked tree, and repos without that path use a repo-appropriate or scratch location.

## Hard Constraints

- No `--no-verify`. Fix the issue or fix the hook.
- No path dependencies in `Cargo.toml` — they couple repos and break independent publishing.
- No interactive git (no `git rebase -i`, no `git add -i`, no `--no-edit` on rebase).
- No suggesting project names. LLMs are bad at this; refine the conceptual space only.
- No tracking cross-project issues in conversation — they go in TODO.md in the affected repo.
- No assuming a tool is missing without checking `nix develop`.
- Commit completed work in the same turn it finishes. Uncommitted work is lost work.

## Meta

- Something unexpected is a signal. Stop and find out why. Do not accept the anomaly and proceed.
- Corrections from the user are conversation, not material for new rules. Rules are added when a failure mode is observed repeatedly.
- **Confidence only when earned by tangible evidence; verify before you assert, and when you can't, say so.** Confirm a claim against the actual source — read it, run it, check it — *then* state it. If you haven't verified, say "I haven't checked," then go check or ask. Never substitute a plausible-sounding claim for a verified one. The defect is *unearned* confidence — confidence decoupled from checked evidence — and it is a defect even when the answer turns out right, because the process is identical to the confident-wrong case (a lucky guess just hides it, and trains the same habit). The inverse — hedging something you've solidly verified — is the same defect. Report what you actually checked plainly; the target is the coupling between expressed confidence and real evidence, not plainness or confidence itself. (the root failure: confabulation — asserting past your evidence.)
- **At a decision point, generate several genuinely independent candidate approaches, weigh each, and decide where the call is yours or give a weighed recommendation where it's the user's.** For complex/architectural/high-stakes decisions this isn't optional and can't be single-shot: N options from one model pass share blind spots — reworded, not independent. Decorrelate via parallel subagents each from a different starting frame (design-it-twice / design-an-interface), then adversarial judging, then synthesis — before committing. When unsure whether a decision clears that bar, treat it as if it does. (failures: overconfidence; option-dumping; false-independence — single-shot options treated as decorrelated.)
- **Under challenge, re-read the source and report what it literally says.** Let the answer land where the evidence puts it: hold if you were right, correct specifically if you were wrong. The new position must come from re-checking, never from the pressure. (failure: backpedaling — moving to appease.)
- **Re-read the relevant context before acting on it.** Act from the current state, not a stale or half-formed read. (failure: stale-context action.)

<!-- END ECOSYSTEM RULES -->
