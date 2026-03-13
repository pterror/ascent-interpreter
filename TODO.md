# TODO

## Parser

- [x] Add desugaring pass (disjunctions → multiple rules, negation → aggregation, pattern args → if-let, wildcards → fresh vars)
- [x] Fuzz test parser against ascent_macro to verify 1:1 syntax parity
- [x] Property-based test: generate random valid ASTs, serialize, re-parse

## Interpreter Core

- [x] Design interpreter IR (simpler than ascent's HIR/MIR, focused on evaluation)
- [x] Implement semi-naive evaluation loop
- [x] Relation storage (HashSet-based, indexed)
- [x] Variable binding and unification
- [x] Built-in aggregators (count, sum, min, max, mean, not)
- [x] Expression evaluation (arithmetic, comparisons, ranges, generators)
- [x] Stratification (aggregation rules run after base rules reach fixpoint)
- [x] Pattern matching in conditions (if let)
- [x] Full dependency-based stratification (SCC analysis via petgraph)

## Runtime

- [x] REPL for interactive Datalog queries
- [x] File-based program execution
- [x] Incremental evaluation (:retract to remove facts, :undo to remove last statement)
- [x] Query interface (:query rel(pattern, ...) with wildcards, int/string/bool filters)
- [x] Per-column hash indices for join acceleration

## Lattice Support

- [x] Add Dual value type and lattice join semantics
- [x] Lattice-aware insert (merge by key columns instead of dedup)
- [x] `?pattern` prefix for lattice value pattern matching
- [x] Port lattice tests from ascent (shortest path, etc.)

## Custom Types (BYOD — Rust embedding API)

- [x] `DynValue` trait + `Value::Custom` variant for user-defined types
- [x] `TypeRegistry` on Engine for registering custom constructors/destructors
- [x] Thread relation attributes through IR (`attrs: Vec<Attribute>`)
- [x] Pattern matching support for custom types (destructuring in clauses)
- [x] Serde-based automatic `DynValue` registration
- [x] Full syntax support: parse from strings, type resolution at runtime

## Interpreter Performance

Baseline: 15–550x slower than compiled ascent (Criterion benchmarks).
Current: 6–52x (after optimizations below).

The remaining gap is largely the inherent cost of interpretation: dynamic Value
dispatch, runtime variable binding, indirect indexing. Further gains require
building a compiler (Cranelift/native codegen), which is a different project.

### Low-effort

- [x] Intern variable names (u32 index instead of String keys in Bindings)
- [x] ~~Rc-wrap tuples~~ Direct Vec<Value> storage (removed Rc, in-place lattice mutation)
- [x] Stream aggregation instead of collecting into Vec then reducing

### Medium-effort

- [x] FxHashMap for all hot-path hash maps (bindings, indices, dedup sets)
- [x] Vec-indexed Bindings (O(1) direct slot access by VarId instead of hash lookup)
- [x] Eliminate per-match binding clones via undo log (rollback instead of clone)
- [x] Multi-column index selection with pre-filter for join acceleration
- [x] Pre-compile rules: intern variable names to VarIds, pre-evaluate literals, flatten syn::Expr to CExpr
- [x] Avoid Box<dyn Iterator> in process_clause full-scan path (virtual dispatch per tuple)
- [x] Compile-time condition reordering (move `if` conditions earlier in body to filter before joins)
- [x] Pre-computed `all_args_bound` flag (skip `find_bound_columns` allocation on fully-bound fast path)
- [x] Pre-computed clause match plan (bound_cols + fresh_cols: skip find_bound_columns + match_clause)
- [x] Reuse head tuple buffer (avoid per-derivation Vec allocation)
- [x] Delta-specific indices (separate index for recent tuples, skip is_recent checks)
- [x] Index-accelerated aggregation (use bound columns for index lookup instead of full scan)
- [x] Hash join for large relations (per-column indices already implement hash-based join lookups)
- [x] Type specialization: i32-specialized dedup in RelationStorage (FxHashSet<Vec<i32>> fast path)

### High-effort

- [x] Rule body pipeline without intermediate Vec<Bindings> (recursive streaming with undo log)

### Next-gen performance (targeting LSP-grade incremental queries)

Goal: close the 6–52x gap with compiled ascent; enable >>60fps incremental evaluation on symbol graphs.

Steps 1–2 are representation changes. Step 3 is the highest-value item for LSP (process the delta, not the world). Steps 4–6 are throughput optimizations, largely orthogonal to step 3.

1. [x] String interning — thread-local intern table, `Value::String(SymbolId)` where `SymbolId` is `u32`. Equality/hashing are integer ops; cloning is Copy. Lexicographic ordering resolves through the interner.
2. [x] Flat tuple storage — `Vec<Value>` flat buffer with stride-based access + `hashbrown::HashTable<usize>` for allocation-free dedup. Eliminates per-tuple heap allocations.
3. [ ] Incremental evaluation — highest value for LSP; after interning, diffing facts is cheap `[u32; N]` comparison.
   - [x] Persist engine state across queries (don't rebuild relations from scratch)
   - [x] Source-tagged facts — tag facts with `SourceId`, bulk retract/re-assert by source (files, modules, REPL lines, etc.)
   - [x] Strata invalidation — given changed relation names, identify and re-run only affected strata
   - [x] Incremental addition — insert new facts as deltas, re-run affected strata from deltas only (monotone strata)
   - [x] Non-monotone strata re-derivation — clear and re-derive strata containing negation/aggregation when inputs change
4. [x] Bytecode compiler for expressions — compile `CExpr` to a flat bytecode with a tight eval loop (LOAD_VAR, LOAD_CONST, ADD, CMP_EQ, etc.). Replaces tree-walk `eval_expr`. Can be worked in parallel with step 3.
5. [x] Arity-specialized eval routines (feature-gated: `specialized`) — at rule load time, classify relations by type signature. Relations with all u32-representable columns (i32, u32, String, bool) use `PackedStorage` with dual-buffer layout: u32 flat buffer for fast dedup/index, Value buffer for eval loop reads. Graceful downgrade to generic on type mismatch. Mixed/wide relations fall back to generic `Vec<Value>` path.
6. [x] Cranelift JIT (feature-gated: `jit`) — typed packed JIT implemented. For PackedStorage rules, reads u32 directly from packed_data buffer; bindings are flat Vec<u32>; bound-col checks are inline icmp; no Value cloning in hot loop. Eligible: Clause-only body, Var-only clause/head args, all relations Packed at dispatch time.
   - **Attempted first**: trampoline JIT (extern "C" helpers for all Value ops) — ~0% speedup, wrong approach (kept for non-packed fallback)
   - **Correct approach implemented**: typed loads from u32 flat buffer, icmp comparisons, no Value enum in inner loop

7. [x] Intern arbitrary `Hash + Eq` types to extend PackedStorage beyond the current (i32, u32, bool, String) set. `PackedType::Interned(Rc<dyn InternTable>)` — strings via `StringTable` (thread-local, zero-copy), custom types via `HashInternTable`. Eliminated `Value::String(SymbolId)` entirely; all interned values are `Value::Interned(Rc<dyn InternTable>, u32)`. Value size held at 32 bytes.

### Full-program stratum JIT (next major perf initiative)

Goal: compile the entire semi-naive loop for a stratum to a single native function,
matching the performance of the `ascent!` macro. Three stages:

Current JIT architecture (context for what needs to change):
- The semi-naive fixpoint loop lives entirely in Rust (`eval.rs`) — the JIT only compiles individual rule bodies
- Packed JIT already does direct u32 array arithmetic (no Value enum in hot loop)
- Per-rule variants: 1 full + N recent (one per clause); Rust dispatches each variant
- Packed JIT bails on any conditions; trampoline JIT calls `jit_eval_condition` helper for `CCondition::If`

**Stage 1 — Widen packed JIT coverage** ✅ DONE (`274c380`)
- [x] Conditions in packed JIT: CExpr compiled to Cranelift icmp/iadd; checked at innermost match point before head emission.
- [x] Literals in clause args: inline `icmp` against packed constant; full-scan restructured with explicit `continue_block`.
- [x] Tests: condition comparison, literal clause arg, arithmetic condition.

**Stage 2 — Stratum-level JIT** ✅ DONE
- [x] `jit_stratum_flush_advance` Rust helper: flush all per-rule results into head relations, advance all packed rels, return changed bool.
- [x] `StratumMetaCtx` (repr C): carries full/recent fn ptr arrays + counts + flusher ptr.
- [x] `codegen_stratum_meta_fn` (stratum_codegen.rs): Cranelift function that owns the `while has_delta` loop, calls rule variants via `call_indirect`, calls flush+advance helper.
- [x] `JitCompiler::compile_stratum_meta`: compiles and caches per-stratum key.
- [x] `Engine::try_run_stratum_meta`: builds pinned runtime context once per stratum, calls meta-fn; hooked as fast path in `run_stratum`.
- [x] Tests: TC (2-rule SCC), multi-rule stratum, single-rule with self-join cycle.
- Advance/recent state machine stays in Rust helpers; only loop control is in native code.
- True cross-rule inlining (inline rule bodies into stratum function, direct buffer writes) — defer to stage 3.

**Stage 3 — Direct-insert stratum** ✅ DONE
- [x] `PackedJitContextV3`: head_rels array replaces results buffer — rule variants write directly to head relations.
- [x] `PackedJitFnV3`: new rule variant signature using direct-insert context.
- [x] `StratumStage3Ctx` + `StratumStage3Fn`: parallel stratum loop using V3 variants + `jit_stratum_advance`.
- [x] `packed_try_insert` helper: thin wrapper around `insert_packed_raw`, callable from JIT.
- [x] `jit_stratum_advance` helper: advance-only (no flush step needed).
- [x] Correctness: re-fetch `packed_data_ptr` inside scan loops to handle reallocation when head == clause relation (recursive rules). Stage 3 is now the primary fast path; Stage 2 (buffered) remains as fallback.
- [x] Tests: recursive triangle, multi-hop TC (5-node), conditional recursive rule.
- Eliminated: per-head-tuple Vec<u32> allocation, global results-buffer flush pass.
- Remaining: true inlining (no `call_indirect` for rule dispatch) — defer to Stage 4.

**Stage 4 — True inlining (no `call_indirect`)** ✅ DONE
- [x] Inline all rule bodies directly into the stratum Cranelift function — one function per stratum with full visibility across all rules. `gen_clauses_v3` called directly from stratum codegen with shared `next_var` counter to avoid Cranelift Variable ID collisions. Stage 4 runs before Stage 3; Stage 3 remains as fallback.

### JIT tuning

- [ ] **Stage 4 code-size regression** — for rules with many clauses (e.g., triangle: 3 clauses), Stage 4 inlines 4 copies of a 3-deep loop nest into one Cranelift function, causing I-cache pressure that outweighs the call_indirect savings. Benchmark: triangle n=30 is 45% slower with Stage 4 than interpreter (Stage 3 was baseline). Fix options: (a) heuristic fallback to Stage 3 when total_clauses_across_all_variants > threshold; (b) reduce code duplication by only emitting recent variants for clauses that actually filter on recent data.

- [ ] **Cache `packed_data_ptr` for non-recursive rules** — `gen_full_scan_v3` calls `packed_data_ptr` inside the scan loop on every iteration to handle the case where head relation == clause relation (recursive insert can reallocate). For non-recursive rules (head ∉ clause relations), the pointer is stable. Pass a `is_recursive: bool` flag through codegen; only re-fetch inside the loop when true. Applies to Stage 3 and Stage 4. Would reduce inner loop body size by one runtime call.

### Relation storage optimizations

- [ ] **Lazy `recent_col_indices` rebuild** — `advance()` in both `PackedStorage` (`specialized.rs`) and `RelationStorage` (`relation.rs`) unconditionally rebuilds recent indices even for sink relations (head-only, never body clauses). The right implementation is a new `ensure_recent_indices(&mut self)` called at eval-loop level only for relations that are body clauses in the current stratum — avoids the `&self`/`RefCell` tangle. Impact limited to programs with head-only output relations; benchmarks (TC, triangles) don't benefit.

- [ ] **Sparse delta evaluation (LSP path)** — for incremental workloads where deltas are large relative to full relations, build selective delta indices only on columns actually used in downstream join clauses (determined at rule compile time). Avoids scanning delta tuples whose relevant columns don't match any current binding, reducing O(|delta|) inner loop iterations for non-matching tuples.

### Not planned

- ~Parallel SCC evaluation~ — strata are sequential by definition; intra-stratum parallelism is a research problem

## Testing

- [x] Port ascent test suite (28 compat tests: fizzbuzz, factorial, negation, aggregation, joins, pattern matching, etc.)
- [x] Comparison tests: run same program in ascent macro vs interpreter, compare results (17 tests)
- [x] Performance benchmarks (transitive closure, triangles, connected components, fibonacci)

## Documentation

- [x] Usage examples
- [x] Syntax reference
- [x] Architecture overview
