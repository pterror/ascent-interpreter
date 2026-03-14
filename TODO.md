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

- [x] **Shareable JitCache across Engine instances** — JIT compilation cache is per-`Engine`; every `Engine::new()` recompiles from scratch. Benchmarks therefore measure compile+run, not execution-only. For the LSP use case (many incremental evaluations of the same program), a shareable `Arc<JitCache>` would let all engine instances reuse compiled strata. Also needed for a clean "hot JIT" benchmark variant that isolates execution cost.

- [x] **Stage 4 benefit for non-recursive strata** — after unblocking triangle JIT (CClause.conditions fix), triangle shows 3.9× speedup over interpreter at n=30. Gap to ascent_macro is 11.6× and is STRUCTURAL (same ratio at n=20 and n=30), ruling out cache capacity as the cause. Profiling (perf stat) shows 17× more instructions per iteration — from memory-based bindings (load/store per variable vs register), linked-list index traversal, and Cranelift vs LLVM code quality. No further low-effort fixes identified for this gap.

- [x] **Handle `CClause.conditions` in JIT eligibility** — repeated-variable equality checks (e.g. `edge(a,b), edge(b,c)` where `b` is shared) are stored as `clause.conditions: Vec<CCondition>`, not as top-level `CBodyItem::Condition`. The JIT rejects any rule with non-empty `clause.conditions`, silently falling back to the interpreter. This blocks triangle detection entirely (33–54x gap to macro is interpreter overhead, not JIT overhead). Fix: emit clause conditions as conditional branches after binding the clause variables in `gen_full_scan_v3` / `gen_index_scan_v3`, same as top-level conditions.

- [x] **Handle head expressions in JIT** — rules with non-`Var` head args (e.g. `fib(n+1, a+b) <-- ...`) are ineligible. Fix: evaluate the expression in Cranelift IR at emit time and pack the result. Unblocks fibonacci and any rule that computes derived values in the head.

- [x] **Eliminate duplicate dedup tables** — unified into `JitDedupTable` as sole authoritative dedup; removed `HashTable<usize>` from PackedStorage.

- [x] **Cranelift Variables for bindings (Phase 1)** — replaced heap `bindings: *mut u32` array (load/store on every inner-loop variable read/write) with Cranelift `Variable` API. All V3/Stage4 codegen declares `var_count` Variables at function entry; `def_var`/`use_var` replace `store`/`load` through a heap pointer. Removed `bindings` field from `PackedJitContextV3`; updated offsets. Added `var_count` to `JitCompiler`, set from `Engine` before each compile. Eliminates the heap store+load pairs (e.g. `store notrap v22; load notrap v6`) visible in CLIF dumps on every inner-loop binding access.

- [ ] **Cache `packed_data_ptr` for recursive rules (Phase 2)** — for recursive rules, `packed_data_ptr` is re-fetched per outer-scan tuple because `packed_try_insert` may reallocate the Vec. **Phase 1 benchmark results (2026-03-14):** fibonacci jit_hot/20 = 17.4 µs vs ascent_macro 7.6 µs (2.28×); triangle jit_hot/20 = 405 µs vs ascent_macro 32.6 µs (12.4×). CLIF confirms fn2 (packed_data_ptr) is called once per outer loop iteration for recursive clauses, contributing ~12% of fibonacci overhead. **Assessment:** savings ≈ 2 µs (2.28× → ~2.0×). Implementation requires threading dptr Variables through gen_clauses_v3 recursion (complex). Triangle is unaffected (non-recursive). Defer — implementation cost exceeds benefit; remaining gap is dominated by Cranelift code quality.

- [ ] **Direct-load `packed_count` (Phase 3)** — replace `call fn1(rel, col)` (once per fixpoint iteration, full-scan path) with a direct `load.i64` at the known byte offset of `count`/`recent.len()` in `PackedStorage`. Not in the inner loop; requires `#[repr(C)]` on `PackedStorage` (currently absent). Defer — negligible impact.

- [x] **dynasmrt asm backend (Phase 4)** — implemented `asm_codegen.rs` (~970 lines): x86-64 hand-written JIT for Stage 4 stratum functions under `jit-asm` feature, tried first before Cranelift fallback. **Benchmark result (2026-03-14):** TC jit_hot/50 with asm = 259 µs vs Cranelift-only = 245 µs — asm is **6% slower**. Root cause: asm backend stores binding variables to rbp-relative stack slots (same memory-based pattern as pre-Phase-1 Cranelift), while Cranelift+Phase-1 Variables keeps them register-allocated. Falls through to Cranelift for triangle (3 clauses), connected_components (1-clause+bound_cols), fibonacci (Deref expr in condition). **Next:** to make asm faster than Cranelift, bind outer-loop-stable variables (x, y from clause0) into callee-saved registers (r12/r13/r15) rather than stack slots, eliminating the load/store pairs on every inner-loop iteration. Alternatively, evaluate QBE (~70% of LLVM quality, comparable compile speed) as a codegen backend.

- [x] **Cache `packed_data_ptr` for non-recursive rules** — `gen_full_scan_v3` calls `packed_data_ptr` inside the scan loop on every iteration to handle the case where head relation == clause relation (recursive insert can reallocate). For non-recursive rules (head ∉ clause relations), the pointer is stable. Pass a `is_recursive: bool` flag through codegen; only re-fetch inside the loop when true. Applies to Stage 3 and Stage 4. Would reduce inner loop body size by one runtime call.

#### Remaining inner-loop Rust calls (the real floor)

The true minimum is **zero per-iteration calls + one per rule-invocation for insertion**:

- [x] **Eliminate `packed_recent_idx`** — called once per recent-scan iteration to map `seq_idx → rel.recent[seq_idx]`. Fix: expose `recent.as_ptr()` via a new `packed_recent_ptr` helper (called once before the loop, same pattern as `packed_data_ptr` caching above); replace the per-iteration call with an inline `load ptr_type from (recent_ptr + i * ptr_size)`.

- [x] **Inline dedup probe for `packed_try_insert`** — at fixpoint, most head tuples are duplicates; `packed_try_insert` is called once per match but returns 0 (duplicate) the vast majority of the time. Implemented: `JitDedupTable` (open-addressed flat u32 hash table, stride = arity+1) embedded in `PackedStorage`, populated incrementally by `update_jit_indices()`. JIT loads `head_dedup_handles` pointer from ctx offset 40 at function entry, computes hash and probes table inline, jumps past `packed_try_insert` on duplicate (zero Rust calls for duplicates; one call for new tuples). Note: bulk-insert scratch buffer not implemented — new tuples still call `packed_try_insert` individually, but this is one call per new tuple (already minimal).

### Near-native performance roadmap

**Goal:** within ~1.5× of `ascent_macro` on join-heavy queries. Current gaps: triangle 12×, connected_components 6.7×, TC 2.6×, fibonacci 2.3×.

**Step 0 — Profile to split the triangle gap** *(done, 2026-03-15)*

perf/valgrind unavailable. Timing-scaling inference suggested cache-miss dominated (ratio grows with n).
Attempted Step 1 first (index structure). **Revised conclusion:** at n≤30, the linked-list values pool
is ≤1.5KB (n=20: 380 u32s) — fits entirely in L1. Step 1 regressed by 3–5% at benchmark sizes. The
gap at benchmark sizes is instruction-count/codegen dominated, not cache dominated.
**Do Step 3 first** for benchmark-size wins; revisit Step 1 when targeting n>100 workloads.

**Step 1 — Fix join index data structure** *(required to close the triangle gap)*

**Root cause (confirmed 2026-03-15):** The linked-list inner loop creates a serialized load-use
dependency chain: each step requires the result of the previous load to compute the next address.
At ~4 cycles/step (L1 load latency) with ~10 steps = 40 serialized cycles per outer iteration.
`ascent_macro` uses `HashMap<K, Vec<V>>` — the inner iteration is a sequential array scan where
the CPU can speculatively load ahead. LLVM also auto-vectorizes it with SIMD. This alone accounts
for most of the 11.5× triangle gap.

**Attempted (2026-03-15):** Global contiguous start+count layout — 3-pass rebuild of entire index
on every `advance()`. Result: 3-5% regression at n≤30 (inner loop adds more stack loads; rebuild
overhead > gain), TC +739% regression (O(N³) total rebuild work).

**Correct approach: EDB-contiguous index strategy**
- EDB/stable relations (never in a head): build contiguous index ONCE before stratum start, never
  rebuild. Enables sequential scan in inner loops for all variants.
- Derived/recursive relations (in a head): keep linked-list full index (O(1) incremental inserts);
  rebuild RECENT index contiguously per `advance()` — O(|recent|) per iteration, not O(|full|).
- Inner loop code: two variants in both asm and Cranelift backends — contiguous scan (EDB) and
  linked-list traversal (derived). Select at variant-emit time based on `is_edb[level]` flag.

For triangle (edge is EDB): full edge index contiguous → sequential inner loops → expected ~3-4×
improvement. For TC (edge EDB, path derived): edge inner probe sequential, path full stays
linked-list; TC gap closes modestly.

Implementation scope: `jit_index.rs` (new index type), `specialized.rs` (detection + build),
asm/Cranelift backends (new contiguous inner loop). ~400 lines across 4 files.

`PackedIndex` currently stores match lists as linked-list chains — pointer-chasing on every inner-loop
iteration. `ascent_macro` uses `HashMap<K, Vec<V>>`: after the key lookup, inner iteration is a
sequential scan of a contiguous array. Replacing chains with contiguous per-key storage (Vec-per-key,
or Robin Hood open-addressing with inline value arrays) gets cache-sequential inner loops. Affects all
backends equally; orthogonal to JIT codegen.

**Step 2 — Write-ahead buffer for head insertion** *(estimated 1.2–1.5×)*

Accumulate new head tuples in a fixed-size stack buffer during the inner loop; flush to `PackedStorage`
after the inner loop body. Eliminates the `packed_try_insert` Rust call from the hot path for
non-dedup-failing cases. For recursive rules (head ∈ clause rels): flush at end of each outer
iteration instead of end of inner loop. Note: dedup probe is already inline; this removes the
remaining call for *new* tuples.

**Step 3 — Complete the asm backend for ~100% rule coverage** *(closes the codegen gap)*

Extends `asm_codegen.rs` with a proper register assigner and full pattern coverage. Keep Cranelift
as fallback for anything the asm backend explicitly rejects.

- **3a — Depth-priority register assignment** (~150 lines): variables bound in clause `i` are live
  from clause `i` entry to head emission; assign by depth so clause-0 variables (longest-lived, in
  every inner-loop iteration) get callee-saved registers first (r12, r13, r15), spill later-bound
  variables to dedicated stack slots only when the register file is exhausted. Eliminates the
  load/store pairs per inner iteration that make the current asm backend slower than Cranelift.

- **3b — N-clause rules** *(committed `cec22d0`, 2026-03-15, at parity with Cranelift)*:
  Recursive `emit_clause_level` with per-depth stack slots (vptr, node-save, dptr-save). Triangle
  (3-clause) now handled by asm backend. Benchmark: n=20 380µs asm ≈ 380µs Cranelift — 11.5×
  gap vs ascent_macro (33µs) unchanged. Attempted: pre-fetch `packed_data_ptr` before outer loop
  for non-recursive clauses (would eliminate 6600 calls per fixpoint iteration). **Result: no
  measurable improvement.** Function calls are hidden by OOO execution — they're not on the
  critical path. **True bottleneck identified: serialized load-use dependency chain in linked-list
  traversal.** Each step: `load values[node*8]` → `load values[node*8+4]` → `mov r15d, result`
  → next step depends on r15. At ~4 cycles/step (L1 load latency) with ~10 steps per inner
  iteration = 40 cycles serialized vs ascent_macro's Vec slice which is speculative/pipelined/
  vectorizable. Fix requires contiguous per-key values arrays (see Step 1 / Step 4).

- **3c — Expression completeness** (~150 lines): `CUnOp::Deref` is a no-op in the packed
  representation (trivial); add Div/Mod/bitwise; handle arbitrary user-defined function calls via
  function pointer with caller-save spill/restore around the call site. Remove all `check_expr` /
  `check_binop` `Err` returns for arithmetic ops.

- **3d — Aggregation codegen** (~250 lines): single scan loop, accumulate into a register (or stack
  slot for non-scalar aggregators), emit result tuple after the loop. The aggregation function itself
  is a call (user-defined); the loop structure is simpler than a join.

- **3e — Negation / anti-join** (~150 lines): probe the negated relation's index; branch to
  `skip_label` on *hit* rather than miss. Reuses all probe infrastructure from 3b.

**Step 4 — EDB-contiguous index (implements Step 1 correctly)** ✅ DONE (2026-03-15)

Implemented EDB-contiguous index strategy. JitHashIndex gains `build_contiguous()` and
`is_contiguous` flag. `PackedStorage` gains `jit_is_edb` flag; `update_jit_indices()` builds
contiguous full+recent indices for EDB, linked-list for derived. Both Cranelift and asm backends
emit contiguous inner loops (sequential j=0..count scan) when `!is_recursive`, linked-list loops
otherwise.

**Actual benchmark results (2026-03-15):**
- triangle `jit_hot/20`: 408µs (-5.8% vs 405µs before; ratio vs ascent_macro: 12.4× unchanged)
- triangle `jit_hot/30`: 1.39ms (-7.8% vs ~1.5ms before; ratio vs ascent_macro ~12×)
- TC `jit_hot/50`: 273µs (no significant change; ratio vs ascent_macro: 2.6×)

**Assessment:** Contiguous inner loops give ~6-8% wins at benchmark sizes (n≤30), not the 3-4×
expected. Root cause of smaller-than-expected gain: at benchmark sizes (n=20: only 190 edges),
each key has only 1-2 values on average — linked-list overhead is negligible for small chains.
The load-use serialization argument only applies when chains are long (n>100+). The 11.5× gap
at benchmark sizes is still dominated by Cranelift vs LLVM code quality (more instructions,
poorer register allocation, no auto-vectorization).

**Residual gap:** triangle still 12×. Next step: asm backend with depth-priority register
assignment (Step 3a) to eliminate stack spills in inner loops, or evaluate QBE/LLVM backend.

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
