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

- [x] **O(1) existence check for fully-bound inner clauses — Cranelift backend (2026-03-16)** — added `gen_tuple_set_probe_v3` to Cranelift codegen; Stage 4 now uses the pre-built `JitTupleSet` for all inner clauses with `fresh_cols.is_empty()`. `tuple_sets_buf` loaded from `StratumStage4Ctx+56`; per-rule offset computed at codegen time (`rule_handle_start * 8`). Block structure: probe_loop → probe_check_found → field_check[0..N] → body/probe_miss → not_found. Stage 3 single-rule path passes `None` (unaffected). **Result:** triangle jit_hot/20: 245 µs → ~165 µs (**20% improvement**, gap 7.2× → 4.7×). fibonacci unchanged (no fully-bound inner clause).

- [ ] **Cache `packed_data_ptr` for recursive rules (Phase 2)** — for recursive rules, `packed_data_ptr` is re-fetched per outer-scan tuple because `packed_try_insert` may reallocate the Vec. **Current benchmark (2026-03-16):** fibonacci jit_hot/20 = ~14.8 µs vs ascent_macro ~9.7 µs (1.5×); triangle jit_hot/20 = ~165 µs vs ascent_macro ~35 µs (4.7×). CLIF confirms fn2 (packed_data_ptr) is called once per outer loop iteration for recursive clauses. **Assessment:** savings < 0.5 µs at current scale. Triangle is unaffected (non-recursive). Defer — small savings.

- [ ] **Direct-load `packed_count` (Phase 3)** — replace `call fn1(rel, col)` (once per fixpoint iteration, full-scan path) with a direct `load.i64` at the known byte offset of `count`/`recent.len()` in `PackedStorage`. Not in the inner loop. Defer — negligible impact.

- [x] **PackedScanInfo (TRIED, REVERTED 2026-03-16)** — attempted to cache `data_ptr`, `count`, `recent_ptr`, `recent_count` in a new `PackedScanInfo` field in `PackedStorage` so the JIT could load directly instead of ABI calls. **Result: net regression in all configurations.** Root cause: the ABI calls (`packed_data_ptr`, `packed_count`, `packed_recent_ptr`) access `packed_data` (offset ~24), `count` (~72), `recent` (~112) — all in cache lines 0-1, which are already warm. Any `PackedScanInfo` placement either adds a cold cache line miss (`#[repr(Rust)]`: compiler put scan_info at offset 296) or disrupts the layout of hot fields (`#[repr(C)]`: pushed `jit_dedup` to offset 312). fibonacci regressed from 1.49× to 1.56–1.73×. **Do not retry.** The ABI call overhead is smaller than the cache cost of any replacement.

- [x] **dynasmrt asm backend (Phase 4)** — implemented `asm_codegen.rs` (~970 lines): x86-64 hand-written JIT for Stage 4 stratum functions under `jit-asm` feature, tried first before Cranelift fallback. **Benchmark result (2026-03-14):** TC jit_hot/50 with asm = 259 µs vs Cranelift-only = 245 µs — asm is **6% slower**. Root cause: asm backend stores binding variables to rbp-relative stack slots (same memory-based pattern as pre-Phase-1 Cranelift), while Cranelift+Phase-1 Variables keeps them register-allocated. Falls through to Cranelift for triangle (3 clauses), connected_components (1-clause+bound_cols), fibonacci (Deref expr in condition). **Next:** to make asm faster than Cranelift, bind outer-loop-stable variables (x, y from clause0) into callee-saved registers (r12/r13/r15) rather than stack slots, eliminating the load/store pairs on every inner-loop iteration. Alternatively, evaluate QBE (~70% of LLVM quality, comparable compile speed) as a codegen backend.

- [x] **asm backend: recursive (IDB) inner clause support (2026-03-16)** — extended `asm_codegen.rs` to handle rules where inner clauses are IDB (appear in a head). Previously rejected with `clause1 is recursive`. Now emits a linked-list traversal: r15d=node index (callee-saved, survives calls), r11=values_ptr saved to `level_vptr_slot` before body calls, node advanced to next BEFORE processing body so sub_exhausted restores correct state. Removed eligibility rejection for recursive inner clauses. Triangle, fibonacci, TC all now handled by asm without Cranelift fallthrough. **Benchmark (2026-03-16):** triangle jit-asm 152 µs vs Cranelift 143 µs — asm still ~6% slower (stack-slot bindings vs register-allocated). SIGSEGV fix also applied: `if level >= 2 { save r15/rbx }` block moved before `emit_tuple_set_probe` fast-path early return — save slots were uninitialized when fully-bound clause took fast path, causing garbage loads in `sub_exhausted` (manifested as SIGSEGV in release bench, silent in debug due to zero-initialized stack).

- [x] **Cache `packed_data_ptr` for non-recursive rules** — `gen_full_scan_v3` calls `packed_data_ptr` inside the scan loop on every iteration to handle the case where head relation == clause relation (recursive insert can reallocate). For non-recursive rules (head ∉ clause relations), the pointer is stable. Pass a `is_recursive: bool` flag through codegen; only re-fetch inside the loop when true. Applies to Stage 3 and Stage 4. Would reduce inner loop body size by one runtime call.

#### Remaining inner-loop Rust calls (the real floor)

The true minimum is **zero per-iteration calls + one per rule-invocation for insertion**:

- [x] **Eliminate `packed_recent_idx`** — called once per recent-scan iteration to map `seq_idx → rel.recent[seq_idx]`. Fix: expose `recent.as_ptr()` via a new `packed_recent_ptr` helper (called once before the loop, same pattern as `packed_data_ptr` caching above); replace the per-iteration call with an inline `load ptr_type from (recent_ptr + i * ptr_size)`.

- [x] **Inline dedup probe for `packed_try_insert`** — at fixpoint, most head tuples are duplicates; `packed_try_insert` is called once per match but returns 0 (duplicate) the vast majority of the time. Implemented: `JitDedupTable` (open-addressed flat u32 hash table, stride = arity+1) embedded in `PackedStorage`, populated incrementally by `update_jit_indices()`. JIT loads `head_dedup_handles` pointer from ctx offset 40 at function entry, computes hash and probes table inline, jumps past `packed_try_insert` on duplicate (zero Rust calls for duplicates; one call for new tuples). Note: bulk-insert scratch buffer not implemented — new tuples still call `packed_try_insert` individually, but this is one call per new tuple (already minimal).

### Zero-overhead JIT (from scratch)

**Goal:** LLVM parity. Current gaps: triangle ~10.2× (333µs / 32.6µs), TC ~2.5×, fibonacci ~2.3×.

**Profiling results (2026-03-15, perf stat, n=20 triangle):**

| | JIT | ascent_macro | ratio |
|---|---|---|---|
| Instructions/iter | ~9.0M | ~1.0M | 9× |
| Cache misses/iter | ~13,100 | ~113 | **116×** |
| Cache miss rate | 14.8% | 1.6% | 9× |
| IPC | 1.97 | 2.21 | |

**Primary bottleneck is cache misses (116× gap), not instruction count.** The JIT's `JitHashIndex`
structure scatters each column index across 3 separate heap allocations (entries/ranges/values), and
edge has 4 indices (full+recent × col0+col1) = 12 separate tiny allocations that thrash the cache.
ascent_macro uses one compact open-addressed `HashMap` per join key — all data fits in 1-2 cache
lines per probe. Closing the cache-miss gap is the highest-value remaining work.

**Root cause of all remaining gaps** — three structural problems, in priority order:

1. **Cache-hostile index structure (PRIMARY)** — `JitHashIndex` with pointer-chased entries/ranges/values
   arrays causes 116× more cache misses per iteration than ascent_macro. Replace with a compact
   open-addressed hash map where each column's keys+ranges+values are stored in a single flat allocation
   (interleaved or packed), so a probe touches at most 2 cache lines. This is the JitColIndex struct
   in the "new architecture" section below.

2. **O(n) existence check for fully-bound clauses** — e.g. triangle's `edge(a,c)` where both a and c are already bound. Current asm probes the column index by key `a`, then scans all values for `c` — O(n). ascent_macro does an O(1) HashSet probe. With n≈10 values per key and 7,220 outer iterations, this is ~72,000 unnecessary comparisons per fixpoint iteration.

2. [ ] **`packed_try_insert` Rust call for new tuples** — JitHeadBuf approach was attempted but reverted (`cdd0791`): asm wrote tuples inline to dedup table without incrementing `JitDedupTable::count`, so `maybe_grow` never fired and the table filled to 100% capacity, causing infinite probe loops (manifested as hanging `tc_shared_jit` tests). New tuples currently route through `packed_try_insert` (one Rust call per new tuple). To re-eliminate this call: add `count` to `JitDedupHandle` so asm can increment it and trigger grow, or redesign around `JitRelData` (new architecture below).

3. **Register allocation** — outer-loop-stable variables spill to stack slots, causing load/store on every inner iteration. Step 3a partial fix exists but callee-saved register assignment is not complete for all rule depths.

**New architecture — JIT-native storage:**

All data structures are `#[repr(C)]` with compile-time-verified fixed offsets, so the JIT can address them without Rust calls. Slow paths (table growth, advance/swap) call Rust; everything in the hot loop is inline.

```
JitColIndex (#[repr(C)]):
  keys:    *mut u32  @ 0   — open-addressed hash table keys; u32::MAX = empty
  offsets: *mut u32  @ 8   — offsets[i] = start in vals for bucket i
  vals:    *mut u32  @ 16  — flat value array (col-values arity-2, row-idx otherwise)
  mask:    u32       @ 24  — cap-1 (power-of-2)
  len:     u32       @ 28  — occupied bucket count

JitTupleSet (#[repr(C)]):          — full-tuple existence/dedup
  slots:  *mut u32   @ 0   — inline storage; stride=arity; slots[i*arity]=tag (0=empty)
  mask:   u64        @ 8   — (cap_in_tuples - 1)
  arity:  u32        @ 16
  len:    u32        @ 20

JitRelData (#[repr(C)]):
  data:        *mut u32        @ 0   — packed tuples, stride=arity
  len:         u64             @ 8   — tuple count
  cap:         u64             @ 16  — capacity in tuples
  col_indices: *mut JitColIndex @ 24  — array[arity]
  tuple_set:   JitTupleSet     @ 32  — for existence checks + head dedup (56 bytes? → offset 88)
  arity:       u32             @ 88
```

**Inline operations in dynasm:**

- **Col index probe** (key → vals range): FxHash(`key`) & mask → linear probe keys[] for match or empty → read offsets[i], offsets[i+1] for range. ~10 instructions.
- **Tuple set probe** (existence check): FxHash of n u32s & mask → linear probe slots[] stride-arity. ~12 instructions for arity-2.
- **Tuple set insert** (new tuple): probe to find empty slot → write tuple inline. Bounds-check `len < cap*0.7`; if full, call `rust_tuple_set_grow`. ~15 instructions fast path.
- **Data write** (append to .new): write arity words at `data + len*arity`; increment len; if `len == cap`, call `rust_data_grow`. ~5 instructions.

**Hot loop for triangle rule** with new architecture (zero Rust calls until growth):
```
outer: for i in 0..edge_delta.len:
  a, b = edge_delta.data[i*2 .. i*2+2]
  probe edge_total.col_indices[0] for key=b → (vals_start, count)
  inner: for j in 0..count:
    c = edge_total.col_indices[0].vals[vals_start+j]
    probe edge_total.tuple_set for (a,c) → exists?    // O(1)!
    if !exists:
      write (a,c) to triangle_new.data, insert into triangle_new.tuple_set
advance: call rust_advance(ctx)   // once per fixpoint iter
```

**Implementation plan (2026-03-16, detailed):**

`JitColIndex`, `JitTupleSet`, `JitRelData` already exist in `storage.rs` as a blueprint — not yet wired. `JitColIndex.ranges` encodes `start | (count << 32)` per bucket (u64). `JitRelData` layout: `data@0`, `len@8`, `cap@16`, `col_indices@24`, `tuple_set@32` (embedded 24-byte JitTupleSet), `arity@56`.

Sequenced steps:
1. ✅ **storage.rs growth callbacks** — `JitColIndex`, `JitTupleSet`, `JitRelData` exist with `build_from_packed`; `JitRelData::new_empty` for write buffers. Growth callbacks not yet needed (write path still uses `packed_try_insert`).
2. ✅ **specialized.rs projection** — `jit_native: Option<JitNativeRelData>` on `PackedStorage`; `build_native_projection()` called from `advance_jit()`. Done.
3. ✅ **eval.rs native runtime** — `StratumStage4NativeCtx`, `build_stratum_stage4_native_runtime`, `jit_advance_native`, `StratumStage4NativeRuntime`. Done (commit `9a83b54`).
4. ✅ **asm read path** — `use_jit_native` flag, inline `JitColIndex` probe (Knuth hash, keys/ranges/vals), direct data load from `JitRelData.data`, head insertion via `ctx->head_specs[flat_hi].rel`. Upfront advance refreshes stale pointers. Native path active for EDB-inner-only strata. Done (commit `363a0fa`).
5. ✅ **asm write path** (~120 lines): replace `emit_heads`+`packed_try_insert` with direct stores to `head_jitrel.data`, inline `JitTupleSet` insert, `len++`, bounds-check → `call jit_rel_data_grow`. Two-phase dedup: probe `total.tuple_set` first (cross-iteration), then probe `new.tuple_set` (within-iteration). `JitNativeRelData{total, recent, new}` on `PackedStorage`; `advance_jit` flushes `new` buffer, rebuilds `jit_native`. Done (commit `62d0cc6`).
   - **Regression fixed (2026-03-16, commit `9c83f68`):** `advance_jit` was unconditionally calling `build_native_projection()` even when jit_native was None, causing the Cranelift path to pay full projection cost on every iteration. Fix: `advance_jit` only rebuilds when `jit_native.is_some()`; `build_stratum_stage4_native_runtime` (asm-only) explicitly initializes; `jit_advance_native` re-initializes after `Engine::clone()`.
   - **Benchmarks after regression fix (2026-03-16, n=20):** Cranelift `jit_hot` ~176 µs (3.5× vs macro ~49 µs). asm native `jit_hot` ~261–288 µs — native path activates (no skipped strata) but is **slower than Cranelift**. Root cause: `build_native_projection()` full rebuild of IDB total+indices on every active fixpoint iteration (growing tc relation over ~20 iterations = O(n²) sort+hash work). Cranelift path pays only `update_jit_indices()` (incremental). Fix: make IDB rebuild incremental (append-only updates to JitColIndex) or defer until needed.
6. [ ] **Cranelift parity** (~150 lines): mirror steps 4–5 in `packed_codegen.rs`.
7. [ ] **Dead code removal** (~50 lines): delete `JitHeadBuf`, `jit_flush_head_bufs`, `tuple_sets_buf`, `head_write_bufs`, `head_rel_ptrs` from `StratumStage4Ctx` once all tests pass.

Total: ~880 lines across 6 files. Steps 1–5 done + regression fixed; steps 6–7 remaining. **Blocker before step 6:** asm path must be faster than Cranelift first — otherwise mirroring to Cranelift regresses both.

**Incremental tuple_set in `extend_and_rebuild_indices` (2026-03-16):** When existing `total.tuple_set` capacity fits `new_len` at <70% load, skip full rebuild — insert only the `n_new` new tuples. When capacity is insufficient, reallocate + reinsert all. Also added `build_from_packed_no_tupleset` (skip tuple_set build for `recent` buffers that are only iterated, never probed) and `JitNativeRelData::deep_clone` (preserve prebuilt native projection across `PackedStorage::Clone`). **Results (2026-03-16):** triangle asm `jit_hot/20`: ~217µs → ~190µs (**12% improvement**, gap to Cranelift ~172µs: 52% → 10%); TC asm `jit_hot/50`: ~2.36ms → ~1.88ms (**20% improvement**).

Remaining 10% asm vs Cranelift gap for triangle: `update_jit_indices()` is called from `advance_jit()` on both Cranelift and asm paths, rebuilding `JitHashIndex` structures that the asm path never uses. Skipping it (`skip_jit_hash_indices` param) was attempted but broke TC correctness (empirically — the interaction was not fully understood). The asm path also still pays for `build_from_packed_no_tupleset` recent rebuild per iteration even though recent is only iterated.

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

- **3a — Depth-priority register assignment** (~150 lines): ✅ IMPLEMENTED (2026-03-16).
  `compute_var_locs()` assigns outer-loop-stable variables to callee-saved registers (r13 for
  first outer-fresh var when `use_recent0=false`, rbx for 1-clause rules), with stack slots for
  remaining vars. `emit_load_var`/`emit_store_var`/`emit_load_var_ecx` dispatch on `VarLoc::Reg`
  vs `VarLoc::Stack`. For 3-clause rules (triangle): r13 = var_a (total-scan variant), all others
  on stack. For 2-clause rules (TC/fibonacci): r13 = first outer var.

  Also implemented: **data_ptr0 cache slot** — outer-loop total-scan path was reloading
  `JitRelData.data` via 4-level pointer chain (ctx→scan_rels→JitRelData→data) on every iteration.
  Added `data_ptr0_slot()` (new stack slot at bottom of frame) and a pre-loop store before the
  outer scan; inner loop loads from that slot (1 load vs 4 serialized loads). Frame size increased
  by 8 bytes to accommodate.

  **Benchmark result (2026-03-16):** triangle jit-asm/20: ~294 µs → ~226 µs (**~23% improvement**),
  now within ~6% of Cranelift (226 µs vs 214 µs). Near parity.

  **Remaining gap (analysis):** For 3-clause rules, all 5 callee-saved regs (r12/r13/r14/r15/rbx)
  are occupied by loop machinery; only r13 is available for one variable. The variable bound at
  level 1 (c for triangle) is stored to stack and immediately reloaded for the level-2 existence
  check — a redundant store/load pair. Eliminating it would require either a 6th callee-saved reg
  (none exists on x86-64) or assigning level-1 vars to caller-saved regs restricted to call-free
  inner body spans. Not implemented.

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
- triangle `jit_hot/20`: ~380µs (ratio vs ascent_macro: 11.5× unchanged; ~1% noise improvement)
- TC `jit_hot/50`: ~268µs (~5% regression vs 254µs before; EDB index rebuild adds overhead)

**Assessment:** Contiguous inner loops give negligible wins at benchmark sizes (n≤30), not the
3-4× expected. Root cause: at n=20 (only ~190 edges in a complete graph), each key has 1-2
values on average — linked-list chain traversal overhead is negligible for short chains. The
load-use serialization argument only applies when chains are long (n>100+). The 11.5× gap
at benchmark sizes is dominated by Cranelift vs LLVM code quality. TC has a slight regression
from EDB detection and index-format conversion overhead on each stratum run.

**Step 5 — Dynasm col-value optimization** ✅ DONE (2026-03-15)

Arity-2 EDB contiguous indices now store the free column value directly (not tuple_idx),
eliminating `imul + add + data_ptr` dereference per inner iteration. Additional micro-opts:
pre-compute `values_base` before inner loop; keep count in `rbx` instead of stack slot;
sort per-key values during `build_contiguous`; early exit after emit_heads for fully-bound
existence-check clauses. Updated both asm and Cranelift backends + specialized.rs index build.

**Benchmark results after Step 5 (2026-03-15):**
- triangle `jit_hot/20`: ~372µs → ratio ~11.6×  (was 421µs / 13.1× before Steps 4–5)
- TC `jit_hot/50`: ~260µs → ratio ~2.5×  (Step 4 regression recovered; was 271µs)
- TC `jit_hot/100`: ~1.10ms → ratio ~2.7×  (was 1.18ms)

**Assessment:** ~8-10% improvement. Smaller than expected because:
1. Inner loop is L1-cache bound at n≤30; OOO execution already hides most load latency.
2. Triangle's dominant remaining costs: (a) level-2 existence check is O(n) sequential scan
   vs ascent_macro's O(1) HashSet probe; (b) 6,840 `packed_try_insert` calls.
3. TC inner loop is linked-list (IDB recursive, not col-value eligible).

**O(1) existence check for fully-bound clauses — implemented for arity 3**
`JitTupleSet` added to `storage.rs` (open-addressed set, stride=arity+1). Added
`tuple_sets_buf: *const *const JitTupleSet` field at offset 56 in `StratumStage4Ctx`
(flat parallel array to `handles_buf`). `emit_tuple_set_probe` in `asm_codegen.rs`
emits inline arity-3 probes (full 3-word hash + comparison, ~15 instructions, zero Rust
calls). `eval.rs` populates `tuple_sets_buf` for all fully-bound inner clauses.
Coverage: arity ≤ 2 falls back to existing col-value path (already O(1) amortized);
arity = 3 uses JitTupleSet; arity > 3 falls back (not yet implemented).

**Impact on triangle benchmark (arity-2 edge, n=20):** ~0% change — triangle uses
arity-2 edge relation for all clauses, so the JitTupleSet path is not triggered. The
existing col-value contiguous path already handles arity-2 fully-bound existence checks
efficiently. The JitTupleSet path will benefit programs with arity-3+ relations and
fully-bound inner clauses.

**`jit_is_sink` optimization — implemented 2026-03-15** ✅

Added `jit_is_sink: bool` flag to `PackedStorage`. `update_jit_indices()` returns early when
true, skipping all JIT column index building for relations that appear only in heads and never
in any body clause of the entire program. Set per-stratum via program-wide body-relation
analysis in `try_run_stratum_stage4`. **Triangle `jit_hot/20`: ~333 µs (was ~390 µs) — ~14%
improvement.** Eliminates ~20,520 hash insertions per advance() call (3 cols × 6,840 tuples).

**Lazy interpreter state sync — implemented 2026-03-15** ✅

`PackedStorage.insert_packed_raw` (called from `packed_try_insert` JIT FFI) now skips
updating `indices`, `value_data`, `source_tags` — interpreter-only structures unused during
pure-JIT stratum evaluation.  Added `ensure_interp_synced()` called on demand: before
interpreter stratum fallback, by `Engine::materialize()` (new public API), and at the start
of `try_insert_with_source`.  Tests call `engine.materialize()` before result comparison.
**Results (n=20, 2026-03-15):** triangle 359 µs → 298 µs = **17% improvement (9.2× vs macro)**;
fibonacci 18.2 µs → 13.3 µs = **27% improvement (1.6× vs macro)**.

**Step 6 (2026-03-16): remove dead `recent_set` + skip `recent_col_indices` in Stage 4 advance.**
Removed `FxHashSet<usize> recent_set` from `PackedStorage` (dead: `is_recent()` had no callers).
Added `advance_jit()` method that skips `recent_col_indices` rebuild (interpreter-only; Stage 4
uses `jit_recent_indices`); called from `jit_stratum_advance_s4` instead of `advance()`.
**Results (n=20, 2026-03-16):** triangle 298 µs → 246 µs = **17% improvement (7.7× vs macro)**;
fibonacci 13.3 µs → 12.0 µs = **10% improvement (1.4× vs macro)**.

**Residual gap:** triangle ~7.7×, fibonacci ~1.4×. The triangle gap is dominated by
instruction-count overhead (Cranelift vs LLVM code quality) in the inner join loop.
Options: (a) asm backend register assignment (Step 3a — ✅ done, see below); (b) eliminate
`packed_try_insert` call for new tuples (inline into JIT).

**Step 3a asm register assignment + data_ptr cache (2026-03-16):**
`compute_var_locs()` assigns outer-fresh variables to callee-saved registers (r13 when
`use_recent0=false`). Also added `data_ptr0_slot`: pre-loop stores `JitRelData.data` to a
new stack slot; inner loop loads from that slot (1 load) instead of 4-level pointer chain.
**triangle jit-asm/20: ~294 µs → ~226 µs (~23% improvement, now ~6% slower than Cranelift ~214 µs).**
Remaining gap: level-1-bound variables (c for triangle) still store/load via stack — all 5
callee-saved regs are occupied by loop machinery; no register remains for c.

**Pre-existing bug:** `tc_shared_jit` test hangs infinitely with `jit-asm` feature. Existed
before 2026-03-15 changes. Root cause unknown — dynasm TC path. Does not affect `jit+specialized`
(Cranelift backend).

### Relation storage optimizations

- [x] **Skip JIT index building for program-wide sink relations** — `jit_is_sink` flag on `PackedStorage`; `update_jit_indices()` returns early when set. 14% improvement on triangle. See above.

- [x] **Skip `recent_col_indices` rebuild in Stage 4** — Stage 4 JIT calls `advance_jit()` (not `advance()`) which skips `recent_col_indices` rebuild. `recent_col_indices` is interpreter-only; Stage 4 uses `jit_recent_indices`. Also removed dead `recent_set: FxHashSet<usize>` from `PackedStorage` (was maintained but never queried).
- [ ] **Lazy `recent_col_indices` rebuild for non-body relations** — `advance()` in `RelationStorage` unconditionally rebuilds recent indices even for sink relations. The right implementation is a new `ensure_recent_indices(&mut self)` called at eval-loop level only for relations that are body clauses in the current stratum. Impact limited to programs with head-only output relations; benchmarks (TC, triangles) don't benefit.

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
