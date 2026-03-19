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

- [x] **sort-based `build_contiguous` + dedup capacity hints (Phase 2, 2026-03-17)** — (1) `build_contiguous` replaced `FxHashMap<u32, Vec<u32>>` intermediate with `sort_unstable()` + single-pass walk. One allocation per call vs ~N+1 (hashmap + N group Vecs). `build_contiguous` dropped from visible hotspot to <2% profile share. (2) `JitDedupTable::dedup_cap_hints` added to `JitCompiler`: after each stratum run, record peak dedup cap for each IDB head relation; before next run, call `reserve(hint)` to pre-size. Eliminates 6-8 reallocs on each fresh-engine iteration. **Results (2026-03-17):** triangle eval gap n=20: 4.1× → 3.4× (133µs vs 39µs); triangle eval gap n=30: 3.5× → 1.7× (276µs vs 161µs, 44% faster at n=30).

- [x] **`#[inline(always)]` `maybe_grow` + `#[cold]` `do_grow` + division elimination (2026-03-17)** — `maybe_grow` was called 1140 times per triangle iteration just for the load-factor check, costing a `len / stride` division each time (seen at 4.44% profile share). Fix: annotate `maybe_grow` `#[inline(always)]` (eliminates call overhead + lets the compiler see the check is always false and optimize it away), `do_grow` `#[cold]` (hints optimizer that grow path is rare), and replace `entries_vec.len() / stride` with `handle.cap as usize` (direct load, no division). **Results (2026-03-17):** triangle jit_hot/20 went from 158µs to 95µs eval (**40% improvement**, gap 4.1× → 2.7×).

- [x] **Tuple count hints for `packed_data`/`delta` pre-sizing (2026-03-17)** — extended `JitCompiler` with `tuple_count_hints: FxHashMap<String, u32>` tracking peak tuple counts per relation; pre-size `packed_data` and `delta` Vec capacity before each stratum run via `PackedStorage::reserve_tuples`. Eliminates realloc overhead for IDB relations with predictable sizes. **Results (2026-03-17):** ~6µs improvement (within noise), marginal win vs dedup cap hint elimination.

- [x] **Pointer-increment loop in sequential index scan (2026-03-17)** — replaced `j = 0..count` with `elem_ptr = start_ptr..end_ptr, += 4`. Hoists 5 per-iteration ops (2 uextends + 1 imul_imm + 2 iadd) into one-time setup. Applies to EDB inner-clause scans (triangle, TC edge clause, connected_components). **Result:** triangle eval n=20 ~95µs → ~70µs on clean system (~11-25% improvement, commit `f43db74`). **asm backend (2026-03-18, commit `70caf94`):** same optimization applied to `is_col_value=true` contiguous scan in `asm_codegen.rs`. Changes: (1) pre-loop: `lea r15, [r11+rsi*4]` (elem_ptr) + `lea rbx, [r15+rax*4]` (end_ptr) instead of saving vals_base to stack; (2) inner header: `cmp r15, rbx` pointer comparison instead of `cmp r15d, ebx` integer; (3) element load: `mov ecx, [r15]; add r15, 4` instead of `mov rax, [rbp+vs]; mov ecx, [rax+r15*4]; inc r15d`. Eliminates 1 stack load per inner iteration. Benchmark too noisy to measure cleanly (variance ±50% on this system).

- [x] **NativeCtx array-ptr cache in stack slots (2026-03-18, commit `180bd88`)** — each NativeCtx pointer chain (CTX_SLOT → ctx[offset] → array → JitRelData) is 3 serial loads (≈12 cycles minimum). Added three new stack slots (`native_total_rels_slot`, `native_head_rels_slot`, `native_head_total_rels_slot`) that cache `ctx[8]`, `ctx[24]`, `ctx[56]` before each rule variant in the native path. Reduces each chain from 3 to 2 loads at: (1) inner existence check probes (`emit_tuple_set_probe`), (2) head total dedup probe 2a, (3) head new dedup probe 2b, (4) grow-reload slow path. Saves ~4 cycles × 3 chains = ~12 cycles/inner iter for triangle. Frame size +24 bytes. Benchmark too noisy to measure on this system (triangle variance ±50%).

- [x] **Dedup probe CSE in gen_emit_heads_v3 (TRIED, REVERTED 2026-03-17)** — reuse `entry_ptr`/`entry_hash` from `probe_loop` in `probe_check_found`/`probe_verify` to avoid redundant imul_imm+iadd+load per probe. Regressed TC by 57% (102µs → 160µs). Root cause: keeping extra values live across the `probe_loop → probe_check_found` edge increases register pressure at the brif from ~6 to ~8 live values; Cranelift spills, costing more than the ~5 instructions eliminated. Do not retry without verifying register pressure stays within available registers.

- [x] **Defer stack stores in gen_emit_heads_v3 (TRIED, REVERTED 2026-03-17)** — attempted to move col_vals stack stores from before the probe into `call_insert` only (~16% of iterations). **Result: triangle +39%, TC +57%, fibonacci +18% regression.** Root cause: the early stores act as Cranelift "cheap spills" — the compiler writes col_vals to the stack slot immediately, freeing their registers for use in the probe loop; probe_verify reloads from the slot. Removing the early stores forces col_vals to stay live (in registers) across the entire probe loop, adding register pressure and causing spills of other values — same mechanism as the dedup probe CSE regression. The pattern: any change that increases the count of live SSA values across a hot loop body triggers spilling that costs more than the instructions eliminated. Do not retry without a Cranelift register-pressure analysis showing headroom.

- [x] **Fast-path aggregation for all-Var args + pre-reserve flat buffer (2026-03-17)** — added `collect_agg_flat_vars` in `stream_aggregation` for the common case where all agg.args are plain Vars. Eliminates per-match binding mutations (match_agg_args + undo/rollback) and nested Vec<Vec<Value>> allocations; uses a single flat Vec<Value> with stride = bound_vars.len() passed directly to apply_aggregator. Pre-sizes the flat buffer using the index lookup count (no reallocs). **Combined results:** CC/20 357µs → ~211µs (-41%, 7.3× → 4.3×); CC/40 2.15ms → ~875µs (-59%, 8.0× → 3.3×); CC/60 5.5ms → ~2.8ms (-49%). Triangle/TC/fibonacci unaffected (no aggregation in hot stratum). **Remaining CC gap:** outer clause `reach(x,_)` scans 380 tuples but produces only 20 distinct x values; stream_aggregation is called 380 times when 20 would suffice. Cache-based dedup was tried but key construction + HashMap cost exceeded savings at n=20. The correct fix is a "distinct-key outer scan" or aggregation-aware semi-naive optimization.

- [x] **Dedup outer full-scan on `meaningful_fresh_cols` (2026-03-17)** — added `meaningful_fresh_cols: Vec<(usize, VarId)>` to `CClause`, computed in Phase 4 of `optimize_body` via a backwards pass (heads ∪ body[i+1..] var union). In `stream_clause`, when `bound_cols` is empty and `meaningful_fresh_cols` is a proper subset of `fresh_cols` (wildcard detection) and conditions are empty, iterate only distinct values of the meaningful cols instead of all tuples. Single-meaningful-col fast path uses `FxHashSet<Value>` directly (no Vec). For CC `reach(x, _0)`, reduces stream_aggregation calls from ~380 to 20 at n=20. **Measured on quiet system (2026-03-17):** CC/20: 211µs → 129µs (-39%, 4.3× → 2.1×); CC/40: 875µs → 290µs (-67%, 3.3× → 1.0× **PARITY**); CC/60: ~2.8ms → 1.03ms (-63%, → 1.5×). Triangle/fibonacci unaffected (no aggregation).

- [ ] **Cache `packed_data_ptr` for recursive rules** — for recursive rules, `packed_data_ptr` is re-fetched per outer-scan tuple because `packed_try_insert` may reallocate the Vec. CLIF confirms fn2 (packed_data_ptr) is called once per outer loop iteration for recursive clauses. **Assessment:** savings < 0.5 µs at current scale. Triangle is unaffected (non-recursive). Defer — small savings.

- [ ] **Direct-load `packed_count` (Phase 3)** — replace `call fn1(rel, col)` (once per fixpoint iteration, full-scan path) with a direct `load.i64` at the known byte offset of `count`/`recent.len()` in `PackedStorage`. Not in the inner loop. Defer — negligible impact.

- [x] **PackedScanInfo (TRIED, REVERTED 2026-03-16)** — attempted to cache `data_ptr`, `count`, `recent_ptr`, `recent_count` in a new `PackedScanInfo` field in `PackedStorage` so the JIT could load directly instead of ABI calls. **Result: net regression in all configurations.** Root cause: the ABI calls (`packed_data_ptr`, `packed_count`, `packed_recent_ptr`) access `packed_data` (offset ~24), `count` (~72), `recent` (~112) — all in cache lines 0-1, which are already warm. Any `PackedScanInfo` placement either adds a cold cache line miss (`#[repr(Rust)]`: compiler put scan_info at offset 296) or disrupts the layout of hot fields (`#[repr(C)]`: pushed `jit_dedup` to offset 312). fibonacci regressed from 1.49× to 1.56–1.73×. **Do not retry.** The ABI call overhead is smaller than the cache cost of any replacement.

- [x] **dynasmrt asm backend (Phase 4)** — implemented `asm_codegen.rs` (~970 lines): x86-64 hand-written JIT for Stage 4 stratum functions under `jit-asm` feature, tried first before Cranelift fallback. **Benchmark result (2026-03-14):** TC jit_hot/50 with asm = 259 µs vs Cranelift-only = 245 µs — asm is **6% slower**. Root cause: asm backend stores binding variables to rbp-relative stack slots (same memory-based pattern as pre-Phase-1 Cranelift), while Cranelift+Phase-1 Variables keeps them register-allocated. Falls through to Cranelift for triangle (3 clauses), connected_components (1-clause+bound_cols), fibonacci (Deref expr in condition). **Next:** to make asm faster than Cranelift, bind outer-loop-stable variables (x, y from clause0) into callee-saved registers (r12/r13/r15) rather than stack slots, eliminating the load/store pairs on every inner-loop iteration. Alternatively, evaluate QBE (~70% of LLVM quality, comparable compile speed) as a codegen backend.

- [x] **asm backend: recursive (IDB) inner clause support (2026-03-16)** — extended `asm_codegen.rs` to handle rules where inner clauses are IDB (appear in a head). Previously rejected with `clause1 is recursive`. Now emits a linked-list traversal: r15d=node index (callee-saved, survives calls), r11=values_ptr saved to `level_vptr_slot` before body calls, node advanced to next BEFORE processing body so sub_exhausted restores correct state. Removed eligibility rejection for recursive inner clauses. Triangle, fibonacci, TC all now handled by asm without Cranelift fallthrough. **Benchmark (2026-03-16):** triangle jit-asm 152 µs vs Cranelift 143 µs — asm still ~6% slower (stack-slot bindings vs register-allocated). SIGSEGV fix also applied: `if level >= 2 { save r15/rbx }` block moved before `emit_tuple_set_probe` fast-path early return — save slots were uninitialized when fully-bound clause took fast path, causing garbage loads in `sub_exhausted` (manifested as SIGSEGV in release bench, silent in debug due to zero-initialized stack).

- [x] **Cache `packed_data_ptr` for non-recursive rules** — `gen_full_scan_v3` calls `packed_data_ptr` inside the scan loop on every iteration to handle the case where head relation == clause relation (recursive insert can reallocate). For non-recursive rules (head ∉ clause relations), the pointer is stable. Pass a `is_recursive: bool` flag through codegen; only re-fetch inside the loop when true. Applies to Stage 3 and Stage 4. Would reduce inner loop body size by one runtime call.

- [x] **SIGSEGV in Cranelift Stage 4 for mutually recursive strata (2026-03-17, fixed `69711da`)** — in strata with mutual recursion (e.g. `odd <-- even, even <-- odd`), body-clause relations are IDB (written by another rule in the same stratum) even though the CURRENT rule's head doesn't write to them. The `is_recursive` check in `stratum_codegen.rs` (`precomputed_packed_bufs`) and `gen_clauses_v3` (`effective_jit_rels`) only examined the current rule's heads, so mutual-recursion IDB relations were treated as EDB. `jit_rels` entries are null for IDB relations (no `jit_native`), causing a null pointer dereference → SIGSEGV. Fix: in `stratum_codegen.rs`, use strat-wide heads (all rules) for `is_recursive` in `precomputed_packed_bufs`. In `gen_clauses_v3`, derive the IDB flag from `precomputed_packed_bufs[clause_offset].is_none() && jit_rels.is_some()` to also set `is_recursive=true` for jit_rels usage. The asm backend (jit-asm feature) was not affected.

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
6. ✅ **Cranelift parity** (~150 lines): replace `packed_count`/`packed_data_ptr`/`packed_recent_ptr` callbacks with direct `JitRelData` field loads for EDB body clauses in Cranelift strata. Done (commits `57d7f36`, `8875009`, `d32ef25`). **Scope limitation:** only EDB body clauses benefit (IDB body clauses fall back to callbacks — per-iteration recent JitRelData rebuild costs more than savings). For current benchmarks: triangle and fibonacci at parity with pre-Step-6 (~208µs and ~18.6µs at n=20). Programs with EDB body clauses in Cranelift strata would benefit; none in current suite since triangle uses asm native.
7. ✅ **Dead code removal** (-198 lines, commit `4eda58e`): removed `JitHeadBuf` struct, `jit_head_buf_grow_and_insert`, `jit_flush_head_bufs`, `head_write_bufs`/`head_rel_ptrs`/`total_heads`/`_pad4` from `StratumStage4Ctx`, and `insert_packed_raw_no_dedup`. Also updated `jit_rel_specs`/`jit_rel_ptrs`/`total_jit_rels` offsets (88/96/104 → 64/72/80). Note: `tuple_sets_buf` was NOT dead (used by Cranelift codegen at offset 56) so it was kept.

Total: ~880 lines across 6 files (net -198 after Step 7). All steps done.

**Incremental tuple_set in `extend_and_rebuild_indices` (2026-03-16):** When existing `total.tuple_set` capacity fits `new_len` at <70% load, skip full rebuild — insert only the `n_new` new tuples. When capacity is insufficient, reallocate + reinsert all. Also added `build_from_packed_no_tupleset` (skip tuple_set build for `recent` buffers that are only iterated, never probed) and `JitNativeRelData::deep_clone` (preserve prebuilt native projection across `PackedStorage::Clone`). **Results (2026-03-16):** triangle asm `jit_hot/20`: ~217µs → ~190µs (**12% improvement**, gap to Cranelift ~172µs: 52% → 10%); TC asm `jit_hot/50`: ~2.36ms → ~1.88ms (**20% improvement**).

**Skip `update_jit_indices()` on asm native advance path (2026-03-16):** ✅ DONE. `advance_jit_inner(update_hash_indices: bool)` refactored from `advance_jit()`. `jit_advance_native` calls `advance_jit_skip_hash_indices()` which skips `update_jit_indices()` when safe. Safety condition: `jit_used_in_cranelift_strata=false` on the relation (no Cranelift stratum has the relation as a body-clause source). When a Cranelift stratum context is built, it marks its body-clause relations and calls `update_jit_indices()` immediately to catch up any stale indices. TC and fibonacci asm native are still blocked on recursive stratum ("clause 1 is recursive, linked-list not supported in native path"). **Results (2026-03-16):** triangle asm `jit_hot/20`: ~241µs → ~218µs (**~10% improvement**, asm now ~5% faster than Cranelift ~230µs). TC: no change (~2.46ms, recursive stratum uses Cranelift).

**IDB inner clauses in asm native path (2026-03-16) — ATTEMPTED, REVERTED.** Enabling IDB inner clauses caused fibonacci to regress 2.5× (50µs vs 20µs Cranelift). Root cause: `extend_and_rebuild_indices` rebuilds fib's JitColIndex every iteration — O(n log n) per step vs O(1) for Cranelift's linked-list inserts. The rejection was restored (commit `2a7fac1`): IDB inner clauses fall back to Cranelift where O(1) linked-list inserts win. **Current state (2026-03-16):** triangle asm `jit_hot/20` ~204µs (**10% faster than Cranelift ~225µs**, 5.5× vs macro ~37µs). Fibonacci asm ~19µs (at parity with Cranelift ~20µs, 2.1× vs macro ~9µs). IDB inner clauses (TC, fibonacci) remain on Cranelift fallback.

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

**Step 3 — Complete the asm backend for ~100% rule coverage** *(closes the codegen gap; end goal: delete Cranelift)*

Extends `asm_codegen.rs` with a proper register assigner and full pattern coverage. **Goal: remove
the Cranelift dependency entirely** — asm handles every rule shape, Cranelift fallback is deleted.
Current state (2026-03-17): IDB inner clauses now handled by the non-native asm linked-list path.
The stale rejection (which fell through to Cranelift) was removed after commit f2901fc fixed
expression handling in bound clause arg positions. TC jit_hot/50 at ~200µs (parity with ascent_macro
~196µs). Triangle gap remains 4.7× (~165µs vs ~35µs) — see Step 1 / Step 4.

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

- **3c — Expression completeness** ✅ IMPLEMENTED (2026-03-18, commit `2f222bb`): expanded
  `is_supported_packed_expr`/`is_supported_packed_binop` to match what `emit_expr` actually handles.
  Added `CExpr::DerefVar` (identity in packed repr), `CUnOp::Deref`, and all missing binops
  (Div, Rem, BitAnd, BitOr, BitXor, Shl, Shr, And, Or). Rules using these ops no longer fall
  back to the interpreter. Arbitrary function calls (user-defined fn pointers) remain unimplemented.

- **3d — Aggregation codegen** ✅ IMPLEMENTED (2026-03-18): `emit_aggregations` in `asm_codegen.rs`
  handles count/sum/min/max for pure-aggregation rules (0 positive clauses). Loads source relation
  via `load_rel_rdi!` at `rels[clause_count + not_count + agg_i]`, calls `packed_agg_count/sum_i32/
  max_i32/min_i32` helpers, stores result to `var_slot(result_var)`. AGG_EMPTY sentinel check skips
  head emission for sum/min/max on empty relations. `build_stratum_stage4_runtime` appends agg rel
  pointers after negation rels; `compile_stratum_stage4_native` skips native path for any rule with
  `CBodyItem::Aggregation(_)` to prevent the do-nothing native function from pre-empting the asm
  aggregation path. `packed_eligible_reason_inner` accepts pure-agg rules (count/sum/min/max with
  validated structure). 4 new tests: test_packed_jit_agg_{count,sum,min,max} all pass.

- **3e — Negation / anti-join** ✅ IMPLEMENTED (2026-03-18, commit `8c7a3e1`): `check_not_packed_1/2/3`
  extern C helpers probe `PackedStorage.jit_dedup`; `emit_not_probes` emits anti-join checks before
  head insertion at all call sites; `packed_eligible_reason_stage4` accepts "not" aggregations (arity
  ≤ 3, all-Var args); `build_stratum_stage4_runtime` appends negation rel pointers to `clause_rels`
  at `[clause_count + neg_i]`; `compile_stratum_stage4_native` skips native path for negation rules
  so the non-native asm path (which has probe code) is used. Bug: native path silently ignored
  anti-joins by wrapping with empty nots.

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

**IDB inner clause rejection — full history (2026-03-17):** The non-native asm linked-list path
was buggy — TC ran at 2.47ms (17.8×) and `tc_shared_jit` hung. The rejection was re-added so TC
fell through to Cranelift. Root cause of the original bug: bound clause arg expressions (like
`fib(nn-1, b)`) were mis-handled before commit f2901fc. After f2901fc, the rejection was confirmed
stale and permanently removed (commit f7e503d). **Result:** TC jit_hot/50 = ~200µs at parity with
ascent_macro ~196µs. All 261 tests pass. fibonacci/triangle unaffected.

**O(n³) jit_native rebuild regression (2026-03-17, fixed `ca33dbd`):** With `--features jit,jit-asm,specialized`, TC jit_hot/50 regressed to ~2ms (11-17×). Root cause: stratum 0 (native asm, copy rule) calls `build_stratum_stage4_native_runtime` which sets `path.jit_native = Some(full)`. Stratum 1 (non-native asm, recursive join) calls `jit_stratum_advance_s4` → `advance_jit()` → `advance_jit_inner(rebuild_jit_native=true)` which triggers `extend_and_rebuild_indices` — a full O(total) JitColIndex rebuild per step. Since non-native asm reads `packed_data_ptr` + `JitHashIndex` (never `jit_native`), the rebuild was pure waste: O(k×n) per step k → O(n³) total. Fix: `advance_jit_no_native_rebuild()` skips the `jit_native` rebuild block; `jit_stratum_advance_s4_inner` uses it instead of `advance_jit()`. **Result:** TC jit_hot/50 = ~170µs (1.49× vs ascent_macro ~114µs). All 157 tests pass.

**`jit_native.new` pre-sizing + skip unused JitHashIndex/JitColIndex builds (2026-03-19, commit `8337ea5`):** Three optimizations:
1. **`pre_size_new`** — `jit_native.new` (head buffer for new tuples per iteration) starts at capacity 16 and grows 7× for triangle n=20 (1140 tuples: 16→32→64→128→256→512→1024→2048). Each grow = malloc + memset + rehash = wasted work. New `JitRelData::pre_size_new(n)` method: realloc data buffer (no zeroing) + alloc/zeroed tuple_set to fit `n` tuples at <70% load. Called in `try_run_stratum_stage4` step 3a using `tuple_count_hints` (already tracking peak per-relation tuple counts from prior runs). Eliminates all 7 grow events per iteration.
2. **Skip `update_jit_indices()`** — `build_stratum_stage4_native_runtime` called `advance_jit()` (= `advance_jit_inner(true, true)`) which rebuilds 4 JitHashIndex structures (full+recent × 2 edge indices) on every advance. Native asm reads `JitColIndex` arrays only, never `JitHashIndex`. Changed both call sites to `advance_jit_skip_hash_indices()` = `advance_jit_inner(false, true)`. Eliminates 4 unused index rebuilds per iteration.
3. **Skip `JitColIndex` for `recent`** — in `build_native_projection` and `advance_jit_inner` IDB non-sink recent rebuild, `build_indices=true` was passed to `build_from_packed_no_tupleset`. The `recent` buffer is only sequentially iterated (outer scan); it is never key-probed. Changed to `build_indices=false`. Eliminates JitColIndex allocation+sort for all recent buffers.
**Results (2026-03-19, n=20 triangle):** jit_hot/20: 162µs → ~103µs (**-36%**, ratio 5.2× → **2.8×**). TC/50 ≈ parity (1.04×). fibonacci/20 ≈ 1.7×.
**Remaining triangle bottleneck (profiled 2026-03-19, perf record, 207µs run):** `extend_and_rebuild_indices` 9.48%, JIT code ~12%, `JitDedupTable::insert` 5.32%, allocator 8.8%, `advance_jit_inner` 3.73%. JIT inner loop itself ~12µs; ascent_macro at 35µs also spends 28.66% on `reserve_rehash`. The gap is mostly structural: JIT maintains JitNativeRelData (alloc+swap per iteration) that ascent_macro avoids. **Attempted: `total.tuple_set` pre-sizing (reverted, commit `20bc316`)** — pre-sizing `total` to 2048 slots (24KB) caused a real regression (103µs → 149µs). Root cause: large cold allocation pushes warm data out of cache; grow-on-demand starts with 16-slot hot table that fits in cache. Same rule as PackedScanInfo regression: "a large upfront alloc that pushes hot data cold costs more than the allocation overhead saved." Do not retry total pre-sizing.

**Attempted: skip `total.tuple_set` update for sink relations (reverted, session 2026-03-19)** — modified `extend_and_rebuild_indices` to accept `build_tuple_set: bool` and skip the tuple_set update section when false; passed `build_tuple_set = build_indices || (prev_count > 0)` at the call site. 5 tests failed: `test_jit_self_join`, `test_packed_jit_multi_bound_columns`, `test_stage4_triangle`, `test_stage4_tuple_set_probe_arity3`, plus one more. Root cause: the native asm Variant 2 (outer=total, inner=recent-designated) uses `total_rels` for inner existence checks. In the second fixpoint iteration, when `triangle.jit_native.total` has the iteration-1 tuples, those tuples ARE found by the inner probe. Without a valid `total.tuple_set`, the cross-iteration head dedup fails, `insert_packed_raw_native_flush` inserts duplicates unconditionally into `packed_data`. Fix requires either (a) changing the native asm head dedup to use `jit_dedup` instead of `total.tuple_set` (possible after hash unification — see below), or (b) fixing the inner probe to use `recent` instead of `total` (separate large change). The condition `prev_count > 0` was too permissive: it correctly handles iteration 2+ but fails iteration 1→2 because tuple_set is not built at end of iteration 1 (prev_count=0) yet is probed at start of iteration 2.

**Implemented: alias `total.tuple_set` to `jit_dedup` data after hash unification (session 2026-03-19)** — unified `jit_dedup_hash` to match `tuple_hash` (h=0x9e3779b9 init, sentinel=0, output 0→1 remap), updated non-native asm probe accordingly, added `alias_tuple_set` method to `JitRelData` (uses `_ts_slots_words=usize::MAX` sentinel instead of a new field to preserve 72-byte struct size), added `build_tuple_set: bool` param to `extend_and_rebuild_indices`, and alias after each `insert_packed_raw_native_flush` flush in `advance_jit_inner` for sink relations. All tests pass. **Performance result: no measurable improvement** — expected savings were ~7µs (7% of ~103µs from profile), but benchmark variance is ±5-10µs so the signal is swamped by noise. Triangle jit_hot/20 remains ~100-110µs (within prior range). The optimization is architecturally correct but the profiled 7% may have been measurement noise or was already eliminated by earlier bookkeeping optimizations. Change committed since it removes a genuine redundant computation and is structurally correct.

**Batch native flush (2026-03-19, commit `bd9df6b`)** — replaced 1140 individual `insert_packed_raw_native_flush` calls in `advance_jit_inner` with batched operations: one `extend_from_slice` for packed_data, one `delta.extend(range)`, then tight per-tuple jit_dedup insert loop (unavoidable per-tuple hash+insert). Eliminated `insert_packed_raw_native_flush` function entirely. **Result: within noise (87-125µs variance range), setup_only ≈ 25-36µs**. The benefit is architectural: removes 1140 Rust call overhead frames, enables sequential-index optimization below.

**Sequential-index recent fast path (2026-03-19, commit `6b5c380`)** — when `self.recent` indices are contiguous (last-first+1==n, always true after batch flush), pass a direct `&packed_data[start..end]` slice to `build_from_packed_no_tupleset` instead of gathering through an intermediate `Vec`. Eliminates ~1140 `extend_from_slice` calls + one Vec alloc per triangle iteration. Applied to all three recent-build sites. **Result: within noise (variance ±10µs).**

**EDB recent rebuild skip (2026-03-19, commit `8b2c8f5`)** — in the EDB advance path, when `self.recent.is_empty() && native.recent.len == 0`, skip the `drop+JitRelData::build_from_packed_no_tupleset` cycle. EDB relations have static facts; after iteration 1, `self.recent` is always empty, so we were paying 3 allocs + 3 frees per EDB relation per iteration for an identical result. **Result (2026-03-19): jit_hot/20 ≈ 79µs, setup ≈ 27µs → eval ≈ 52µs vs ascent_macro ≈ 40µs = ~1.3×. fibonacci/20 ≈ 13µs vs 9µs = ~1.5×. TC/50 ≈ 142µs vs 108µs = ~1.3×.** Significant improvement — the allocator overhead (8.8% of old profile) was the primary remaining bottleneck.

**Analysis of remaining gap (2026-03-19):** After all bookkeeping eliminations, the ~1.3–1.5× residual gap is dominated by JIT inner-loop code quality vs LLVM, not overhead. Specific costs identified:

- **`packed_try_insert` for non-native path (TC/fibonacci)** — asm already probes dedup inline; `packed_try_insert` is only called for genuinely new tuples. It redundantly re-probes (`insert_if_new`) before appending. Replacing with a `packed_insert_no_dedup` helper would save ~5–10 instructions per new tuple. TC/50: ~1225 new tuples across the full run → ~2–4µs total savings. **Below noise floor; not worth implementing.**

- **`jit_dedup.insert` batch in advance (triangle native path)** — 1140 per-tuple hash+insert calls after batch flush, inserting into a pre-sized table (~2048 slots, ~55% load). ~20 cycles/insert × 1140 = ~23K cycles = ~7.5µs per benchmark run. Unavoidable with current architecture (needed for cross-iteration dedup alias). A bulk-load rebuild (knowing all tuples unique) could save the per-slot collision checks, but savings are small (~2–3 instructions per insert at low load factor).

- **Scalar inner-loop column scan vs LLVM vectorized** — triangle inner loop iterates a flat u32 array (col-values in JitColIndex.vals). LLVM auto-vectorizes this with AVX2 (~8 values/iteration); our asm emits scalar. For triangle n=20, each outer iteration scans ~19 values per key → 190 outer × 19 inner = 3610 scalar comparisons. With AVX2 this is ~452 SIMD iterations. Estimated: ~3–5µs savings at n=20, larger gains at higher n. **Clear implementation path: emit `vmovdqu`/`vpcmpeqd`/`vmovmskps` for the contiguous values scan in `asm_codegen.rs`. ~150 lines.**

- **Hash probe overhead (JitColIndex key lookup)** — JitColIndex uses FxHash (Knuth multiply) for key→bucket lookup. `ascent_macro` uses `std::collections::HashMap` which LLVM can inline and optimize more aggressively. At ~190 outer iterations × 1 key probe each = 190 probes total — probably only 1–2µs at most.

- **`JitTupleSet` existence check (triangle level-2 clause)** — native asm probes `edge.total.tuple_set` for (a,c) using a 2-word hash + open-address probe. `ascent_macro` probes a Rust `HashSet<(u32,u32)>` that LLVM has inlined and optimized. Comparable per-probe cost, but LLVM may use SIMD for the comparison. Estimated: ~1–2µs gap at n=20.

**Note on JitTupleSet for arity-2 (2026-03-19):** The asm `emit_tuple_set_probe` already supports arity 1–3 and IS used for triangle's level-2 `edge(a,c)` existence check (fully-bound clause, `fresh_cols.is_empty()`). The TODO comment "~0% change for arity-2" was about the old Cranelift `gen_tuple_set_probe_v3` which only handled arity=3; the asm version is more general. The 1.3× benchmark already reflects this O(1) probe.

**Note on `packed_try_insert` elimination (2026-03-19):** Only non-native path (TC, fibonacci) calls `packed_try_insert`, and only for genuinely new tuples (asm already probes dedup inline for duplicates). Eliminating the redundant Rust re-probe would save ~5–10 cycles/new tuple. TC/50: ~1225 new tuples → ~2–4µs total savings. Below noise floor. Not worth implementing.

**perf stat comparison (2026-03-20, n=20 triangle, same binary same run):**

| Metric | jit_hot | ascent_macro | ratio |
|---|---|---|---|
| Wall time | ~117µs | ~46µs | **2.5×** |
| Instructions/iter | 2.92M | 1.03M | **2.83×** |
| Cycles/iter | 1.51M | 583K | **2.59×** |
| IPC | 1.93 | 1.77 | — |
| LLC misses/iter | 6,487 | 64 | **101×** |
| Branch-misses/iter | 7,636 | 2,100 | 3.6× |

(jit_hot: 40K iterations; ascent_macro: 111K iterations, same perf window size)

The JIT executes **2.83× more instructions** and hits main memory **101× more often** per triangle solve. Despite higher IPC (1.93 vs 1.77 — the CPU is not memory-stalled, it is doing work between misses), the instruction overhead dominates the gap. This is not a bookkeeping problem. The LLC miss rate (6,487 vs 64/iter) reflects hash table probing patterns: JIT probes JitTupleSet + JitColIndex + JitDedupHandle for every inner-loop iteration, each a separate pointer-chase; ascent_macro uses a single LLVM-inlined HashMap per join.

**Conclusion:** The bookkeeping overhead has been reduced to noise level. The remaining gap (2.5× wall-clock, 2.83× instructions) is structural — from the JIT generating more instructions per tuple and having worse cache locality than LLVM-compiled Rust. Closing it further requires either: (a) a smarter hash table design (fewer probes per tuple), (b) SIMD batching of the inner-loop scan (~150 lines, estimated 3–5µs at n=20), or (c) an LLVM-backed JIT. For the LSP use case (incremental eval, small deltas, n≤50), the current ratio may be acceptable.

**JitSwissTable SIMD existence probe (2026-03-20, commit `3392df1`):** Added `JitSwissTable` (hashbrown-style, 1-byte ctrl tags, SIMD group probe via pcmpeqb+pmovmskb) alongside `JitTupleSet` in `JitRelData` for arity 1-3. Native JIT path now uses SIMD probe for fully-bound arity-1/2 body clauses. **Result: no measurable hot-benchmark improvement** (triangle-20 jit_hot ≈ 120-150µs, ratio ≈ 3-4× vs ascent_macro ≈ 38-45µs — within variance of previous 2.5×). Root cause: for n=20 (190 edges), both JitTupleSet and JitSwissTable fit entirely in L1/L2 cache after criterion warmup, so LLC misses are not the hot-path bottleneck. The SIMD setup overhead (~15 extra instructions vs scalar probe) offsets any cache benefit. The Swiss table infrastructure may help for larger graphs where ctrl bytes (1 byte/slot) fit in L1 while full tuple data doesn't. Also fixed critical reg-aliasing bug: `rdi`/`edi` share the same register; the JitRelData pointer must be extracted into r8/r9 *before* loading args into `edi`/`esi` (both in the new scalar helper and the SIMD path). The old code happened to be correct by accident (pointer extracted before arg load in original order); the refactored helpers initially had the order wrong, causing wrong results for triangle tests.

### Relation storage optimizations

- [x] **Skip JIT index building for program-wide sink relations** — `jit_is_sink` flag on `PackedStorage`; `update_jit_indices()` returns early when set. 14% improvement on triangle. See above.

- [x] **Skip `recent_col_indices` rebuild in Stage 4** — Stage 4 JIT calls `advance_jit()` (not `advance()`) which skips `recent_col_indices` rebuild. `recent_col_indices` is interpreter-only; Stage 4 uses `jit_recent_indices`. Also removed dead `recent_set: FxHashSet<usize>` from `PackedStorage` (was maintained but never queried).
- [ ] **Lazy `recent_col_indices` rebuild for non-body relations** — `advance()` in `RelationStorage` unconditionally rebuilds recent indices even for sink relations. The right implementation is a new `ensure_recent_indices(&mut self)` called at eval-loop level only for relations that are body clauses in the current stratum. Impact limited to programs with head-only output relations; benchmarks (TC, triangles) don't benefit.

- [ ] **Sparse delta evaluation (LSP path)** — for incremental workloads where deltas are large relative to full relations, build selective delta indices only on columns actually used in downstream join clauses (determined at rule compile time). Avoids scanning delta tuples whose relevant columns don't match any current binding, reducing O(|delta|) inner loop iterations for non-matching tuples.

- [ ] **SIMD inner-loop scan in asm backend** — for EDB contiguous indices, the middle scan loop iterates a flat `u32` array. LLVM auto-vectorizes this with AVX2 (~8 values/iteration); our asm backend emits scalar code. Emitting AVX2 `vmovdqu`/`vpcmpeqd`/`vmovmskps` for the scan + scalar fallback for the existence check would close the remaining vectorizable fraction of the triangle gap. Prerequisite: `target_feature = avx2` detection at JIT init; fallback to scalar for non-AVX2 targets. Estimated: ~150 lines in `asm_codegen.rs`. Do after existence-check register spill is resolved (need to know how much gap remains before committing to SIMD complexity).

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
