# Changelog

## [0.2.0-alpha.1] - 2026-04-26

### Added

- **x86-64 ASM JIT backend** (`jit-asm` feature, default) — compiles Datalog rule bodies to native x86-64 machine code via `dynasm`. Covers arithmetic, comparisons, aggregation (count/sum/min/max), negation/anti-join, and multi-stratum programs. Cranelift removed as a dependency.
- **Stratum-level JIT** — the entire semi-naive fixpoint loop for a stratum compiles to a single native function; per-rule `call_indirect` dispatch replaced by true inlining (Stages 1–4).
- **`PackedStorage`** (`specialized` feature, implied by `jit-asm`) — flat `u32` buffer layout for relations whose columns are all representable as 32-bit values (`i32`, `u32`, `bool`, `String`, arbitrary interned types). Eliminates `Value` enum allocation in the JIT hot path.
- **String interning** — `String` values are interned at insert time to a thread-local table; equality and hashing are integer compares; cloning is `Copy`.
- **`PackedType::Interned`** — extends `PackedStorage` to arbitrary `Hash + Eq` types registered via `TypeRegistry`. Custom Rust types participate in JIT evaluation without boxing on the hot path.
- **Bytecode expression compiler** — `CExpr` trees compile to a flat bytecode (LOAD_VAR, LOAD_CONST, ADD, CMP_EQ, …) evaluated by a tight dispatch loop; replaces tree-walk evaluation.
- **Incremental evaluation** — `Engine` persists relation state across `run()` calls. Facts are tagged with a `SourceId`; bulk retract/re-assert by source is O(delta). Strata invalidation reruns only affected strata on change; non-monotone strata (negation/aggregation) are cleared and re-derived.
- **Lattice support** — `Dual<T>` value type with merge-by-key semantics; `?pattern` prefix for lattice value matching in clause heads.
- **Custom types (BYOD)** — `DynValue` trait + `Value::Custom` variant; `TypeRegistry` on `Engine` for registering constructors/destructors. Optional `serde` feature for automatic registration.
- **SCC-based stratification** — full dependency-graph stratification via `petgraph`; replaces the earlier 2-stratum (base → aggregation) heuristic.
- **`EvalError` public type** — typed errors from `Engine::run()` and `Engine::insert()` instead of `Box<dyn Error>`.
- **REPL commands**: `:retract`, `:undo`, `:query rel(pat, …)` with wildcard and literal filters.

### Performance (x86-64, triangle query n=20)

| Path | Wall time | vs `ascent_macro` |
|---|---|---|
| Interpreter | ~1.5 ms | ~30× |
| JIT (eval only) | ~59 µs | ~1.3× |
| JIT (total incl. setup) | ~110 µs | ~2.1× |

### Changed

- Sub-crates (`ascent-syntax`, `ascent-ir`, `ascent-eval`) merged into the root crate as modules; only `ascent-interpreter` is published.
- `Engine::relation()` now takes `&self` (was `&mut self`).
- `TypeRegistry::get` renamed to `TypeRegistry::constructor`.
- `IfLetClause` / `LetClause` `.exp` field renamed to `.expr`.

### Fixed

- Multi-rule stratum SIGSEGV in native JIT when EDB relations are shared across rules.
- Ordering comparisons on interned columns now use a `jit_cmp_interned` trampoline (raw u32 ID comparison gave wrong results for `Lt`/`Le`/`Gt`/`Ge`).
- Fully-bound existence check paths now emit `clause.conditions` (previously dropped conditions merged by `optimize_body` Phase 2, silently breaking `if a < b` guards).
- Arithmetic panics on overflow/underflow replaced with wrapping semantics.
- Unsuffixed integer literals coerce to the declared column type.

## [0.1.5] - 2026-03-26

Patch: gate `Debug` impl and JIT helper functions behind `target_arch = "x86_64"`.

## [0.1.4] - 2026-03-25

Patch: gate x86-64 JIT code behind `#[cfg(target_arch = "x86_64")]`; emit `jit_cmp_interned` trampoline for ordering comparisons on interned columns; fix skipped `clause.conditions` in fully-bound fast paths.

## [0.1.3] - 2026-03-21

Patch: fix multi-rule stratum SIGSEGV; merge-intersection optimization for sorted EDB scans.

## [0.1.2] - 2026-03-19

Patch: fix default features; polish API surface; add library embedding guide.
