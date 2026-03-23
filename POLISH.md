# Polish State

Created: 98bf464
Last run: 2026-03-23
Round: 4 (converged)
Project type: Rust multi-crate workspace (interpreter + JIT compiler for Ascent Datalog)

## Lenses
- api-clarity
- doc-coverage
- naming-consistency
- error-surface
- adversarial

## Scope
Public-facing crates (ascent-eval, ascent-interpreter, ascent-syntax, ascent-ir), docs/, README.md

## Findings — Round 1: 45/45 resolved

### HIGH — 10/10 resolved

- [DONE] `value.rs` — Arithmetic panics (div/0, shift, overflow) fixed with checked/wrapping ops
- [DONE] `eval.rs` — `Engine::insert` returns `Result<bool, EvalError>` (unknown relation, arity mismatch)
- [DONE] `eval.rs` — `relation()` takes `&self`, warns on stale data via `Cell<bool>`
- [DONE] `eval.rs` — `enable_jit()` returns `Result<(), EvalError>`
- [DONE] `ascent-ir` — `Program::from_ast` returns `Result<Program, String>`
- [DONE] `aggregators.rs` — Unknown aggregator warns to stderr
- [DONE] `README.md` — Full rewrite with installation, usage, examples, architecture
- [DONE] `docs/guide/embedding.md` — Library embedding guide

### MEDIUM — 20/20 resolved

- [DONE] `eval.rs` — Fixpoint iteration limit (10,000), range expansion limit (1M)
- [DONE] `eval.rs` — `type_registry` field private with getter, `constructor`/`destructor` naming
- [DONE] `eval.rs` — `Engine` stores `Program` internally, `run()` takes no args
- [DONE] `eval.rs` — `run_incremental` accepts `&[&str]` instead of `FxHashSet`
- [DONE] `lib.rs` — `Bindings`, `VarId`, `VarInterner`, `RelationStorage` removed from public exports
- [DONE] `syntax.rs` — `exp` → `expr`, `field_types` → `column_types` naming
- [DONE] `value.rs` — `DynValue` documented, module-level examples, `#[non_exhaustive]`, accessors
- [DONE] `docs/reference/types.md` — Fixed `Engine::new()` examples
- [DONE] `ascent-ir` — Crate-level usage example added
- [DONE] `ascent-ir` — `syn::Expr`/`Type`/`Pat` replaced with `IrExpr`/`IrType`/`IrPattern`
- [DONE] `value.rs` — `Value::Interned` hidden behind `#[non_exhaustive]` + accessor methods
- [DONE] `error.rs` — `EvalError` type with 12 variants, wired into insert/run/enable_jit
- [DONE] `eval_expr` — documented as future `Result<Value, EvalError>` migration target

### LOW — 15/15 resolved

- [DONE] Safety comments on all `unsafe impl Send/Sync` (13 locations)
- [DONE] `unwrap()` → `expect()` with context throughout
- [DONE] REPL error handling (read_line, :retract parse errors)
- [DONE] Naming: `is_wildcard`, `set_jit_compiler`, `try_cmp`, `column_types`
- [DONE] `Tuple` type alias documented
- [DONE] Bytecode: constant pool overflow check, stack underflow expect()
- [DONE] JIT: var_count > 10,000 guard
- [DONE] Mutex: poisoned-mutex recovery

### Conflicts
None identified.

## Findings — Round 2: 20/20 resolved

### HIGH — 4/4 resolved

- [DONE] `eval.rs` — `relation()` takes `&self`, warns on stale data
- [DONE] `eval.rs` — `insert_with_source` returns `Result<bool, EvalError>` with full validation
- [DONE] `value.rs` — `type_name()` delegates to `InternTable::type_name()` for interned values
- [DONE] `expr.rs` — Range size computed as `i128` to prevent overflow

### MEDIUM — 9/9 resolved

- [DONE] `value.rs` — `neg()` uses `wrapping_neg()`, `abs()` uses `wrapping_abs()`
- [DONE] `eval.rs` — `run_stratum_incremental` has iteration limit
- [DONE] `eval.rs` — `evaluate_rule` documented as future error propagation target
- [DONE] `aggregators.rs` — Documented as future `Result<AggResult, EvalError>` target
- [DONE] `error.rs` — `EvalError` is `#[non_exhaustive]`
- [DONE] `eval.rs` — `set_max_iterations` doc updated to match `Err` behavior
- [DONE] `value.rs` — `Value::downcast_custom()` added, `Engine::downcast_custom` deprecated
- [DONE] `lib.rs` — `OrderedFloat` re-exported
- [DONE] `ascent-ir` — Half-open range preserves original expression in Raw fallback

### LOW — 7/7 resolved

- [DONE] `value.rs` — Shift amounts checked via `u32::try_from` before cast
- [DONE] `eval.rs` — VarId and SourceId overflow debug_asserts
- [DONE] `main.rs` — `show_changes` uses `i64` subtraction
- [DONE] `ascent-ir` — Qualified call paths preserved in `IrExpr::Call`
- [DONE] `ascent-ir` — `unwrap()` → `expect()` in aggregation lowering
- [DONE] `eval.rs` — `coerce_to_col_type` behavior documented

### Conflicts
None identified.

## Findings — Round 3: 10/10 resolved

### HIGH — 3/3

- [DONE] `aggregators.rs` — `agg_count` returns `I64` instead of truncating to `i32`
- [DONE] `intern.rs` — `StringTable::resolve` uses `.get()` with descriptive panic
- [DONE] `README.md` — Code example fixed with `.unwrap()` on Result-returning methods

### MEDIUM — 4/4

- [DONE] `ascent-ir` — `IrLit::Int(i128, Option<String>)` preserves integer suffix
- [DONE] `intern.rs` — `Box::leak` memory lifecycle documented on `StringTable`
- [DONE] `expr.rs` — Oversized range returns `None` instead of `Some(vec![])`
- [DONE] `aggregators.rs` — Unknown aggregator behavior documented

### LOW — 3/3

- [DONE] `value.rs` — `as_i64()` handles `Isize`, `as_f64()` added, `cast_to` handles float-to-int
- [DONE] `aggregators.rs` — `agg_mean` handles floats, returns `F64` for mixed inputs
- [DONE] `value.rs` — `cast_to` supports `f32`/`f64` targets
