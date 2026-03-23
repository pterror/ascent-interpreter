# Polish State

Created: 98bf464
Last run: 2026-03-23
Round: 1
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

## Findings — Round 2

### HIGH

- [PENDING] `eval.rs:488` — `relation(&mut self)` still requires `&mut self` despite Cell flag — fix signature to `&self` _(severity: high)_
- [PENDING] `eval.rs:547` — `insert_with_source` bypasses all validation (no arity check, no unknown-relation error) _(severity: high)_
- [PENDING] `value.rs:502` — `type_name()` returns `"interned"` — should delegate to intern table _(severity: high)_
- [PENDING] `expr.rs:367` — Range size overflow: `i64::MIN..=i64::MAX` bypasses limit via negative size _(severity: high)_

### MEDIUM

- [PENDING] `value.rs:632,665` — `neg()` and `abs()` panic on MIN values in debug mode _(severity: medium)_
- [PENDING] `eval.rs:1810` — `run_stratum_incremental` has no iteration limit _(severity: medium)_
- [PENDING] `eval.rs:1834` — `evaluate_rule` silently drops expression failures — 4 EvalError variants dead code _(severity: medium)_
- [PENDING] `aggregators.rs:21` — Still uses `eprintln!` not `EvalError::UnknownAggregator` _(severity: medium)_
- [PENDING] `error.rs:7` — `EvalError` not `#[non_exhaustive]` _(severity: medium)_
- [PENDING] `eval.rs:424` — `set_max_iterations` doc says "prints warning" but returns `Err` _(severity: medium)_
- [PENDING] `eval.rs:476` — `downcast_custom` static on Engine, should be on Value _(severity: medium)_
- [PENDING] `lib.rs:54` — `OrderedFloat` not re-exported (needed for Value::F32/F64) _(severity: medium)_
- [PENDING] `ascent-ir:163` — Half-open range fallback discards present start/end _(severity: medium)_

### LOW

- [PENDING] `compiled.rs:919` + `value.rs:599` — Shift amount `as u32` truncates large values _(severity: low)_
- [PENDING] `eval.rs:121` — VarId u32 wraps on overflow _(severity: low)_
- [PENDING] `eval.rs:562` — SourceId u32 wraps on overflow _(severity: low)_
- [PENDING] `main.rs:327` — `usize as isize` subtraction overflow in `show_changes` _(severity: low)_
- [PENDING] `ascent-ir:178` — Qualified call path loses qualification in IrExpr::Call _(severity: low)_
- [PENDING] `ascent-ir:620` — `unwrap()` on `syn::parse2` in aggregation lowering _(severity: low)_
- [PENDING] `eval.rs:2812` — Silent type mismatch for I128 exceeding i64 in coerce_to_col_type _(severity: low)_

### Conflicts
None identified.
