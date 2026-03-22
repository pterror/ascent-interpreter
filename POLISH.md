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
