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

## Findings — Round 1

### HIGH — 10/10 resolved

- [DONE] `value.rs` — Integer div/rem by zero, shift overflow, add/sub/mul overflow panics — fixed with checked/wrapping ops
- [DONE] `eval.rs` — `Engine::insert` warns on unknown relation and arity mismatch
- [DONE] `eval.rs` — `relation()` takes `&self`, warns on stale data via `Cell<bool>`
- [DONE] `eval.rs` — `enable_jit()` returns `Result<(), String>`
- [DONE] `ascent-ir` — `Program::from_ast` returns `Result<Program, String>`
- [DONE] `aggregators.rs` — Unknown aggregator warns to stderr
- [DONE] `README.md` — Full rewrite with installation, usage, examples, architecture
- [DONE] `docs/guide/embedding.md` — Library embedding guide

### MEDIUM — 16/20 resolved

- [DONE] `eval.rs` — Fixpoint iteration limit (10,000), range expansion limit (1M)
- [DONE] `eval.rs` — `type_registry` field private with getter, `constructor`/`destructor` naming
- [DONE] `eval.rs` — `Engine` stores `Program` internally, `run()` takes no args
- [DONE] `eval.rs` — `run_incremental` accepts `&[&str]` instead of `FxHashSet`
- [DONE] `eval.rs` — `Engine::insert` warns on arity mismatch
- [DONE] `lib.rs` — `Bindings`, `VarId`, `VarInterner`, `RelationStorage` removed from public exports
- [DONE] `syntax.rs` — `exp` → `expr`, `field_types` → `column_types` naming
- [DONE] `value.rs` — `DynValue` trait methods documented, module-level examples added
- [DONE] `ascent-ir` — Crate-level usage example added
- [DONE] `docs/reference/types.md` — Fixed `Engine::new()` examples
- [PENDING] `ascent-ir/lib.rs:33` — `syn::Expr` leaks into IR public API _(architectural — requires new expression AST)_
- [PENDING] `value.rs:114` — `Value::Interned` variant exposes `Rc<dyn InternTable>` _(architectural — requires Value enum redesign)_
- [PENDING] `expr.rs:12` — `eval_expr` returns `Option<Value>` with no error diagnostics _(requires EvalError type)_
- [PENDING] No error types exist anywhere in the codebase _(design work needed)_

### LOW — 15/15 resolved

- [DONE] Safety comments on all `unsafe impl Send/Sync` (13 locations)
- [DONE] `unwrap()` → `expect()` with context on stratification, min_by_key, desugar parse2
- [DONE] REPL: `read_line` error handling, `:retract` parse error reporting
- [DONE] `is_wild_card` → `is_wildcard`, `with_jit_compiler` → `set_jit_compiler`, `partial_cmp_val` → `try_cmp`
- [DONE] `Tuple` type alias documented
- [DONE] Bytecode constant pool overflow check, stack underflow `expect()`
- [DONE] JIT var_count > 10,000 guard
- [DONE] Mutex `.lock().unwrap_or_else(|e| e.into_inner())` for poisoned-mutex recovery

### Remaining (4 items — all architectural)

These require design decisions, not mechanical fixes:

1. `syn::Expr` in IR API — needs a new `IrExpr` AST independent of `syn`
2. `Value::Interned` exposure — needs Value enum redesign to hide internals
3. `eval_expr` error handling — needs `EvalError` type with diagnostic variants
4. No error types — needs `EvalError`, `LoweringError`, `ParseError` designed as a coherent set

### Conflicts
None identified.
