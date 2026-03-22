# Polish State

Created: 98bf464
Last run: 2026-03-22
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

### HIGH — all resolved

- [DONE] `value.rs:481-482` — Integer div/rem by zero panics — fixed with `checked_div`/`checked_rem` _(severity: high)_
- [DONE] `value.rs:487-488` — Shift overflow panics — fixed with `checked_shl`/`checked_shr` _(severity: high)_
- [DONE] `value.rs:478-480` — Integer overflow panics in debug — fixed with `wrapping_*` _(severity: high)_
- [DONE] `eval.rs:482` — `Engine::insert` warns on unknown relation _(severity: high)_
- [DONE] `eval.rs:472` — `relation()` now `&self` with stale-data warning via `Cell<bool>` _(severity: high)_
- [DONE] `eval.rs:384` — `enable_jit()` returns `Result<(), String>` _(severity: high)_
- [DONE] `ascent-ir/lib.rs:204` — `Program::from_ast` returns `Result<Program, String>` _(severity: high)_
- [DONE] `aggregators.rs:20` — Unknown aggregator warns to stderr _(severity: high)_
- [DONE] `README.md` — Rewritten with installation, usage, examples, features _(severity: high)_
- [DONE] `docs/` — Embedding guide added at `docs/guide/embedding.md` _(severity: high)_

### MEDIUM — 14 resolved, 6 remaining

- [DONE] `eval.rs:770` — Fixpoint iteration limit (10,000 default) _(severity: medium)_
- [DONE] `expr.rs:352` — Range expansion limit (1M) _(severity: medium)_
- [DONE] `eval.rs:309` — `type_registry` field now `pub(crate)` with getter _(severity: medium)_
- [DONE] `eval.rs:159` — `TypeRegistry::get` renamed to `constructor` _(severity: medium)_
- [DONE] `eval.rs:482` — `Engine::insert` warns on arity mismatch _(severity: medium)_
- [DONE] `lib.rs:53` — `Bindings`, `VarId`, `VarInterner` removed from public exports _(severity: medium)_
- [DONE] `lib.rs:56` — `RelationStorage` removed from public exports _(severity: medium)_
- [DONE] `syntax.rs:342,359` — `exp` → `expr` field naming _(severity: medium)_
- [DONE] `value.rs:31-38` — `DynValue` trait methods documented _(severity: medium)_
- [DONE] `docs/reference/types.md:82,118` — Fixed `Engine::new()` examples _(severity: medium)_
- [DONE] `eval.rs:554` — `Engine::run` takes redundant `&Program` param _(severity: medium)_
- [DONE] `eval.rs:588` — `run_incremental` leaks `FxHashSet` in public API _(severity: medium)_
- [PENDING] `ascent-ir/lib.rs:33` — `syn::Expr` leaks into IR public API _(severity: medium)_
- [PENDING] `value.rs:114` — `Value::Interned` variant exposes `Rc<dyn InternTable>` _(severity: medium)_
- [DONE] `eval.rs/syntax.rs/ir` — `field_types` vs `column_types` vs `col_types` inconsistency _(severity: medium)_
- [PENDING] `ascent-ir/lib.rs` — No crate-level usage example _(severity: medium)_
- [PENDING] `value.rs` — No module-level usage examples _(severity: medium)_
- [PENDING] `expr.rs:12` — `eval_expr` returns `Option<Value>` with no error diagnostics _(severity: medium)_
- [PENDING] No error types exist anywhere in the codebase _(severity: medium)_
- [PENDING] `syntax.rs` — Inconsistent `Node` suffix on AST types _(severity: medium)_

### LOW — 10 resolved, 5 remaining

- [DONE] `jit/asm_codegen.rs:75` — Safety comments on `unsafe impl Send/Sync` _(severity: low)_
- [DONE] `jit_index.rs:85,494,750,777` — Safety comments on `unsafe impl Send/Sync` _(severity: low)_
- [DONE] `jit/packed_helpers.rs:611,623,683` — Safety comments on `unsafe impl Send/Sync` _(severity: low)_
- [DONE] `eval.rs` — `unwrap()` → `expect()` with context on stratification/min_by_key _(severity: low)_
- [DONE] `main.rs:83` — REPL `read_line` error handling improved _(severity: low)_
- [DONE] `main.rs:217` — `:retract` now prints parse errors _(severity: low)_
- [DONE] `desugar.rs` — `unwrap()` → `expect("internal: generated syntax must parse")` _(severity: low)_
- [DONE] `eval.rs:403` — `with_jit_compiler` → `set_jit_compiler` _(severity: low)_
- [PENDING] `value.rs:368` — `Tuple` is a type alias, not a newtype _(severity: low)_
- [DONE] `value.rs:537` — `partial_cmp_val` → `try_cmp` _(severity: low)_
- [DONE] `syntax.rs:20` — `is_wild_card` → `is_wildcard` _(severity: low)_
- [PENDING] `bytecode.rs:58` — Constant pool index truncated to u16 _(severity: low)_
- [PENDING] `bytecode.rs:233` — Stack underflow panics via `unwrap()` _(severity: low)_
- [PENDING] `jit/asm_codegen.rs:82` — Stack frame overflow with many variables _(severity: low)_
- [PENDING] `eval.rs:873+` — `lock().unwrap()` on JIT mutex _(severity: low)_

### Conflicts
None identified.
