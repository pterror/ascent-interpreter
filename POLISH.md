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

### HIGH

- [DONE] `value.rs:481-482` — Integer div/rem by zero panics — fixed with `checked_div`/`checked_rem` (commit `a30eb60`) _(severity: high)_
- [DONE] `value.rs:487-488` — Shift overflow panics — fixed with `checked_shl`/`checked_shr` (commit `a30eb60`) _(severity: high)_
- [DONE] `value.rs:478-480` — Integer overflow panics in debug — fixed with `wrapping_*` (commit `a30eb60`) _(severity: high)_
- [PENDING] `eval.rs:482` — `Engine::insert` silently discards inserts to unknown relations _(severity: high)_
- [PENDING] `eval.rs:472` — `materialize()` footgun: must call before reading data _(severity: high)_
- [PENDING] `eval.rs:384` — `enable_jit()` prints to stderr on failure instead of returning Result _(severity: high)_
- [PENDING] `ascent-ir/lib.rs:204` — `Program::from_ast` panics on un-desugared input _(severity: high)_
- [PENDING] `aggregators.rs:20` — Unknown aggregator silently returns empty results _(severity: high)_
- [DONE] `README.md` — Rewritten with installation, usage, examples, features (commit `4d26761`) _(severity: high)_
- [PENDING] `docs/` — No embedding/library-usage guide _(severity: high)_

### MEDIUM

- [DONE] `eval.rs:770` — No fixpoint iteration limit — added `max_iterations` default 10,000 (commit `6ac3f60`) _(severity: medium)_
- [DONE] `expr.rs:352` — Unbounded range expansion — added MAX_RANGE_SIZE=1M limit (commit `6ac3f60`) _(severity: medium)_
- [PENDING] `eval.rs:309` — `type_registry` pub field should be private _(severity: medium)_
- [PENDING] `eval.rs:159` — `TypeRegistry::get` should be `constructor` (matches `destructor`) _(severity: medium)_
- [PENDING] `eval.rs:554` — `Engine::run` takes redundant `&Program` param _(severity: medium)_
- [PENDING] `eval.rs:588` — `run_incremental` leaks `FxHashSet` in public API _(severity: medium)_
- [PENDING] `eval.rs:482` — `Engine::insert` accepts wrong-arity tuples silently _(severity: medium)_
- [PENDING] `lib.rs:53` — Internal types (`Bindings`, `VarId`) re-exported as public _(severity: medium)_
- [PENDING] `lib.rs:56` — `RelationStorage` re-exported (semi-naive internals) _(severity: medium)_
- [PENDING] `ascent-ir/lib.rs:33` — `syn::Expr` leaks into IR public API _(severity: medium)_
- [PENDING] `value.rs:114` — `Value::Interned` variant exposes `Rc<dyn InternTable>` _(severity: medium)_
- [PENDING] `syntax.rs:342,359` — `exp` vs `expr` vs `cond` field naming inconsistency _(severity: medium)_
- [PENDING] `eval.rs/syntax.rs/ir` — `field_types` vs `column_types` vs `col_types` inconsistency _(severity: medium)_
- [PENDING] `value.rs:31-38` — `DynValue` trait methods lack doc comments _(severity: medium)_
- [DONE] `docs/reference/types.md:82,118` — Fixed `Engine::new()` examples (commit `7c834ad`) _(severity: medium)_
- [PENDING] `ascent-ir/lib.rs` — No crate-level usage example _(severity: medium)_
- [PENDING] `value.rs` — No module-level usage examples _(severity: medium)_
- [PENDING] `expr.rs:12` — `eval_expr` returns `Option<Value>` with no error diagnostics _(severity: medium)_
- [PENDING] No error types exist anywhere in the codebase _(severity: medium)_
- [PENDING] `syntax.rs` — Inconsistent `Node` suffix on AST types _(severity: medium)_

### LOW

- [PENDING] `value.rs:368` — `Tuple` is a type alias, not a newtype (no arity validation) _(severity: low)_
- [PENDING] `value.rs:537` — `partial_cmp_val` naming — consider implementing `PartialOrd` _(severity: low)_
- [PENDING] `eval.rs:403` — `with_jit_compiler` name suggests builder pattern — rename to `set_jit_compiler` _(severity: low)_
- [PENDING] `syntax.rs:20` — `is_wild_card` spelling (should be `is_wildcard`) _(severity: low)_
- [PENDING] `bytecode.rs:58` — Constant pool index truncated to u16 silently _(severity: low)_
- [PENDING] `bytecode.rs:233` — Stack underflow panics via `unwrap()` on `pop()` _(severity: low)_
- [PENDING] `jit/asm_codegen.rs:82` — Stack frame overflow with many variables (no limit check) _(severity: low)_
- [PENDING] `jit/asm_codegen.rs:75` — `unsafe impl Send/Sync` without safety documentation _(severity: low)_
- [PENDING] `jit_index.rs:85,494,750,777` — `unsafe impl Send/Sync` without safety docs _(severity: low)_
- [PENDING] `jit/packed_helpers.rs:611,623,683` — `unsafe impl Send/Sync` without safety docs _(severity: low)_
- [PENDING] `eval.rs:873+` — `lock().unwrap()` on JIT mutex — poisoned mutex = panic _(severity: low)_
- [PENDING] `main.rs:83` — REPL `read_line` swallows I/O errors _(severity: low)_
- [PENDING] `main.rs:217` — `:retract` rebuild silently swallows parse errors _(severity: low)_
- [PENDING] `desugar.rs:149+` — `.unwrap()` on `parse2(quote!{})` — add `.expect()` context _(severity: low)_
- [PENDING] `eval.rs:2123,2424` — `.unwrap()` on `min_by_key` (logically safe) — use `.expect()` _(severity: low)_

### Conflicts
None identified.
