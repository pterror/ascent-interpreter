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

## Custom Types (BYOD)

- [x] `DynValue` trait + `Value::Custom` variant for user-defined types
- [x] `TypeRegistry` on Engine for registering custom constructors
- [x] Thread relation attributes through IR (`attrs: Vec<Attribute>`)
- [x] Pattern matching support for custom types (destructuring in clauses)
- [x] Serde-based automatic `DynValue` registration

## Interpreter Performance

Baseline: 15–550x slower than compiled ascent (Criterion benchmarks).
Current: 6–32x (after optimizations below).

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

### Diminishing returns (not planned)

- ~Expression compilation~ — CExpr already flattened syn::Expr; typical Datalog expressions are trivial, not the bottleneck
- ~Parallel SCC evaluation~ — strata are sequential by definition; intra-stratum parallelism is a research problem
- ~Cranelift backend / native codegen~ — effectively building a compiler; different project

## Testing

- [x] Port ascent test suite (28 compat tests: fizzbuzz, factorial, negation, aggregation, joins, pattern matching, etc.)
- [x] Comparison tests: run same program in ascent macro vs interpreter, compare results (17 tests)
- [x] Performance benchmarks (transitive closure, triangles, connected components, fibonacci)

## Documentation

- [x] Usage examples
- [x] Syntax reference
- [x] Architecture overview
