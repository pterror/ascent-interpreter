# TODO

## Parser

- [x] Add desugaring pass (disjunctions → multiple rules, negation → aggregation, pattern args → if-let, wildcards → fresh vars)
- [ ] Fuzz test parser against ascent_macro to verify 1:1 syntax parity
- [ ] Property-based test: generate random valid ASTs, serialize, re-parse

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
- [ ] Incremental evaluation (add/retract facts)
- [ ] Query interface (ask questions about computed relations)

## JIT (Future)

- [ ] Cranelift backend for hot loops
- [ ] Compile frequently-used rules to native code
- [ ] Benchmark against interpreted mode

## Testing

- [x] Port ascent test suite (28 compat tests: fizzbuzz, factorial, negation, aggregation, joins, pattern matching, etc.)
- [x] Comparison tests: run same program in ascent macro vs interpreter, compare results (17 tests)
- [x] Performance benchmarks (transitive closure, triangles, connected components, fibonacci)

## Documentation

- [ ] Usage examples
- [ ] Syntax reference
- [ ] Architecture overview
