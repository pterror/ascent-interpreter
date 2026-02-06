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
- [ ] Built-in aggregators (count, sum, min, max, mean)

## Runtime

- [ ] REPL for interactive Datalog queries
- [ ] File-based program execution
- [ ] Incremental evaluation (add/retract facts)
- [ ] Query interface (ask questions about computed relations)

## JIT (Future)

- [ ] Cranelift backend for hot loops
- [ ] Compile frequently-used rules to native code
- [ ] Benchmark against interpreted mode

## Testing

- [ ] Port ascent test suite
- [ ] Comparison tests: run same program in ascent macro vs interpreter, compare results
- [ ] Performance benchmarks (transitive closure, graph algorithms)

## Documentation

- [ ] Usage examples
- [ ] Syntax reference
- [ ] Architecture overview
