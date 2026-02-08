//! Compatibility tests ported from ascent's test suite.
//!
//! These validate that the interpreter produces the same results as ascent_macro.
//! Each test documents the original source location in the ascent repo.

use std::collections::HashSet;

use ascent_eval::Engine;
use ascent_eval::value::Value;
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

fn run(input: &str) -> Engine {
    let ast: AscentProgram = syn::parse_str(input).unwrap();
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    engine.run(&program);
    engine
}

fn run_with_facts(input: &str, facts: Vec<(&str, Vec<Vec<Value>>)>) -> Engine {
    let ast: AscentProgram = syn::parse_str(input).unwrap();
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    for (rel, tuples) in facts {
        for tuple in tuples {
            engine.insert(rel, tuple);
        }
    }
    engine.run(&program);
    engine
}

/// Wrapper around HashSet<Vec<Value>> that accepts `&[Value]` in `contains`,
/// enabling `&[Value; N]` → `&[Value]` coercion at call sites.
#[derive(Debug)]
struct RelSet(HashSet<Vec<Value>>);

impl RelSet {
    fn contains(&self, tuple: &[Value]) -> bool {
        self.0.contains(tuple)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl PartialEq for RelSet {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl FromIterator<Vec<Value>> for RelSet {
    fn from_iter<I: IntoIterator<Item = Vec<Value>>>(iter: I) -> Self {
        RelSet(iter.into_iter().collect())
    }
}

fn rel_set(engine: &Engine, name: &str) -> RelSet {
    RelSet(
        engine
            .relation(name)
            .unwrap()
            .iter()
            .map(|t| t.to_vec())
            .collect::<HashSet<_>>(),
    )
}

fn i(v: i32) -> Value {
    Value::I32(v)
}

// ─── FizzBuzz ────────────────────────────────────────────────────────
// From: ascent/examples/fizz_buzz.rs
// Tests: conditions, negation, multiple strata

#[test]
fn test_fizzbuzz() {
    let engine = run(r#"
        relation number(i32);
        relation divisible(i32, i32);
        relation fizz(i32);
        relation buzz(i32);
        relation fizz_buzz(i32);
        relation other(i32);

        number(x) <-- for x in 1..16;
        divisible(x, 3) <-- number(x), if x % 3 == 0;
        divisible(x, 5) <-- number(x), if x % 5 == 0;
        fizz(x) <-- number(x), divisible(x, 3), !divisible(x, 5);
        buzz(x) <-- number(x), !divisible(x, 3), divisible(x, 5);
        fizz_buzz(x) <-- number(x), divisible(x, 3), divisible(x, 5);
        other(x) <-- number(x), !divisible(x, 3), !divisible(x, 5);
    "#);

    let fizz = rel_set(&engine, "fizz");
    assert_eq!(
        fizz,
        [3, 6, 9, 12].into_iter().map(|v| vec![i(v)]).collect()
    );

    let buzz = rel_set(&engine, "buzz");
    assert_eq!(buzz, [5, 10].into_iter().map(|v| vec![i(v)]).collect());

    let fizz_buzz = rel_set(&engine, "fizz_buzz");
    assert_eq!(fizz_buzz, [vec![i(15)]].into_iter().collect());

    let other = rel_set(&engine, "other");
    assert_eq!(
        other,
        [1, 2, 4, 7, 8, 11, 13, 14]
            .into_iter()
            .map(|v| vec![i(v)])
            .collect()
    );
}

// ─── Even/Odd with negation ─────────────────────────────────────────
// From: ascent/examples/ascent_negation_clause.rs
// Tests: negation as absence

#[test]
fn test_even_odd_negation() {
    let engine = run(r#"
        relation number(i32);
        relation even(i32);
        relation odd(i32);

        number(x) <-- for x in 1..6;
        even(x) <-- number(x), if x % 2 == 0;
        odd(x) <-- number(x), !even(x);
    "#);

    let even = rel_set(&engine, "even");
    assert_eq!(even, [2, 4].into_iter().map(|v| vec![i(v)]).collect());

    let odd = rel_set(&engine, "odd");
    assert_eq!(odd, [1, 3, 5].into_iter().map(|v| vec![i(v)]).collect());
}

// ─── Nested generators ──────────────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_generators)
// Tests: variable-dependent range expressions

#[test]
fn test_nested_generators() {
    let engine = run(r#"
        relation pair(i32, i32);
        pair(x, y) <-- for x in 0..10, for y in (x + 1)..10;
    "#);

    let pair = engine.relation("pair").unwrap();
    // Combinations where x < y from 0..10: C(10,2) = 45
    assert_eq!(pair.len(), 45);
    assert!(pair.contains(&[i(0), i(1)]));
    assert!(pair.contains(&[i(0), i(9)]));
    assert!(pair.contains(&[i(8), i(9)]));
    assert!(!pair.contains(&[i(5), i(5)])); // x < y, not x == y
    assert!(!pair.contains(&[i(9), i(0)])); // not reversed
}

// ─── Cross product ──────────────────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_multiple_rules)
// Tests: join between independently generated relations

#[test]
fn test_cross_product() {
    let engine = run(r#"
        relation foo(i32, i32);
        relation bar(i32, i32);
        relation baz(i32, i32, i32, i32);

        foo(x, x + 1) <-- for x in 0..5;
        bar(11, 12);
        bar(12, 13);
        baz(a, b, c, d) <-- foo(a, b), bar(c, d);
    "#);

    let foo = engine.relation("foo").unwrap();
    assert_eq!(foo.len(), 5);

    let baz = engine.relation("baz").unwrap();
    assert_eq!(baz.len(), 10); // 5 × 2
    assert!(baz.contains(&[i(0), i(1), i(11), i(12)]));
    assert!(baz.contains(&[i(4), i(5), i(12), i(13)]));
}

// ─── Arithmetic joins ───────────────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl2)
// Tests: join with condition and arithmetic in head

#[test]
fn test_arithmetic_join() {
    let engine = run_with_facts(
        r#"
        relation foo1(i32, i32);
        relation foo2(i32, i32);
        relation bar(i32, i32);
        bar(x, y + z) <-- foo1(x, y), if x != 0, foo2(y, z);
    "#,
        vec![
            (
                "foo1",
                vec![vec![i(1), i(2)], vec![i(10), i(20)], vec![i(0), i(2)]],
            ),
            (
                "foo2",
                vec![
                    vec![i(2), i(4)],
                    vec![i(2), i(1)],
                    vec![i(20), i(40)],
                    vec![i(20), i(0)],
                ],
            ),
        ],
    );

    let bar = rel_set(&engine, "bar");
    assert!(bar.contains(&[i(1), i(6)])); // 1, 2+4
    assert!(bar.contains(&[i(1), i(3)])); // 1, 2+1
    assert!(bar.contains(&[i(10), i(60)])); // 10, 20+40
    assert!(bar.contains(&[i(10), i(20)])); // 10, 20+0
    assert_eq!(bar.len(), 4);
}

// ─── Factorial ──────────────────────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_fact)
// Tests: recursive computation with arithmetic

#[test]
fn test_factorial() {
    let engine = run(r#"
        relation do_fac(i32);
        relation fac(i32, i32);

        do_fac(10);
        do_fac(x - 1) <-- do_fac(x), if x > 0;
        fac(0, 1);
        fac(x, x * sub1fac) <-- do_fac(x), if x > 0, fac(x - 1, sub1fac);
    "#);

    let fac = engine.relation("fac").unwrap();
    assert!(fac.contains(&[i(0), i(1)]));
    assert!(fac.contains(&[i(1), i(1)]));
    assert!(fac.contains(&[i(5), i(120)]));
    assert!(fac.contains(&[i(10), i(3628800)]));
}

// ─── Multi-head rules ───────────────────────────────────────────────
// Tests: inserting into multiple relations from a single rule

#[test]
fn test_multi_head_rule() {
    let engine = run(r#"
        relation source(i32, i32);
        relation left(i32);
        relation right(i32);

        source(1, 10);
        source(2, 20);
        source(3, 30);
        left(x), right(y) <-- source(x, y);
    "#);

    let left = rel_set(&engine, "left");
    assert_eq!(left, [1, 2, 3].into_iter().map(|v| vec![i(v)]).collect());

    let right = rel_set(&engine, "right");
    assert_eq!(
        right,
        [10, 20, 30].into_iter().map(|v| vec![i(v)]).collect()
    );
}

// ─── Self-join (repeated variables) ─────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_repeated_vars)
// Tests: repeated variable in clause args → equality check

#[test]
fn test_self_loop_repeated_vars() {
    let engine = run_with_facts(
        r#"
        relation edge(i32, i32);
        relation self_loop(i32);
        self_loop(x) <-- edge(x, x);
    "#,
        vec![(
            "edge",
            vec![
                vec![i(1), i(2)],
                vec![i(2), i(2)],
                vec![i(3), i(3)],
                vec![i(4), i(5)],
            ],
        )],
    );

    let self_loop = rel_set(&engine, "self_loop");
    assert_eq!(self_loop, [2, 3].into_iter().map(|v| vec![i(v)]).collect());
}

// ─── Disjunction ────────────────────────────────────────────────────
// From: ascent/examples/ascent_disjunction_clause.rs
// Tests: disjunction desugaring

#[test]
fn test_disjunction() {
    let engine = run(r#"
        relation number(i32);
        relation even(i32);
        relation square(i32);
        relation even_or_square(i32);

        number(x) <-- for x in 1..11;
        even(x) <-- number(x), if x % 2 == 0;
        square(x * x) <-- number(x);
        even_or_square(x) <-- (even(x) | square(x));
    "#);

    let eos = rel_set(&engine, "even_or_square");
    // Even: 2, 4, 6, 8, 10
    // Square: 1, 4, 9, 16, 25, 36, 49, 64, 81, 100
    // Union: 1, 2, 4, 6, 8, 9, 10, 16, 25, 36, 49, 64, 81, 100
    assert!(eos.contains(&[i(2)]));
    assert!(eos.contains(&[i(4)])); // both even and square
    assert!(eos.contains(&[i(9)])); // square only
    assert!(eos.contains(&[i(10)])); // even only
    assert!(!eos.contains(&[i(3)])); // neither
}

// ─── Disjunction with recursion ─────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_disjunctions2)
// Tests: disjunction in recursive transitive closure

#[test]
fn test_disjunction_transitive() {
    let engine = run(r#"
        relation road(i32, i32);
        relation rail(i32, i32);
        relation connected(i32, i32);

        road(1, 2);
        road(2, 3);
        rail(3, 4);
        rail(4, 5);

        connected(x, y) <-- (road(x, y) | rail(x, y));
        connected(x, z) <-- connected(x, y), (road(y, z) | rail(y, z));
    "#);

    let conn = rel_set(&engine, "connected");
    // 1→2 (road), 2→3 (road), 3→4 (rail), 4→5 (rail)
    // Plus transitive: 1→3, 1→4, 1→5, 2→4, 2→5, 3→5
    assert!(conn.contains(&[i(1), i(2)]));
    assert!(conn.contains(&[i(1), i(5)])); // 1→2→3→4→5
    assert!(conn.contains(&[i(3), i(5)])); // 3→4→5
    assert_eq!(conn.len(), 10); // 4 direct + 6 transitive
}

// ─── Negation - simple ──────────────────────────────────────────────
// From: ascent_tests/src/agg_tests.rs (test_ascent_negation)
// Tests: negation as absence from another relation

#[test]
fn test_negation_simple() {
    let engine = run_with_facts(
        r#"
        relation foo(i32, i32);
        relation bar(i32, i32, i32);
        relation baz(i32, i32);
        baz(x, y) <-- foo(x, y), !bar(x, y, _);
    "#,
        vec![
            (
                "foo",
                vec![
                    vec![i(0), i(1)],
                    vec![i(1), i(2)],
                    vec![i(10), i(11)],
                    vec![i(100), i(101)],
                ],
            ),
            (
                "bar",
                vec![
                    vec![i(1), i(2), i(102)],
                    vec![i(10), i(11), i(20)],
                    vec![i(10), i(11), i(12)],
                ],
            ),
        ],
    );

    let baz = rel_set(&engine, "baz");
    assert_eq!(
        baz,
        [vec![i(0), i(1)], vec![i(100), i(101)]]
            .into_iter()
            .collect()
    );
}

// ─── Negation with arithmetic ───────────────────────────────────────
// From: ascent_tests/src/agg_tests.rs (test_ascent_negation3)
// Tests: computed expression in negated clause

#[test]
fn test_negation_with_arithmetic() {
    let engine = run_with_facts(
        r#"
        relation foo(i32, i32);
        relation bar(i32, i32, i32);
        relation baz(i32, i32);
        baz(x, y) <-- foo(x, y), !bar(x, y, y + 1);
    "#,
        vec![
            (
                "foo",
                vec![
                    vec![i(0), i(1)],
                    vec![i(1), i(2)],
                    vec![i(10), i(11)],
                    vec![i(100), i(101)],
                ],
            ),
            (
                "bar",
                vec![
                    vec![i(1), i(2), i(3)],    // matches y+1=3 for (1,2)
                    vec![i(10), i(11), i(13)], // doesn't match y+1=12
                ],
            ),
        ],
    );

    let baz = rel_set(&engine, "baz");
    // (0,1): no bar(0,1,2) → included
    // (1,2): bar(1,2,3) matches y+1=3 → excluded
    // (10,11): bar(10,11,13) but y+1=12≠13 → included
    // (100,101): no bar match → included
    assert_eq!(
        baz,
        [vec![i(0), i(1)], vec![i(10), i(11)], vec![i(100), i(101)]]
            .into_iter()
            .collect()
    );
}

// ─── Aggregation - all built-ins ────────────────────────────────────
// From: ascent/examples/ascent_agg_clause.rs
// Tests: min, max, sum, count over generated data

#[test]
fn test_aggregation_all() {
    let engine = run(r#"
        relation number(i32);
        relation lowest(i32);
        relation greatest(i32);
        relation total(i32);
        relation card(i32);

        number(x) <-- for x in 1..6;
        lowest(y) <-- agg y = min(x) in number(x);
        greatest(y) <-- agg y = max(x) in number(x);
        total(y) <-- agg y = sum(x) in number(x);
        card(y) <-- agg y = count() in number(_);
    "#);

    assert!(engine.relation("lowest").unwrap().contains(&[i(1)]));
    assert!(engine.relation("greatest").unwrap().contains(&[i(5)]));
    assert!(engine.relation("total").unwrap().contains(&[i(15)]));
    assert!(engine.relation("card").unwrap().contains(&[i(5)]));
}

// ─── Aggregation with grouping ──────────────────────────────────────
// Tests: aggregation that groups by a key from another relation

#[test]
fn test_aggregation_grouped() {
    let engine = run(r#"
        relation score(i32, i32);
        relation team_max(i32, i32);

        score(1, 10);
        score(1, 20);
        score(1, 15);
        score(2, 5);
        score(2, 25);

        team_max(team, m) <-- score(team, _), agg m = max(s) in score(team, s);
    "#);

    let team_max = rel_set(&engine, "team_max");
    assert_eq!(
        team_max,
        [vec![i(1), i(20)], vec![i(2), i(25)]].into_iter().collect()
    );
}

// ─── Graph: connected components ────────────────────────────────────
// Tests: bidirectional transitive closure

#[test]
fn test_connected_components() {
    let engine = run(r#"
        relation edge(i32, i32);
        relation reach(i32, i32);

        edge(1, 2);
        edge(2, 3);
        edge(4, 5);

        reach(x, y) <-- edge(x, y);
        reach(x, y) <-- edge(y, x);
        reach(x, z) <-- reach(x, y), reach(y, z);
    "#);

    let reach = engine.relation("reach").unwrap();
    // Component {1,2,3}: 3×3 = 9 pairs (including self-loops via reach(x,y),reach(y,x))
    // Component {4,5}: 2×2 = 4 pairs (including self-loops)
    assert!(reach.contains(&[i(1), i(3)]));
    assert!(reach.contains(&[i(3), i(1)]));
    assert!(reach.contains(&[i(1), i(1)])); // self-loop
    assert!(reach.contains(&[i(4), i(5)]));
    assert!(reach.contains(&[i(5), i(4)]));
    assert!(!reach.contains(&[i(1), i(4)])); // different component
    assert_eq!(reach.len(), 13); // 9 + 4
}

// ─── Generator with seed ────────────────────────────────────────────
// From: ascent/examples/ascent_for_in_clause.rs
// Tests: generator expression depends on bound variable

#[test]
fn test_generator_with_seed() {
    let engine = run(r#"
        relation seed(i32);
        relation number(i32);

        seed(0);
        seed(10);
        number(x + y) <-- seed(x), for y in 0..3;
    "#);

    let number = rel_set(&engine, "number");
    assert_eq!(
        number,
        [0, 1, 2, 10, 11, 12]
            .into_iter()
            .map(|v| vec![i(v)])
            .collect()
    );
}

// ─── Multiple conditions ────────────────────────────────────────────
// Tests: chaining several filter conditions

#[test]
fn test_multiple_conditions() {
    let engine = run(r#"
        relation number(i32);
        relation special(i32);

        number(x) <-- for x in 1..100;
        special(x) <-- number(x), if x % 3 == 0, if x % 5 == 0, if x < 50;
    "#);

    let special = rel_set(&engine, "special");
    assert_eq!(
        special,
        [15, 30, 45].into_iter().map(|v| vec![i(v)]).collect()
    );
}

// ─── Three-way join ─────────────────────────────────────────────────
// Tests: joining three relations

#[test]
fn test_three_way_join() {
    let engine = run(r#"
        relation a(i32, i32);
        relation b(i32, i32);
        relation c(i32, i32);
        relation result(i32, i32);

        a(1, 2);
        a(1, 3);
        b(2, 4);
        b(3, 5);
        c(4, 100);
        c(5, 200);

        result(x, w) <-- a(x, y), b(y, z), c(z, w);
    "#);

    let result = rel_set(&engine, "result");
    assert_eq!(
        result,
        [vec![i(1), i(100)], vec![i(1), i(200)]]
            .into_iter()
            .collect()
    );
}

// ─── Mutual recursion ───────────────────────────────────────────────
// Tests: two relations that depend on each other

#[test]
fn test_mutual_recursion() {
    let engine = run(r#"
        relation is_even(i32);
        relation is_odd(i32);

        is_even(0);
        is_odd(x + 1) <-- is_even(x), if x < 10;
        is_even(x + 1) <-- is_odd(x), if x < 10;
    "#);

    let is_even = rel_set(&engine, "is_even");
    let is_odd = rel_set(&engine, "is_odd");

    assert_eq!(
        is_even,
        [0, 2, 4, 6, 8, 10]
            .into_iter()
            .map(|v| vec![i(v)])
            .collect()
    );
    assert_eq!(
        is_odd,
        [1, 3, 5, 7, 9].into_iter().map(|v| vec![i(v)]).collect()
    );
}

// ─── Duplicate elimination ──────────────────────────────────────────
// Tests: set semantics — duplicates are automatically eliminated

#[test]
fn test_duplicate_elimination() {
    let engine = run(r#"
        relation source(i32);
        relation result(i32);

        source(1);
        source(2);
        source(3);

        result(1) <-- source(1);
        result(1) <-- source(2);
        result(x) <-- source(x);
    "#);

    let result = rel_set(&engine, "result");
    // Despite multiple rules producing result(1), only one copy exists
    assert_eq!(result, [1, 2, 3].into_iter().map(|v| vec![i(v)]).collect());
}

// ─── Empty relation handling ────────────────────────────────────────
// Tests: aggregation over empty relations, joins with empty relations

#[test]
fn test_empty_relations() {
    let engine = run(r#"
        relation empty(i32);
        relation nonempty(i32);
        relation joined(i32, i32);
        relation negated(i32);

        nonempty(1);
        nonempty(2);

        joined(x, y) <-- nonempty(x), empty(y);
        negated(x) <-- nonempty(x), !empty(x);
    "#);

    assert!(engine.relation("joined").unwrap().is_empty());

    let negated = rel_set(&engine, "negated");
    assert_eq!(negated, [1, 2].into_iter().map(|v| vec![i(v)]).collect());
}

// ─── Rule chaining ──────────────────────────────────────────────────
// Tests: output of one rule feeds into another non-recursively

#[test]
fn test_rule_chaining() {
    let engine = run(r#"
        relation input(i32);
        relation doubled(i32);
        relation filtered(i32);
        relation final_result(i32);

        input(x) <-- for x in 1..10;
        doubled(x * 2) <-- input(x);
        filtered(x) <-- doubled(x), if x > 10;
        final_result(x - 10) <-- filtered(x);
    "#);

    let result = rel_set(&engine, "final_result");
    // input: 1..10, doubled: 2,4,6,8,10,12,14,16,18
    // filtered: 12,14,16,18, final_result: 2,4,6,8
    assert_eq!(
        result,
        [2, 4, 6, 8].into_iter().map(|v| vec![i(v)]).collect()
    );
}

// ─── Aggregation after recursion ────────────────────────────────────
// Tests: stratified aggregation over recursively-computed relation

#[test]
fn test_aggregation_over_recursive() {
    let engine = run(r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        relation path_count(i32);

        edge(1, 2);
        edge(2, 3);
        edge(3, 4);
        edge(4, 5);

        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);

        path_count(n) <-- agg n = count() in path(_, _);
    "#);

    let path = engine.relation("path").unwrap();
    assert_eq!(path.len(), 10); // 4 + 3 + 2 + 1

    let pc = engine.relation("path_count").unwrap();
    assert!(pc.contains(&[i(10)]));
}

// ─── Constant in body clause ────────────────────────────────────────
// Tests: matching against a literal value in a clause

#[test]
fn test_constant_in_clause() {
    let engine = run(r#"
        relation data(i32, i32);
        relation ones(i32);

        data(1, 10);
        data(1, 20);
        data(2, 30);
        data(2, 40);

        ones(y) <-- data(1, y);
    "#);

    let ones = rel_set(&engine, "ones");
    assert_eq!(ones, [10, 20].into_iter().map(|v| vec![i(v)]).collect());
}

// ─── If let: Option pattern matching ────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_pattern_args)
// Tests: if let Some(y) = x pattern in conditions

#[test]
fn test_if_let_option() {
    let engine = run_with_facts(
        r#"
        relation input(i32, Option<i32>);
        relation output(i32, i32);
        output(x, y) <-- input(x, opt), if let Some(y) = opt;
    "#,
        vec![(
            "input",
            vec![
                vec![i(1), Value::Option(None)],
                vec![i(2), Value::Option(Some(Box::new(i(20))))],
                vec![i(3), Value::Option(Some(Box::new(i(30))))],
                vec![i(4), Value::Option(None)],
            ],
        )],
    );

    let output = rel_set(&engine, "output");
    assert_eq!(
        output,
        [vec![i(2), i(20)], vec![i(3), i(30)]].into_iter().collect()
    );
}

// ─── If let with filter ─────────────────────────────────────────────
// From: ascent_tests/src/tests.rs (test_dl_pattern_args)
// Tests: if let combined with inequality filter

#[test]
fn test_if_let_with_filter() {
    let engine = run_with_facts(
        r#"
        relation input(i32, Option<i32>);
        relation output(i32, i32);
        output(x, y) <-- input(x, opt), if let Some(y) = opt, if y != x;
    "#,
        vec![(
            "input",
            vec![
                vec![i(2), Value::Option(Some(Box::new(i(2))))],
                vec![i(3), Value::Option(Some(Box::new(i(30))))],
            ],
        )],
    );

    let output = rel_set(&engine, "output");
    // (2, 2) filtered out because y == x
    assert_eq!(output, [vec![i(3), i(30)]].into_iter().collect());
}

// ─── Let binding ────────────────────────────────────────────────────
// Tests: let binding introduces a new variable

#[test]
fn test_let_binding() {
    let engine = run(r#"
        relation input(i32);
        relation output(i32, i32);

        input(x) <-- for x in 1..6;
        output(x, doubled) <-- input(x), let doubled = x * 2;
    "#);

    let output = rel_set(&engine, "output");
    assert!(output.contains(&[i(1), i(2)]));
    assert!(output.contains(&[i(3), i(6)]));
    assert!(output.contains(&[i(5), i(10)]));
    assert_eq!(output.len(), 5);
}

// ─── If let with join ───────────────────────────────────────────────
// Tests: pattern-matched value used in subsequent join

#[test]
fn test_if_let_join() {
    let engine = run_with_facts(
        r#"
        relation source(i32, Option<i32>);
        relation lookup(i32, i32);
        relation result(i32, i32);
        result(x, z) <-- source(x, opt), if let Some(y) = opt, lookup(y, z);
    "#,
        vec![
            (
                "source",
                vec![
                    vec![i(1), Value::Option(Some(Box::new(i(10))))],
                    vec![i(2), Value::Option(None)],
                    vec![i(3), Value::Option(Some(Box::new(i(20))))],
                ],
            ),
            ("lookup", vec![vec![i(10), i(100)], vec![i(20), i(200)]]),
        ],
    );

    let result = rel_set(&engine, "result");
    assert_eq!(
        result,
        [vec![i(1), i(100)], vec![i(3), i(200)]]
            .into_iter()
            .collect()
    );
}
