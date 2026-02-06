//! Comparison tests: run the same program through ascent macro and interpreter,
//! verify both produce identical results.

use std::collections::BTreeSet;

use ascent::Dual;
use ascent::aggregators::{count, max, min, sum};
use ascent::ascent;
use ascent_eval::Engine;
use ascent_eval::value::Value;
use ascent_ir::Program;
use ascent_syntax::AscentProgram as AscentAst;

fn run(input: &str) -> Engine {
    let ast: AscentAst = syn::parse_str(input).unwrap();
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    engine.run(&program);
    engine
}

/// Extract a 1-column i32 relation from the interpreter.
fn interp1(engine: &Engine, name: &str) -> BTreeSet<(i32,)> {
    engine
        .relation(name)
        .unwrap()
        .iter()
        .map(|t| match t.as_slice() {
            [Value::I32(a)] => (*a,),
            other => panic!("expected 1-col i32, got {other:?}"),
        })
        .collect()
}

/// Extract a 2-column i32 relation from the interpreter.
fn interp2(engine: &Engine, name: &str) -> BTreeSet<(i32, i32)> {
    engine
        .relation(name)
        .unwrap()
        .iter()
        .map(|t| match t.as_slice() {
            [Value::I32(a), Value::I32(b)] => (*a, *b),
            other => panic!("expected 2-col i32, got {other:?}"),
        })
        .collect()
}

/// Extract a 3-column i32 relation from the interpreter.
fn interp3(engine: &Engine, name: &str) -> BTreeSet<(i32, i32, i32)> {
    engine
        .relation(name)
        .unwrap()
        .iter()
        .map(|t| match t.as_slice() {
            [Value::I32(a), Value::I32(b), Value::I32(c)] => (*a, *b, *c),
            other => panic!("expected 3-col i32, got {other:?}"),
        })
        .collect()
}

fn set1(v: Vec<(i32,)>) -> BTreeSet<(i32,)> {
    v.into_iter().collect()
}

fn set2(v: Vec<(i32, i32)>) -> BTreeSet<(i32, i32)> {
    v.into_iter().collect()
}

// ─── Transitive Closure ─────────────────────────────────────────────

#[test]
fn compare_transitive_closure() {
    let engine = run("
        relation edge(i32, i32);
        relation path(i32, i32);
        edge(1, 2); edge(2, 3); edge(3, 4);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    ");

    ascent! {
        relation edge(i32, i32);
        relation path(i32, i32);
        edge(1, 2); edge(2, 3); edge(3, 4);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "path"), set2(prog.path));
}

// ─── FizzBuzz ───────────────────────────────────────────────────────

#[test]
fn compare_fizzbuzz() {
    let engine = run("
        relation number(i32);
        relation divisible(i32, i32);
        relation fizz(i32);
        relation buzz(i32);
        relation fizz_buzz(i32);
        relation other(i32);

        number(x) <-- for x in 1..16;
        divisible(x, y) <-- number(x), for y in [3, 5], if x % y == 0;
        fizz(x) <-- divisible(x, 3), !divisible(x, 5);
        buzz(x) <-- divisible(x, 5), !divisible(x, 3);
        fizz_buzz(x) <-- divisible(x, 3), divisible(x, 5);
        other(x) <-- number(x), !divisible(x, 3), !divisible(x, 5);
    ");

    ascent! {
        relation number(i32);
        relation divisible(i32, i32);
        relation fizz(i32);
        relation buzz(i32);
        relation fizz_buzz(i32);
        relation other(i32);

        number(x) <-- for x in 1..16;
        divisible(x, y) <-- number(x), for y in [3, 5], if x % y == 0;
        fizz(x) <-- divisible(x, 3), !divisible(x, 5);
        buzz(x) <-- divisible(x, 5), !divisible(x, 3);
        fizz_buzz(x) <-- divisible(x, 3), divisible(x, 5);
        other(x) <-- number(x), !divisible(x, 3), !divisible(x, 5);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "fizz"), set1(prog.fizz), "fizz");
    assert_eq!(interp1(&engine, "buzz"), set1(prog.buzz), "buzz");
    assert_eq!(
        interp1(&engine, "fizz_buzz"),
        set1(prog.fizz_buzz),
        "fizz_buzz"
    );
    assert_eq!(interp1(&engine, "other"), set1(prog.other), "other");
}

// ─── Factorial ──────────────────────────────────────────────────────

#[test]
fn compare_factorial() {
    let engine = run("
        relation fac(i32, i32);
        fac(0, 1);
        fac(n + 1, (n + 1) * f) <-- fac(n, f), if *n < 5;
    ");

    ascent! {
        relation fac(i32, i32);
        fac(0, 1);
        fac(n + 1, (n + 1) * f) <-- fac(n, f), if *n < 5;
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "fac"), set2(prog.fac));
}

// ─── Generators ─────────────────────────────────────────────────────

#[test]
fn compare_generators() {
    let engine = run("
        relation nums(i32);
        relation pairs(i32, i32);
        nums(x) <-- for x in 0..5;
        pairs(x, y) <-- nums(x), nums(y), if x < y;
    ");

    ascent! {
        relation nums(i32);
        relation pairs(i32, i32);
        nums(x) <-- for x in 0..5;
        pairs(x, y) <-- nums(x), nums(y), if x < y;
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "nums"), set1(prog.nums));
    assert_eq!(interp2(&engine, "pairs"), set2(prog.pairs));
}

// ─── Three-Way Join ─────────────────────────────────────────────────

#[test]
fn compare_three_way_join() {
    let engine = run("
        relation a(i32, i32);
        relation b(i32, i32);
        relation c(i32, i32);
        relation result(i32, i32, i32, i32);

        a(1, 2); a(2, 3);
        b(2, 10); b(3, 20);
        c(10, 100); c(20, 200);

        result(x, y, z, w) <-- a(x, y), b(y, z), c(z, w);
    ");

    ascent! {
        relation a(i32, i32);
        relation b(i32, i32);
        relation c(i32, i32);
        relation result(i32, i32, i32, i32);

        a(1, 2); a(2, 3);
        b(2, 10); b(3, 20);
        c(10, 100); c(20, 200);

        result(x, y, z, w) <-- a(x, y), b(y, z), c(z, w);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    let interp: BTreeSet<(i32, i32, i32, i32)> = engine
        .relation("result")
        .unwrap()
        .iter()
        .map(|t| match t.as_slice() {
            [Value::I32(a), Value::I32(b), Value::I32(c), Value::I32(d)] => (*a, *b, *c, *d),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    let macro_result: BTreeSet<(i32, i32, i32, i32)> = prog.result.into_iter().collect();

    assert_eq!(interp, macro_result);
}

// ─── Mutual Recursion (even/odd) ────────────────────────────────────

#[test]
fn compare_mutual_recursion() {
    // Use head arithmetic (y+1) instead of body arithmetic (x-1) since
    // ascent macro requires body clause args to be bound variables.
    let engine = run("
        relation even(i32);
        relation odd(i32);

        even(0);
        odd(y + 1) <-- even(y), if *y < 9;
        even(y + 1) <-- odd(y), if *y < 9;
    ");

    ascent! {
        relation even(i32);
        relation odd(i32);

        even(0);
        odd(y + 1) <-- even(y), if *y < 9;
        even(y + 1) <-- odd(y), if *y < 9;
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "even"), set1(prog.even), "even");
    assert_eq!(interp1(&engine, "odd"), set1(prog.odd), "odd");
}

// ─── Self-Join ──────────────────────────────────────────────────────

#[test]
fn compare_self_join() {
    let engine = run("
        relation edge(i32, i32);
        relation triangle(i32, i32, i32);

        edge(1, 2); edge(2, 3); edge(3, 1);
        edge(4, 5); edge(5, 6);

        triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(c, a);
    ");

    ascent! {
        relation edge(i32, i32);
        relation triangle(i32, i32, i32);

        edge(1, 2); edge(2, 3); edge(3, 1);
        edge(4, 5); edge(5, 6);

        triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(c, a);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(
        interp3(&engine, "triangle"),
        prog.triangle.into_iter().collect()
    );
}

// ─── Negation ───────────────────────────────────────────────────────

#[test]
fn compare_negation() {
    let engine = run("
        relation a(i32);
        relation b(i32);
        relation only_a(i32);

        a(1); a(2); a(3); a(4);
        b(2); b(4);
        only_a(x) <-- a(x), !b(x);
    ");

    ascent! {
        relation a(i32);
        relation b(i32);
        relation only_a(i32);

        a(1); a(2); a(3); a(4);
        b(2); b(4);
        only_a(x) <-- a(x), !b(x);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "only_a"), set1(prog.only_a));
}

// ─── Arithmetic Conditions ──────────────────────────────────────────

#[test]
fn compare_arithmetic() {
    let engine = run("
        relation n(i32);
        relation square(i32, i32);
        relation big_square(i32, i32);

        n(x) <-- for x in 1..11;
        square(x, x * x) <-- n(x);
        big_square(x, s) <-- square(x, s), if *s > 50;
    ");

    ascent! {
        relation n(i32);
        relation square(i32, i32);
        relation big_square(i32, i32);

        n(x) <-- for x in 1..11;
        square(x, x * x) <-- n(x);
        big_square(x, s) <-- square(x, s), if *s > 50;
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "square"), set2(prog.square));
    assert_eq!(interp2(&engine, "big_square"), set2(prog.big_square));
}

// ─── Duplicate Elimination ──────────────────────────────────────────

#[test]
fn compare_duplicate_elimination() {
    let engine = run("
        relation input(i32, i32);
        relation unique_first(i32);

        input(1, 10); input(1, 20); input(2, 30); input(2, 40); input(3, 50);
        unique_first(x) <-- input(x, _);
    ");

    ascent! {
        relation input(i32, i32);
        relation unique_first(i32);

        input(1, 10); input(1, 20); input(2, 30); input(2, 40); input(3, 50);
        unique_first(x) <-- input(x, _);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "unique_first"), set1(prog.unique_first));
}

// ─── Disjunction ────────────────────────────────────────────────────

#[test]
fn compare_disjunction() {
    let engine = run("
        relation a(i32);
        relation b(i32);
        relation c(i32);

        a(1); a(2); a(3);
        b(3); b(4); b(5);
        c(x) <-- (a(x) || b(x));
    ");

    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);

        a(1); a(2); a(3);
        b(3); b(4); b(5);
        c(x) <-- (a(x) || b(x));
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "c"), set1(prog.c));
}

// ─── Connected Components ───────────────────────────────────────────

#[test]
fn compare_connected_components() {
    let engine = run("
        relation edge(i32, i32);
        relation reach(i32, i32);
        relation comp(i32, i32);

        edge(1, 2); edge(2, 3);
        edge(4, 5);

        reach(x, y) <-- edge(x, y);
        reach(x, y) <-- edge(y, x);
        reach(x, z) <-- reach(x, y), reach(y, z);

        comp(x, min) <-- reach(x, _), agg min = min(y) in reach(x, y);
    ");

    ascent! {
        relation edge(i32, i32);
        relation reach(i32, i32);
        relation comp(i32, i32);

        edge(1, 2); edge(2, 3);
        edge(4, 5);

        reach(x, y) <-- edge(x, y);
        reach(x, y) <-- edge(y, x);
        reach(x, z) <-- reach(x, y), reach(y, z);

        comp(x, min) <-- reach(x, _), agg min = min(y) in reach(x, y);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "reach"), set2(prog.reach), "reach");
    assert_eq!(interp2(&engine, "comp"), set2(prog.comp), "comp");
}

// ─── Aggregation: count and sum ─────────────────────────────────────

#[test]
fn compare_aggregation() {
    let engine = run("
        relation score(i32, i32);
        relation total(i32);
        relation cnt(i32);

        score(1, 10); score(2, 20); score(3, 30);
        total(s) <-- agg s = sum(x) in score(_, x);
        cnt(c) <-- agg c = count() in score(_, _);
    ");

    // ascent count() returns usize, so use separate relation types
    ascent! {
        relation score(i32, i32);
        relation total(i32);
        relation cnt(usize);

        score(1, 10); score(2, 20); score(3, 30);
        total(s) <-- agg s = sum(x) in score(_, x);
        cnt(c) <-- agg c = count() in score(_, _);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "total"), set1(prog.total), "total");
    // count: compare as i32 (interpreter) vs usize (ascent)
    let interp_cnt = interp1(&engine, "cnt");
    let macro_cnt: BTreeSet<(i32,)> = prog.cnt.into_iter().map(|(c,)| (c as i32,)).collect();
    assert_eq!(interp_cnt, macro_cnt, "cnt");
}

// ─── Grouped Aggregation ────────────────────────────────────────────

#[test]
fn compare_grouped_aggregation() {
    let engine = run("
        relation score(i32, i32);
        relation best(i32, i32);

        score(1, 10); score(1, 20); score(1, 5);
        score(2, 30); score(2, 15);
        score(3, 25);

        best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
    ");

    ascent! {
        relation score(i32, i32);
        relation best(i32, i32);

        score(1, 10); score(1, 20); score(1, 5);
        score(2, 30); score(2, 15);
        score(3, 25);

        best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "best"), set2(prog.best));
}

// ─── Recursive with Aggregation ─────────────────────────────────────

#[test]
fn compare_recursive_with_aggregation() {
    let engine = run("
        relation edge(i32, i32);
        relation path(i32, i32);
        relation path_count(i32);

        edge(1, 2); edge(2, 3); edge(3, 4); edge(1, 3);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);

        path_count(c) <-- agg c = count() in path(_, _);
    ");

    // ascent count() returns usize
    ascent! {
        relation edge(i32, i32);
        relation path(i32, i32);
        relation path_count(usize);

        edge(1, 2); edge(2, 3); edge(3, 4); edge(1, 3);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);

        path_count(c) <-- agg c = count() in path(_, _);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "path"), set2(prog.path), "path");
    let interp_cnt = interp1(&engine, "path_count");
    let macro_cnt: BTreeSet<(i32,)> = prog
        .path_count
        .into_iter()
        .map(|(c,)| (c as i32,))
        .collect();
    assert_eq!(interp_cnt, macro_cnt, "path_count");
}

// ─── Constant in Clause ─────────────────────────────────────────────

#[test]
fn compare_constant_in_clause() {
    let engine = run("
        relation data(i32, i32);
        relation filtered(i32);

        data(1, 100); data(2, 200); data(3, 100); data(4, 300);
        filtered(x) <-- data(x, 100);
    ");

    ascent! {
        relation data(i32, i32);
        relation filtered(i32);

        data(1, 100); data(2, 200); data(3, 100); data(4, 300);
        filtered(x) <-- data(x, 100);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "filtered"), set1(prog.filtered));
}

// ─── Rule Chaining ──────────────────────────────────────────────────

#[test]
fn compare_rule_chaining() {
    let engine = run("
        relation base(i32);
        relation step1(i32);
        relation step2(i32);
        relation step3(i32);

        base(x) <-- for x in 1..6;
        step1(x * 2) <-- base(x);
        step2(x + 1) <-- step1(x);
        step3(x) <-- step2(x), if *x > 5;
    ");

    ascent! {
        relation base(i32);
        relation step1(i32);
        relation step2(i32);
        relation step3(i32);

        base(x) <-- for x in 1..6;
        step1(x * 2) <-- base(x);
        step2(x + 1) <-- step1(x);
        step3(x) <-- step2(x), if *x > 5;
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp1(&engine, "step1"), set1(prog.step1), "step1");
    assert_eq!(interp1(&engine, "step2"), set1(prog.step2), "step2");
    assert_eq!(interp1(&engine, "step3"), set1(prog.step3), "step3");
}

// ─── Cascading Aggregation ──────────────────────────────────────────

#[test]
fn compare_cascading_aggregation() {
    let engine = run("
        relation score(i32, i32);
        relation best(i32, i32);
        relation overall_best(i32);

        score(1, 10); score(1, 20);
        score(2, 30); score(2, 15);

        best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
        overall_best(m) <-- agg m = max(s) in best(_, s);
    ");

    ascent! {
        relation score(i32, i32);
        relation best(i32, i32);
        relation overall_best(i32);

        score(1, 10); score(1, 20);
        score(2, 30); score(2, 15);

        best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
        overall_best(m) <-- agg m = max(s) in best(_, s);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "best"), set2(prog.best), "best");
    assert_eq!(
        interp1(&engine, "overall_best"),
        set1(prog.overall_best),
        "overall_best"
    );
}

// ─── Lattice: Max Value ────────────────────────────────────────────

#[test]
fn compare_lattice_max() {
    let engine = run("
        lattice best(i32, i32);
        best(1, 10);
        best(1, 20);
        best(1, 5);
        best(2, 30);
        best(2, 15);
    ");

    ascent! {
        lattice best(i32, i32);
        best(1, 10);
        best(1, 20);
        best(1, 5);
        best(2, 30);
        best(2, 15);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(interp2(&engine, "best"), set2(prog.best));
}

// ─── Lattice: Shortest Path with Dual ──────────────────────────────

#[test]
fn compare_lattice_shortest_path() {
    let engine = run("
        relation edge(i32, i32, i32);
        lattice shortest(i32, i32, Dual<i32>);

        edge(1, 2, 1);
        edge(2, 3, 2);
        edge(1, 3, 10);

        shortest(x, y, Dual(*w)) <-- edge(x, y, w);
        shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
    ");

    ascent! {
        relation edge(i32, i32, i32);
        lattice shortest(i32, i32, Dual<i32>);

        edge(1, 2, 1);
        edge(2, 3, 2);
        edge(1, 3, 10);

        shortest(x, y, Dual(*w)) <-- edge(x, y, w);
        shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
    }
    let mut prog = AscentProgram::default();
    prog.run();

    // Extract as (src, dst, inner_value) for comparison
    let interp_sp: BTreeSet<(i32, i32, i32)> = engine
        .relation("shortest")
        .unwrap()
        .iter()
        .map(|t| match t.as_slice() {
            [Value::I32(a), Value::I32(b), Value::Dual(d)] => match d.as_ref() {
                Value::I32(v) => (*a, *b, *v),
                other => panic!("expected Dual(i32), got Dual({other:?})"),
            },
            other => panic!("expected (i32, i32, Dual<i32>), got {other:?}"),
        })
        .collect();

    let macro_sp: BTreeSet<(i32, i32, i32)> = prog
        .shortest
        .into_iter()
        .map(|(a, b, d)| (a, b, *d))
        .collect();

    assert_eq!(interp_sp, macro_sp);
}

// ─── Lattice: Recursive Max Propagation ────────────────────────────

#[test]
fn compare_lattice_recursive_max() {
    let engine = run("
        relation edge(i32, i32);
        relation source_val(i32, i32);
        lattice max_reach(i32, i32);

        edge(1, 2); edge(2, 3);
        source_val(1, 100);
        source_val(2, 50);

        max_reach(x, v) <-- source_val(x, v);
        max_reach(y, v) <-- edge(x, y), max_reach(x, v);
    ");

    ascent! {
        relation edge(i32, i32);
        relation source_val(i32, i32);
        lattice max_reach(i32, i32);

        edge(1, 2); edge(2, 3);
        source_val(1, 100);
        source_val(2, 50);

        max_reach(x, v) <-- source_val(x, v);
        max_reach(y, v) <-- edge(x, y), max_reach(x, v);
    }
    let mut prog = AscentProgram::default();
    prog.run();

    assert_eq!(
        interp2(&engine, "max_reach"),
        set2(prog.max_reach),
        "max_reach"
    );
}
