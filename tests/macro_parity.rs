//! Compare JIT output against ascent_macro output (LLVM-compiled ground truth).
//! Catches bugs where both JIT and interpreter agree but are both wrong.

#![cfg(feature = "jit")]

use ascent_interpreter::eval::value::Value;
use ascent_interpreter::eval::Engine;
use ascent_interpreter::ir::Program;
use ascent_interpreter::syntax::AscentProgram as AstProgram;

fn jit_run(src: &str, facts: &[(&str, Vec<Vec<Value>>)], rel: &str) -> Vec<Vec<Value>> {
    let ast: AstProgram = syn::parse_str(src).expect("parse");
    let program = Program::from_ast(ast).expect("lowering should succeed");
    let mut engine = Engine::new(program);
    engine.enable_jit().expect("JIT init should succeed");
    for (name, tuples) in facts {
        for t in tuples {
            engine.insert(name, t.clone()).unwrap();
        }
    }
    engine.run().unwrap();
    engine.materialize();
    engine
        .relation(rel)
        .unwrap()
        .iter()
        .map(|t| t.to_vec())
        .collect()
}

fn extract_i32_pair(t: &[Value]) -> (i32, i32) {
    match (&t[0], &t[1]) {
        (Value::I32(a), Value::I32(b)) => (*a, *b),
        _ => panic!("expected i32 pair"),
    }
}

fn extract_i32_single(t: &[Value]) -> (i32,) {
    match &t[0] {
        Value::I32(a) => (*a,),
        _ => panic!("expected i32 single"),
    }
}

fn extract_i32_triple(t: &[Value]) -> (i32, i32, i32) {
    match (&t[0], &t[1], &t[2]) {
        (Value::I32(a), Value::I32(b), Value::I32(c)) => (*a, *b, *c),
        _ => panic!("expected i32 triple"),
    }
}

// ─── TC chain (30 edges, 465 paths) ─────────────────────────────────────

mod macro_tc {
    pub fn run(n: i32) -> Vec<(i32, i32)> {
        ascent::ascent! {
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        }
        let mut prog = AscentProgram::default();
        for i in 1..=n {
            prog.edge.push((i, i + 1));
        }
        prog.run();
        prog.path
    }
}

#[test]
fn test_jit_tc_chain_30_standalone() {
    let src = "relation edge(i32, i32);\n\
               relation path(i32, i32);\n\
               path(x, y) <-- edge(x, y);\n\
               path(x, z) <-- edge(x, y), path(y, z);";
    let results = jit_run(src, &[("edge", (1..=30).map(|i| vec![Value::I32(i), Value::I32(i + 1)]).collect())], "path");
    assert_eq!(results.len(), 465, "TC chain-30 should produce 465 pairs");
}

#[test]
fn test_macro_parity_tc_chain_30() {
    let mut macro_results = macro_tc::run(30);
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation path(i32, i32);\n\
               path(x, y) <-- edge(x, y);\n\
               path(x, z) <-- edge(x, y), path(y, z);";
    let edges: Vec<Vec<Value>> = (1..=30)
        .map(|i| vec![Value::I32(i), Value::I32(i + 1)])
        .collect();
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[("edge", edges)], "path")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    assert_eq!(jit_results.len(), 465, "TC chain-30: expected 465 pairs");
    assert_eq!(
        jit_results, macro_results,
        "JIT TC differs from ascent_macro"
    );
}

// ─── Triangle K_15 (455 triangles) ──────────────────────────────────────

mod macro_triangle {
    pub fn run(n: i32) -> Vec<(i32, i32, i32)> {
        ascent::ascent! {
            relation edge(i32, i32);
            relation triangle(i32, i32, i32);
            triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c),
                if a < b, if b < c;
        }
        let mut prog = AscentProgram::default();
        for i in 0..n {
            for j in (i + 1)..n {
                prog.edge.push((i, j));
            }
        }
        prog.run();
        prog.triangle
    }
}

#[test]
fn test_macro_parity_triangle_k15() {
    let n = 15i32;
    let mut macro_results = macro_triangle::run(n);
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation triangle(i32, i32, i32);\n\
               triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;";
    let edges: Vec<Vec<Value>> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| vec![Value::I32(i), Value::I32(j)]))
        .collect();
    let mut jit_results: Vec<(i32, i32, i32)> = jit_run(src, &[("edge", edges)], "triangle")
        .iter()
        .map(|t| extract_i32_triple(t))
        .collect();
    jit_results.sort();

    assert_eq!(jit_results.len(), 455, "K_15: expected C(15,3) = 455");
    assert_eq!(
        jit_results, macro_results,
        "JIT triangle differs from ascent_macro"
    );
}

// ─── Triangle K_30 (larger, stress test) ────────────────────────────────

#[test]
fn test_macro_parity_triangle_k30() {
    let n = 30i32;
    let mut macro_results = macro_triangle::run(n);
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation triangle(i32, i32, i32);\n\
               triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;";
    let edges: Vec<Vec<Value>> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| vec![Value::I32(i), Value::I32(j)]))
        .collect();
    let mut jit_results: Vec<(i32, i32, i32)> = jit_run(src, &[("edge", edges)], "triangle")
        .iter()
        .map(|t| extract_i32_triple(t))
        .collect();
    jit_results.sort();

    let expected = (n * (n - 1) * (n - 2) / 6) as usize;
    assert_eq!(jit_results.len(), expected, "K_{n}: expected C({n},3) = {expected}");
    assert_eq!(
        jit_results, macro_results,
        "JIT triangle K_{n} differs from ascent_macro"
    );
}

// ─── Fibonacci (recursive IDB body clause) ──────────────────────────────

mod macro_fib {
    pub fn run() -> Vec<(i32, i32)> {
        ascent::ascent! {
            relation fib(i32, i32);
            fib(0, 0);
            fib(1, 1);
            fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if *n < 20;
        }
        let mut prog = AscentProgram::default();
        prog.run();
        prog.fib
    }
}

#[test]
fn test_macro_parity_fibonacci() {
    let mut macro_results = macro_fib::run();
    macro_results.sort();

    let src = "relation fib(i32, i32);\n\
               fib(0, 0);\n\
               fib(1, 1);\n\
               fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if n < 20;";
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[], "fib")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    assert_eq!(
        jit_results.len(),
        macro_results.len(),
        "Fibonacci: count mismatch (jit={}, macro={})",
        jit_results.len(),
        macro_results.len()
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT fibonacci differs from ascent_macro"
    );
    // Verify fib(20, 6765) is present
    assert!(
        jit_results.contains(&(20, 6765)),
        "fib(20, 6765) missing from JIT results"
    );
}

// ─── Connected Components with min aggregation ──────────────────────────

mod macro_components {
    use ascent::aggregators::min;
    pub fn run() -> Vec<(i32, i32)> {
        ascent::ascent! {
            relation edge(i32, i32);
            relation reach(i32, i32);
            relation component(i32, i32);
            reach(x, y) <-- edge(x, y);
            reach(x, y) <-- edge(y, x);
            reach(x, z) <-- reach(x, y), reach(y, z);
            component(x, m) <-- reach(x, _), agg m = min(y) in reach(x, y);
        }
        let mut prog = AscentProgram::default();
        // Component 1: {1, 2, 3}
        prog.edge.push((1, 2));
        prog.edge.push((2, 3));
        // Component 2: {10, 11}
        prog.edge.push((10, 11));
        prog.run();
        prog.component
    }
}

#[test]
fn test_macro_parity_connected_components() {
    let mut macro_results = macro_components::run();
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation reach(i32, i32);\n\
               relation component(i32, i32);\n\
               reach(x, y) <-- edge(x, y);\n\
               reach(x, y) <-- edge(y, x);\n\
               reach(x, z) <-- reach(x, y), reach(y, z);\n\
               component(x, m) <-- reach(x, _), agg m = min(y) in reach(x, y);";
    let edges = vec![
        vec![Value::I32(1), Value::I32(2)],
        vec![Value::I32(2), Value::I32(3)],
        vec![Value::I32(10), Value::I32(11)],
    ];
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[("edge", edges)], "component")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    assert_eq!(
        jit_results.len(),
        macro_results.len(),
        "Components: count mismatch (jit={}, macro={})",
        jit_results.len(),
        macro_results.len()
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT connected components differs from ascent_macro"
    );
}

// ─── Negation (anti-join) ───────────────────────────────────────────────

mod macro_negation {
    pub fn run() -> Vec<(i32,)> {
        ascent::ascent! {
            relation person(i32);
            relation has_parent(i32);
            relation orphan(i32);
            orphan(x) <-- person(x), !has_parent(x);
        }
        let mut prog = AscentProgram::default();
        for p in [1, 2, 3, 4, 5] {
            prog.person.push((p,));
        }
        for hp in [2, 3, 5] {
            prog.has_parent.push((hp,));
        }
        prog.run();
        prog.orphan
    }
}

#[test]
fn test_macro_parity_negation() {
    let mut macro_results = macro_negation::run();
    macro_results.sort();

    let src = "relation person(i32);\n\
               relation has_parent(i32);\n\
               relation orphan(i32);\n\
               orphan(x) <-- person(x), !has_parent(x);";
    let persons: Vec<Vec<Value>> = [1, 2, 3, 4, 5]
        .iter()
        .map(|&v| vec![Value::I32(v)])
        .collect();
    let parents: Vec<Vec<Value>> = [2, 3, 5].iter().map(|&v| vec![Value::I32(v)]).collect();
    let mut jit_results: Vec<(i32,)> =
        jit_run(src, &[("person", persons), ("has_parent", parents)], "orphan")
            .iter()
            .map(|t| extract_i32_single(t))
            .collect();
    jit_results.sort();

    assert_eq!(jit_results.len(), 2, "Negation: expected 2 orphans");
    assert_eq!(
        jit_results, macro_results,
        "JIT negation differs from ascent_macro"
    );
}

// ─── Multiple strata (aggregation depends on base stratum) ──────────────

mod macro_winner {
    use ascent::aggregators::max;
    pub fn run() -> Vec<(i32,)> {
        ascent::ascent! {
            relation score(i32, i32);
            relation max_score(i32, i32);
            relation winner(i32);
            max_score(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
            winner(player) <-- max_score(player, s), if *s >= 100;
        }
        let mut prog = AscentProgram::default();
        prog.score.push((1, 50));
        prog.score.push((1, 120));
        prog.score.push((2, 30));
        prog.score.push((2, 80));
        prog.score.push((3, 100));
        prog.score.push((3, 99));
        prog.run();
        prog.winner
    }
}

#[test]
fn test_macro_parity_winner() {
    let mut macro_results = macro_winner::run();
    macro_results.sort();

    let src = "relation score(i32, i32);\n\
               relation max_score(i32, i32);\n\
               relation winner(i32);\n\
               max_score(player, m) <-- score(player, _), agg m = max(s) in score(player, s);\n\
               winner(player) <-- max_score(player, s), if s >= 100;";
    let scores = vec![
        vec![Value::I32(1), Value::I32(50)],
        vec![Value::I32(1), Value::I32(120)],
        vec![Value::I32(2), Value::I32(30)],
        vec![Value::I32(2), Value::I32(80)],
        vec![Value::I32(3), Value::I32(100)],
        vec![Value::I32(3), Value::I32(99)],
    ];
    let mut jit_results: Vec<(i32,)> = jit_run(src, &[("score", scores)], "winner")
        .iter()
        .map(|t| extract_i32_single(t))
        .collect();
    jit_results.sort();

    assert_eq!(
        jit_results.len(),
        macro_results.len(),
        "Winner: count mismatch (jit={}, macro={})",
        jit_results.len(),
        macro_results.len()
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT winner differs from ascent_macro"
    );
}

// ─── Self-join (same relation in body twice) ────────────────────────────

mod macro_grandparent {
    pub fn run() -> Vec<(i32, i32)> {
        ascent::ascent! {
            relation parent(i32, i32);
            relation grandparent(i32, i32);
            grandparent(x, z) <-- parent(x, y), parent(y, z);
        }
        let mut prog = AscentProgram::default();
        prog.parent.push((1, 2));
        prog.parent.push((2, 3));
        prog.parent.push((3, 4));
        prog.parent.push((2, 4));
        prog.run();
        prog.grandparent
    }
}

#[test]
fn test_macro_parity_grandparent() {
    let mut macro_results = macro_grandparent::run();
    macro_results.sort();

    let src = "relation parent(i32, i32);\n\
               relation grandparent(i32, i32);\n\
               grandparent(x, z) <-- parent(x, y), parent(y, z);";
    let parents = vec![
        vec![Value::I32(1), Value::I32(2)],
        vec![Value::I32(2), Value::I32(3)],
        vec![Value::I32(3), Value::I32(4)],
        vec![Value::I32(2), Value::I32(4)],
    ];
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[("parent", parents)], "grandparent")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    assert_eq!(
        jit_results.len(),
        macro_results.len(),
        "Grandparent: count mismatch (jit={}, macro={})",
        jit_results.len(),
        macro_results.len()
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT grandparent differs from ascent_macro"
    );
}

// ─── Large TC (100-edge chain, stress test) ─────────────────────────────

#[test]
fn test_macro_parity_tc_chain_100() {
    let n = 100;
    let mut macro_results = macro_tc::run(n);
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation path(i32, i32);\n\
               path(x, y) <-- edge(x, y);\n\
               path(x, z) <-- edge(x, y), path(y, z);";
    let edges: Vec<Vec<Value>> = (1..=n)
        .map(|i| vec![Value::I32(i), Value::I32(i + 1)])
        .collect();
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[("edge", edges)], "path")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    let expected = (n * (n + 1) / 2) as usize;
    assert_eq!(
        jit_results.len(),
        expected,
        "TC chain-100: expected {} pairs",
        expected
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT TC chain-100 differs from ascent_macro"
    );
}

// ─── Diamond pattern (multiple derivation paths, tests dedup) ───────────

mod macro_diamond {
    pub fn run() -> Vec<(i32, i32)> {
        ascent::ascent! {
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        }
        let mut prog = AscentProgram::default();
        prog.edge.push((1, 2));
        prog.edge.push((1, 3));
        prog.edge.push((2, 4));
        prog.edge.push((3, 4));
        prog.run();
        prog.path
    }
}

#[test]
fn test_macro_parity_diamond() {
    let mut macro_results = macro_diamond::run();
    macro_results.sort();

    let src = "relation edge(i32, i32);\n\
               relation path(i32, i32);\n\
               path(x, y) <-- edge(x, y);\n\
               path(x, z) <-- edge(x, y), path(y, z);";
    let edges = vec![
        vec![Value::I32(1), Value::I32(2)],
        vec![Value::I32(1), Value::I32(3)],
        vec![Value::I32(2), Value::I32(4)],
        vec![Value::I32(3), Value::I32(4)],
    ];
    let mut jit_results: Vec<(i32, i32)> = jit_run(src, &[("edge", edges)], "path")
        .iter()
        .map(|t| extract_i32_pair(t))
        .collect();
    jit_results.sort();

    // Edges: (1,2), (1,3), (2,4), (3,4)
    // Paths: (1,2), (1,3), (2,4), (3,4), (1,4) — path (1,4) derivable two ways but appears once
    assert_eq!(
        jit_results.len(),
        macro_results.len(),
        "Diamond: count mismatch (jit={}, macro={})",
        jit_results.len(),
        macro_results.len()
    );
    assert_eq!(
        jit_results, macro_results,
        "JIT diamond path differs from ascent_macro"
    );
    // Verify (1,4) appears exactly once
    assert_eq!(
        jit_results.iter().filter(|&&(a, b)| a == 1 && b == 4).count(),
        1,
        "path(1,4) should appear exactly once (dedup)"
    );
}
