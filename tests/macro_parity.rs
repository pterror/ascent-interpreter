//! Compare JIT output against ascent_macro output (LLVM-compiled ground truth).
//! Catches bugs where both JIT and interpreter agree but are both wrong.

#![cfg(feature = "jit")]

use ascent_eval::value::Value;
use ascent_eval::Engine;
use ascent_ir::Program;
use ascent_syntax::AscentProgram as AstProgram;

fn jit_run(src: &str, facts: &[(&str, Vec<Vec<Value>>)], rel: &str) -> Vec<Vec<Value>> {
    let ast: AstProgram = syn::parse_str(src).expect("parse");
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    engine.enable_jit();
    for (name, tuples) in facts {
        for t in tuples {
            engine.insert(name, t.clone());
        }
    }
    engine.run(&program);
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
