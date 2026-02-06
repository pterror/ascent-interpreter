//! Performance benchmarks: interpreter vs ascent macro.
//!
//! Run with: cargo bench

#![allow(clippy::field_reassign_with_default)]

use ascent::aggregators::min;
use ascent::ascent;
use ascent_eval::Engine;
use ascent_ir::Program;
use ascent_syntax::AscentProgram;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn run_interpreter(input: &str) -> Engine {
    let ast: AscentProgram = syn::parse_str(input).unwrap();
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    engine.run(&program);
    engine
}

// ─── Transitive Closure ─────────────────────────────────────────────

fn transitive_closure_interpreter(n: i32) -> Engine {
    let mut source = String::from("relation edge(i32, i32);\nrelation path(i32, i32);\n");
    // Linear chain: 1→2→3→...→n
    for i in 1..n {
        source.push_str(&format!("edge({}, {});\n", i, i + 1));
    }
    source.push_str("path(x, y) <-- edge(x, y);\n");
    source.push_str("path(x, z) <-- edge(x, y), path(y, z);\n");
    run_interpreter(&source)
}

fn bench_transitive_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("transitive_closure");

    for &n in &[50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("interpreter", n), &n, |b, &n| {
            b.iter(|| transitive_closure_interpreter(n));
        });

        group.bench_with_input(BenchmarkId::new("ascent_macro", n), &n, |b, &n| {
            b.iter(|| {
                ascent! {
                    relation edge(i32, i32);
                    relation path(i32, i32);
                    path(x, y) <-- edge(x, y);
                    path(x, z) <-- edge(x, y), path(y, z);
                }
                let mut prog = AscentProgram::default();
                prog.edge = (1..n).map(|i| (i, i + 1)).collect();
                prog.run();
                prog
            });
        });
    }
    group.finish();
}

// ─── Triangle Detection ─────────────────────────────────────────────

fn triangle_interpreter(n: i32) -> Engine {
    let mut source = String::from("relation edge(i32, i32);\nrelation triangle(i32, i32, i32);\n");
    // Complete graph K_n
    for i in 1..=n {
        for j in (i + 1)..=n {
            source.push_str(&format!("edge({i}, {j});\n"));
        }
    }
    source.push_str(
        "triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;\n",
    );
    run_interpreter(&source)
}

fn bench_triangle(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangle_detection");

    for &n in &[10, 20, 30] {
        group.bench_with_input(BenchmarkId::new("interpreter", n), &n, |b, &n| {
            b.iter(|| triangle_interpreter(n));
        });

        group.bench_with_input(BenchmarkId::new("ascent_macro", n), &n, |b, &n| {
            b.iter(|| {
                ascent! {
                    relation edge(i32, i32);
                    relation triangle(i32, i32, i32);
                    triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c),
                        if a < b, if b < c;
                }
                let mut prog = AscentProgram::default();
                prog.edge = (1..=n)
                    .flat_map(|i| ((i + 1)..=n).map(move |j| (i, j)))
                    .collect();
                prog.run();
                prog
            });
        });
    }
    group.finish();
}

// ─── Connected Components ───────────────────────────────────────────

fn connected_components_interpreter(n: i32) -> Engine {
    let mut source = String::from(
        "relation edge(i32, i32);\nrelation reach(i32, i32);\nrelation comp(i32, i32);\n",
    );
    // Two disconnected linear chains: 1→2→...→n/2 and n/2+1→...→n
    let half = n / 2;
    for i in 1..half {
        source.push_str(&format!("edge({}, {});\n", i, i + 1));
    }
    for i in (half + 1)..n {
        source.push_str(&format!("edge({}, {});\n", i, i + 1));
    }
    source.push_str("reach(x, y) <-- edge(x, y);\n");
    source.push_str("reach(x, y) <-- edge(y, x);\n");
    source.push_str("reach(x, z) <-- reach(x, y), reach(y, z);\n");
    source.push_str("comp(x, m) <-- reach(x, _), agg m = min(y) in reach(x, y);\n");
    run_interpreter(&source)
}

fn bench_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");

    for &n in &[20, 40, 60] {
        group.bench_with_input(BenchmarkId::new("interpreter", n), &n, |b, &n| {
            b.iter(|| connected_components_interpreter(n));
        });

        group.bench_with_input(BenchmarkId::new("ascent_macro", n), &n, |b, &n| {
            b.iter(|| {
                ascent! {
                    relation edge(i32, i32);
                    relation reach(i32, i32);
                    relation comp(i32, i32);
                    reach(x, y) <-- edge(x, y);
                    reach(x, y) <-- edge(y, x);
                    reach(x, z) <-- reach(x, y), reach(y, z);
                    comp(x, m) <-- reach(x, _), agg m = min(y) in reach(x, y);
                }
                let mut prog = AscentProgram::default();
                let half = n / 2;
                let mut edges: Vec<(i32, i32)> = (1..half).map(|i| (i, i + 1)).collect();
                edges.extend((half + 1..n).map(|i| (i, i + 1)));
                prog.edge = edges;
                prog.run();
                prog
            });
        });
    }
    group.finish();
}

// ─── Fibonacci ──────────────────────────────────────────────────────

fn fibonacci_interpreter(limit: i32) -> Engine {
    run_interpreter(&format!(
        "
        relation fib(i32, i32);
        fib(0, 0);
        fib(1, 1);
        fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if *n < {limit};
    "
    ))
}

fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    for &limit in &[10, 15, 20] {
        group.bench_with_input(BenchmarkId::new("interpreter", limit), &limit, |b, &n| {
            b.iter(|| fibonacci_interpreter(n));
        });

        group.bench_with_input(BenchmarkId::new("ascent_macro", limit), &limit, |b, &n| {
            b.iter(|| {
                ascent! {
                    relation fib(i32, i32);
                    relation limit(i32);
                    fib(0, 0);
                    fib(1, 1);
                    fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), limit(lim), if *n < *lim;
                }
                let mut prog = AscentProgram::default();
                prog.limit = vec![(n,)];
                prog.run();
                prog
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transitive_closure,
    bench_triangle,
    bench_connected_components,
    bench_fibonacci
);
criterion_main!(benches);
