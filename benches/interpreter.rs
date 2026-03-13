//! Performance benchmarks: interpreter vs ascent macro (vs JIT when available).
//!
//! Run with: cargo bench
//! Run with JIT: cargo bench --features jit

#![allow(clippy::field_reassign_with_default)]

use ascent::aggregators::min;
use ascent::ascent;
use ascent_eval::Engine;
use ascent_eval::value::Value;
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

/// Pre-parse a program, returning (program, source_with_facts) for benchmarks
/// that want to separate parse from run.
fn prepare_program(input: &str) -> (Program, Engine) {
    let ast: AscentProgram = syn::parse_str(input).unwrap();
    let program = Program::from_ast(ast);
    let engine = Engine::new(&program);
    (program, engine)
}

// ─── Transitive Closure ─────────────────────────────────────────────

fn tc_source(n: i32) -> String {
    let mut source = String::from("relation edge(i32, i32);\nrelation path(i32, i32);\n");
    for i in 1..n {
        source.push_str(&format!("edge({}, {});\n", i, i + 1));
    }
    source.push_str("path(x, y) <-- edge(x, y);\n");
    source.push_str("path(x, z) <-- edge(x, y), path(y, z);\n");
    source
}

fn tc_source_no_facts() -> String {
    String::from(
        "relation edge(i32, i32);\n\
         relation path(i32, i32);\n\
         path(x, y) <-- edge(x, y);\n\
         path(x, z) <-- edge(x, y), path(y, z);\n",
    )
}

fn bench_transitive_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("transitive_closure");

    for &n in &[50, 100, 200] {
        // Interpreter (includes parse)
        group.bench_with_input(BenchmarkId::new("interpreter", n), &n, |b, &n| {
            b.iter(|| run_interpreter(&tc_source(n)));
        });

        // Interpreter runtime only (pre-parsed, facts via insert)
        group.bench_with_input(BenchmarkId::new("interp_run_only", n), &n, |b, &n| {
            let source = tc_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                for i in 1..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
        });

        // JIT runtime only (pre-parsed, facts via insert).
        // NOTE: includes JIT compilation cost on every iteration — JIT cache
        // is per-Engine and each iter creates a fresh one. Numbers reflect
        // "first-run" latency.
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_run_only", n), &n, |b, &n| {
            let source = tc_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.enable_jit();
                for i in 1..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
        });

        // JIT hot-cache: pre-compile once, share compiled JIT across iterations.
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_hot", n), &n, |b, &n| {
            let source = tc_source_no_facts();
            let (program, _) = prepare_program(&source);
            // Warmup: compile JIT once
            let mut warmup = Engine::new(&program);
            warmup.enable_jit();
            for i in 1..n {
                warmup.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
            }
            warmup.run(&program);
            let compiled = warmup.share_jit_compiler().unwrap();
            // Hot iterations: reuse compiled JIT
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.with_jit_compiler(compiled.clone());
                for i in 1..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
        });

        // Native ascent macro
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

fn triangle_source(n: i32) -> String {
    let mut source = String::from("relation edge(i32, i32);\nrelation triangle(i32, i32, i32);\n");
    for i in 1..=n {
        for j in (i + 1)..=n {
            source.push_str(&format!("edge({i}, {j});\n"));
        }
    }
    source.push_str(
        "triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;\n",
    );
    source
}

fn triangle_source_no_facts() -> String {
    String::from(
        "relation edge(i32, i32);\n\
         relation triangle(i32, i32, i32);\n\
         triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;\n",
    )
}

fn bench_triangle(c: &mut Criterion) {
    let mut group = c.benchmark_group("triangle_detection");

    for &n in &[10, 20, 30] {
        group.bench_with_input(BenchmarkId::new("interpreter", n), &n, |b, &n| {
            b.iter(|| run_interpreter(&triangle_source(n)));
        });

        group.bench_with_input(BenchmarkId::new("interp_run_only", n), &n, |b, &n| {
            let source = triangle_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                for i in 1..=n {
                    for j in (i + 1)..=n {
                        engine.insert("edge", vec![Value::I32(i), Value::I32(j)]);
                    }
                }
                engine.run(&program);
                engine
            });
        });

        // NOTE: includes JIT compilation cost on every iteration (see TC comment).
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_run_only", n), &n, |b, &n| {
            let source = triangle_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.enable_jit();
                for i in 1..=n {
                    for j in (i + 1)..=n {
                        engine.insert("edge", vec![Value::I32(i), Value::I32(j)]);
                    }
                }
                engine.run(&program);
                engine
            });
        });

        // JIT hot-cache: pre-compile once, share compiled JIT across iterations.
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_hot", n), &n, |b, &n| {
            let source = triangle_source_no_facts();
            let (program, _) = prepare_program(&source);
            // Warmup: compile JIT once
            let mut warmup = Engine::new(&program);
            warmup.enable_jit();
            for i in 1..=n {
                for j in (i + 1)..=n {
                    warmup.insert("edge", vec![Value::I32(i), Value::I32(j)]);
                }
            }
            warmup.run(&program);
            let compiled = warmup.share_jit_compiler().unwrap();
            // Hot iterations: reuse compiled JIT
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.with_jit_compiler(compiled.clone());
                for i in 1..=n {
                    for j in (i + 1)..=n {
                        engine.insert("edge", vec![Value::I32(i), Value::I32(j)]);
                    }
                }
                engine.run(&program);
                engine
            });
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

fn connected_components_source_no_facts() -> String {
    String::from(
        "relation edge(i32, i32);\n\
         relation reach(i32, i32);\n\
         relation comp(i32, i32);\n\
         reach(x, y) <-- edge(x, y);\n\
         reach(x, y) <-- edge(y, x);\n\
         reach(x, z) <-- reach(x, y), reach(y, z);\n\
         comp(x, m) <-- reach(x, _), agg m = min(y) in reach(x, y);\n",
    )
}

fn connected_components_interpreter(n: i32) -> Engine {
    let mut source = String::from(
        "relation edge(i32, i32);\nrelation reach(i32, i32);\nrelation comp(i32, i32);\n",
    );
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

        group.bench_with_input(BenchmarkId::new("interp_run_only", n), &n, |b, &n| {
            let source = connected_components_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let half = n / 2;
                let mut engine = Engine::new(&program);
                for i in 1..half {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                for i in (half + 1)..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
        });

        // NOTE: includes JIT compilation cost on every iteration (see TC comment).
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_run_only", n), &n, |b, &n| {
            let source = connected_components_source_no_facts();
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let half = n / 2;
                let mut engine = Engine::new(&program);
                engine.enable_jit();
                for i in 1..half {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                for i in (half + 1)..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
        });

        // JIT hot-cache: pre-compile once, share compiled JIT across iterations.
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_hot", n), &n, |b, &n| {
            let source = connected_components_source_no_facts();
            let (program, _) = prepare_program(&source);
            // Warmup: compile JIT once
            let mut warmup = Engine::new(&program);
            warmup.enable_jit();
            let half = n / 2;
            for i in 1..half {
                warmup.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
            }
            for i in (half + 1)..n {
                warmup.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
            }
            warmup.run(&program);
            let compiled = warmup.share_jit_compiler().unwrap();
            // Hot iterations: reuse compiled JIT
            b.iter(|| {
                let half = n / 2;
                let mut engine = Engine::new(&program);
                engine.with_jit_compiler(compiled.clone());
                for i in 1..half {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                for i in (half + 1)..n {
                    engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
                }
                engine.run(&program);
                engine
            });
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

        group.bench_with_input(BenchmarkId::new("interp_run_only", limit), &limit, |b, &n| {
            let source = format!(
                "relation fib(i32, i32);\n\
                 fib(0, 0);\n\
                 fib(1, 1);\n\
                 fib(nn + 1, a + b) <-- fib(nn, a), fib(nn - 1, b), if *nn < {n};\n"
            );
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.run(&program);
                engine
            });
        });

        // NOTE: includes JIT compilation cost on every iteration (see TC comment).
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_run_only", limit), &limit, |b, &n| {
            let source = format!(
                "relation fib(i32, i32);\n\
                 fib(0, 0);\n\
                 fib(1, 1);\n\
                 fib(nn + 1, a + b) <-- fib(nn, a), fib(nn - 1, b), if *nn < {n};\n"
            );
            let (program, _) = prepare_program(&source);
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.enable_jit();
                engine.run(&program);
                engine
            });
        });

        // JIT hot-cache: pre-compile once, share compiled JIT across iterations.
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_hot", limit), &limit, |b, &n| {
            let source = format!(
                "relation fib(i32, i32);\n\
                 fib(0, 0);\n\
                 fib(1, 1);\n\
                 fib(nn + 1, a + b) <-- fib(nn, a), fib(nn - 1, b), if *nn < {n};\n"
            );
            let (program, _) = prepare_program(&source);
            // Warmup: compile JIT once (no facts to insert — seeds are in program text)
            let mut warmup = Engine::new(&program);
            warmup.enable_jit();
            warmup.run(&program);
            let compiled = warmup.share_jit_compiler().unwrap();
            // Hot iterations: reuse compiled JIT
            b.iter(|| {
                let mut engine = Engine::new(&program);
                engine.with_jit_compiler(compiled.clone());
                engine.run(&program);
                engine
            });
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
