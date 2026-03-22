//! Tests for the JIT compiler.

use crate::Engine;
use crate::value::Value;
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

/// Run a program with JIT and without, assert same results.
fn assert_jit_equivalence(input: &str, facts: &[(&str, Vec<Vec<Value>>)], query_rel: &str) {
    let ast: AscentProgram = syn::parse_str(input).expect("parse");
    let program = Program::from_ast(ast).expect("lowering should succeed");

    // Run without JIT
    let mut engine_interp = Engine::new(program.clone());
    for (rel, tuples) in facts {
        for tuple in tuples {
            engine_interp.insert(rel, tuple.clone());
        }
    }
    engine_interp.run();
    let mut interp_results: Vec<Vec<Value>> = engine_interp
        .relation(query_rel)
        .unwrap()
        .iter()
        .map(|t| t.to_vec())
        .collect();
    interp_results.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    // Run with JIT
    let mut engine_jit = Engine::new(program);
    engine_jit.enable_jit().expect("JIT init should succeed");
    for (rel, tuples) in facts {
        for tuple in tuples {
            engine_jit.insert(rel, tuple.clone());
        }
    }
    engine_jit.run();
    engine_jit.materialize();
    let mut jit_results: Vec<Vec<Value>> = engine_jit
        .relation(query_rel)
        .unwrap()
        .iter()
        .map(|t| t.to_vec())
        .collect();
    jit_results.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    assert_eq!(
        interp_results, jit_results,
        "JIT results differ from interpreted for relation '{query_rel}'"
    );
}

/// Run a program with JIT+specialized enabled, assert same results as interpreter.
#[cfg(feature = "specialized")]
fn assert_packed_jit_equivalence(input: &str, facts: &[(&str, Vec<Vec<Value>>)], query_rel: &str) {
    assert_jit_equivalence(input, facts, query_rel);
}

#[test]
fn test_jit_compiler_creation() {
    let compiler = super::JitCompiler::new();
    assert!(
        compiler.is_ok(),
        "JIT compiler creation failed: {:?}",
        compiler.err()
    );
}

#[test]
fn test_jit_single_clause_copy() {
    assert_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[test]
fn test_jit_transitive_closure() {
    assert_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[test]
fn test_jit_with_condition() {
    assert_jit_equivalence(
        r#"
            relation r(i32, i32);
            relation s(i32, i32);
            s(x, y) <-- r(x, y), if x > 0;
        "#,
        &[(
            "r",
            vec![
                vec![Value::I32(-1), Value::I32(10)],
                vec![Value::I32(1), Value::I32(20)],
                vec![Value::I32(2), Value::I32(30)],
            ],
        )],
        "s",
    );
}

#[test]
fn test_jit_multi_clause_join() {
    assert_jit_equivalence(
        r#"
            relation a(i32, i32);
            relation b(i32, i32);
            relation c(i32, i32);
            c(x, z) <-- a(x, y), b(y, z);
        "#,
        &[
            (
                "a",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(2), Value::I32(3)],
                ],
            ),
            (
                "b",
                vec![
                    vec![Value::I32(2), Value::I32(20)],
                    vec![Value::I32(3), Value::I32(30)],
                ],
            ),
        ],
        "c",
    );
}

#[test]
fn test_jit_self_join() {
    assert_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation triangle(i32, i32, i32);
            triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(c, a);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(1)],
            ],
        )],
        "triangle",
    );
}

// ─── Packed JIT tests ───────────────────────────────────────────────

/// These test the typed packed JIT path, which requires the `specialized` feature.
/// The packed JIT reads u32 directly from PackedStorage, bypassing Value enum.
#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_single_clause_copy() {
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_transitive_closure() {
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_triangle_detection() {
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation triangle(i32, i32, i32);
            triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(c, a);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(1)],
                vec![Value::I32(4), Value::I32(5)], // not part of a triangle
            ],
        )],
        "triangle",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_string_relations() {
    let hello = Value::string("hello");
    let world = Value::string("world");
    let foo = Value::string("foo");
    let bar = Value::string("bar");
    assert_packed_jit_equivalence(
        r#"
            relation src(String, String);
            relation dst(String, String);
            dst(x, y) <-- src(x, y);
        "#,
        &[(
            "src",
            vec![
                vec![hello.clone(), world.clone()],
                vec![foo.clone(), bar.clone()],
            ],
        )],
        "dst",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_two_clause_join() {
    assert_packed_jit_equivalence(
        r#"
            relation a(i32, i32);
            relation b(i32, i32);
            relation c(i32, i32);
            c(x, z) <-- a(x, y), b(y, z);
        "#,
        &[
            (
                "a",
                vec![
                    vec![Value::I32(1), Value::I32(10)],
                    vec![Value::I32(2), Value::I32(20)],
                ],
            ),
            (
                "b",
                vec![
                    vec![Value::I32(10), Value::I32(100)],
                    vec![Value::I32(20), Value::I32(200)],
                ],
            ),
        ],
        "c",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_multi_bound_columns() {
    // Tests the secondary-column icmp check path
    assert_packed_jit_equivalence(
        r#"
            relation r(i32, i32, i32);
            relation s(i32, i32, i32);
            s(x, y, z) <-- r(x, y, z), r(y, x, z);
        "#,
        &[(
            "r",
            vec![
                vec![Value::I32(1), Value::I32(2), Value::I32(99)],
                vec![Value::I32(2), Value::I32(1), Value::I32(99)], // forms a pair
                vec![Value::I32(3), Value::I32(4), Value::I32(77)], // no match
            ],
        )],
        "s",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_condition_comparison() {
    // Rule with if x > y condition on packed i32 columns
    assert_packed_jit_equivalence(
        r#"
            relation r(i32, i32);
            relation s(i32, i32);
            s(x, y) <-- r(x, y), if x > y;
        "#,
        &[(
            "r",
            vec![
                vec![Value::I32(5), Value::I32(3)],  // 5 > 3: included
                vec![Value::I32(1), Value::I32(2)],  // 1 > 2: excluded
                vec![Value::I32(4), Value::I32(4)],  // 4 > 4: excluded
            ],
        )],
        "s",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_literal_clause_arg() {
    // Rule with literal constant in clause arg: edge(0, y)
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation reachable(i32);
            reachable(y) <-- edge(0, y);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(0), Value::I32(10)],  // matches literal 0
                vec![Value::I32(1), Value::I32(20)],  // doesn't match
                vec![Value::I32(0), Value::I32(30)],  // matches literal 0
            ],
        )],
        "reachable",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_arithmetic_condition() {
    // Rule with arithmetic condition: if x + 1 == y
    assert_packed_jit_equivalence(
        r#"
            relation r(i32, i32);
            relation s(i32, i32);
            s(x, y) <-- r(x, y), if x + 1 == y;
        "#,
        &[(
            "r",
            vec![
                vec![Value::I32(1), Value::I32(2)],  // 1+1==2: included
                vec![Value::I32(3), Value::I32(4)],  // 3+1==4: included
                vec![Value::I32(5), Value::I32(7)],  // 5+1!=7: excluded
            ],
        )],
        "s",
    );
}

// ─── Stratum meta-function tests ────────────────────────────────────

/// These tests exercise the stratum-level JIT path, which compiles
/// a whole fixpoint loop per stratum into a single asm function.
#[cfg(feature = "specialized")]
#[test]
fn test_stratum_meta_tc() {
    // Classic transitive closure: two rules in the same SCC, exercises the
    // full+recent loop in the meta-function.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- path(x, y), edge(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
                vec![Value::I32(4), Value::I32(5)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stratum_meta_multi_rule_stratum() {
    // Multiple rules writing to the same output relation in the same SCC.
    assert_packed_jit_equivalence(
        r#"
            relation a(i32, i32);
            relation b(i32, i32);
            relation reach(i32, i32);
            reach(x, y) <-- a(x, y);
            reach(x, y) <-- b(x, y);
            reach(x, z) <-- reach(x, y), a(y, z);
            reach(x, z) <-- reach(x, y), b(y, z);
        "#,
        &[
            (
                "a",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(3), Value::I32(4)],
                ],
            ),
            (
                "b",
                vec![
                    vec![Value::I32(2), Value::I32(3)],
                    vec![Value::I32(4), Value::I32(5)],
                ],
            ),
        ],
        "reach",
    );
}

// ─── Stage 4 direct-insert tests ────────────────────────────────────

/// These tests specifically exercise the Stage 4 (direct-insert) path and verify
/// correctness across recursive, multi-rule, and conditional scenarios.
#[cfg(feature = "specialized")]
#[test]
fn test_stage3_recursive_triangle() {
    // Recursive rule where head relation appears in body — exercises the
    // re-fetch-packed_data_ptr fix for reallocation safety.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- path(x, y), path(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(1)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage3_multi_hop_tc() {
    // 5-node chain — verifies correct multi-iteration convergence with direct inserts.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- path(x, y), edge(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
                vec![Value::I32(4), Value::I32(5)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage3_condition_in_recursive_rule() {
    // Recursive rule with condition — verifies conditions work under direct insert.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y), if x < y;
            path(x, z) <-- path(x, y), edge(y, z), if x < z;
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(2)], // back edge — filtered by condition
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stratum_meta_single_rule_fixpoint() {
    // Single rule with self-join — exercises fixpoint convergence.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(10), Value::I32(20)],
                vec![Value::I32(20), Value::I32(30)],
                vec![Value::I32(30), Value::I32(10)], // cycle
            ],
        )],
        "path",
    );
}

// ─── Stage 4 inlined-body tests ─────────────────────────────────────

/// These tests specifically exercise the Stage 4 (inlined rule bodies) path,
/// which compiles all rule bodies directly into a single asm function,
/// eliminating per-rule call overhead.
#[cfg(feature = "specialized")]
#[test]
fn test_stage4_transitive_closure() {
    // Classic TC — same as test_stratum_meta_tc; Stage 4 must produce identical results.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- path(x, y), edge(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
                vec![Value::I32(4), Value::I32(5)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage4_tc_edge_first() {
    // TC with edge first, path second — exercises linked-list index scan (is_recursive=true).
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage4_triangle() {
    // Triangle detection: 3-clause body rule.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation tri(i32, i32, i32);
            tri(x, y, z) <-- edge(x, y), edge(y, z), edge(z, x);
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(1)],
                vec![Value::I32(4), Value::I32(5)], // not in a triangle
            ],
        )],
        "tri",
    );
}

/// Test triangle detection with enough edges to trigger the adaptive col-scan path
/// (JitRelData.len > 2048 triggers JitColIndex scan instead of JitTupleSet probe).
#[cfg(feature = "specialized")]
#[test]
fn test_stage4_triangle_large() {
    // n=65 gives 65*64/2 = 2080 edges, exceeding the 2048 threshold.
    let n = 65i32;
    let edges: Vec<Vec<Value>> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| vec![Value::I32(i), Value::I32(j)]))
        .collect();
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation tri(i32, i32, i32);
            tri(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c);
        "#,
        &[("edge", edges)],
        "tri",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage4_multi_rule() {
    // Multiple rules writing to the same relation — exercises inlining of N rules.
    assert_packed_jit_equivalence(
        r#"
            relation a(i32, i32);
            relation b(i32, i32);
            relation reach(i32, i32);
            reach(x, y) <-- a(x, y);
            reach(x, y) <-- b(x, y);
            reach(x, z) <-- reach(x, y), a(y, z);
            reach(x, z) <-- reach(x, y), b(y, z);
        "#,
        &[
            (
                "a",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(3), Value::I32(4)],
                ],
            ),
            (
                "b",
                vec![
                    vec![Value::I32(2), Value::I32(3)],
                    vec![Value::I32(4), Value::I32(5)],
                ],
            ),
        ],
        "reach",
    );
}

/// Test the AVX2 SIMD filter path for an `is_col_value` scan with a `<` condition.
///
/// `reach(x, z) <-- edge(x, y), edge(y, z) if z < x` — after optimize_body
/// `if z < x` is pushed to clause 1's conditions (inner col-value scan where z is fresh).
/// detect_simd_filter should find it and emit the SIMD prefix.
/// Correctness is verified by comparing against the interpreter.
#[cfg(feature = "specialized")]
#[test]
fn test_stage4_simd_filter_lt() {
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation reach(i32, i32);
            reach(x, z) <-- edge(x, y), edge(y, z) if z < x;
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(1)],
                vec![Value::I32(2), Value::I32(1)],
                vec![Value::I32(1), Value::I32(3)],
                vec![Value::I32(3), Value::I32(2)],
                // Add more to potentially exceed 8 elements in one range
                vec![Value::I32(4), Value::I32(5)],
                vec![Value::I32(4), Value::I32(6)],
                vec![Value::I32(4), Value::I32(7)],
                vec![Value::I32(4), Value::I32(8)],
                vec![Value::I32(4), Value::I32(9)],
            ],
        )],
        "reach",
    );
}

/// Test that the JitTupleSet fast path for fully-bound arity-3 clauses works correctly.
///
/// The rule `result(x) <-- triple(x, y, z), triple(y, z, x)` has:
///   clause 0: `triple(x, y, z)` — full scan, binds x, y, z
///   clause 1: `triple(y, z, x)` — all args bound (fresh_cols empty), arity=3
/// The last clause triggers the JitTupleSet probe path.
#[cfg(feature = "specialized")]
#[test]
fn test_stage4_tuple_set_probe_arity3() {
    assert_packed_jit_equivalence(
        r#"
            relation triple(i32, i32, i32);
            relation result(i32);
            result(x) <-- triple(x, y, z), triple(y, z, x);
        "#,
        &[(
            "triple",
            vec![
                vec![Value::I32(1), Value::I32(2), Value::I32(3)],
                vec![Value::I32(2), Value::I32(3), Value::I32(1)], // forms cycle with first
                vec![Value::I32(4), Value::I32(5), Value::I32(6)], // no cycle partner
            ],
        )],
        "result",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage4_conditional_recursive() {
    // Recursive rule with an `if` condition — verifies conditions are handled inline.
    assert_packed_jit_equivalence(
        r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y), if x < y;
            path(x, z) <-- path(x, y), edge(y, z), if x < z;
        "#,
        &[(
            "edge",
            vec![
                vec![Value::I32(1), Value::I32(2)],
                vec![Value::I32(2), Value::I32(3)],
                vec![Value::I32(3), Value::I32(2)], // back edge — filtered by condition
                vec![Value::I32(3), Value::I32(4)],
            ],
        )],
        "path",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_stage4_fibonacci() {
    // Fibonacci — exercises col-value linked-list for arity-2 recursive inner clauses
    // (two inner fib clauses, both recursive, both arity-2).
    assert_packed_jit_equivalence(
        r#"
            relation fib(i32, i32);
            fib(0, 0);
            fib(1, 1);
            fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if n < 8;
        "#,
        &[],
        "fib",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_div_rem_bitops() {
    // Exercises Div, Rem, BitAnd, BitOr in asm backend conditions and head expressions.
    assert_packed_jit_equivalence(
        r#"
            relation r(i32);
            relation div2(i32, i32);   // (x, x/2)
            relation mod3(i32, i32);   // (x, x%3)
            relation band(i32, i32);   // (x, x & 6)
            relation bor(i32, i32);    // (x, x | 1)
            div2(x, x / 2) <-- r(x);
            mod3(x, x % 3) <-- r(x);
            band(x, x & 6) <-- r(x);
            bor(x,  x | 1) <-- r(x);
        "#,
        &[(
            "r",
            vec![
                vec![Value::I32(0)],
                vec![Value::I32(1)],
                vec![Value::I32(6)],
                vec![Value::I32(7)],
                vec![Value::I32(10)],
            ],
        )],
        "div2",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_negation() {
    // Exercises negation (anti-join) in the asm backend.
    // `filtered(x)` = base(x) that is not in `excluded`.
    assert_packed_jit_equivalence(
        r#"
            relation base(i32);
            relation excluded(i32);
            relation filtered(i32);
            filtered(x) <-- base(x), !excluded(x);
        "#,
        &[
            (
                "base",
                vec![
                    vec![Value::I32(1)],
                    vec![Value::I32(2)],
                    vec![Value::I32(3)],
                    vec![Value::I32(4)],
                    vec![Value::I32(5)],
                ],
            ),
            (
                "excluded",
                vec![vec![Value::I32(2)], vec![Value::I32(4)]],
            ),
        ],
        "filtered",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_negation_2tuple() {
    // Exercises negation with a 2-tuple negated relation.
    // `noedge(x, y)` = pair(x, y) where edge(x, y) does NOT exist.
    assert_packed_jit_equivalence(
        r#"
            relation pair(i32, i32);
            relation edge(i32, i32);
            relation noedge(i32, i32);
            noedge(x, y) <-- pair(x, y), !edge(x, y);
        "#,
        &[
            (
                "pair",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(1), Value::I32(3)],
                    vec![Value::I32(2), Value::I32(3)],
                    vec![Value::I32(3), Value::I32(4)],
                ],
            ),
            (
                "edge",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(3), Value::I32(4)],
                ],
            ),
        ],
        "noedge",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_agg_count() {
    // Pure count() aggregation: count all tuples in `data`.
    assert_packed_jit_equivalence(
        r#"
            relation data(i32);
            relation total(i32);
            total(n) <-- agg n = count() in data(_x);
        "#,
        &[(
            "data",
            vec![
                vec![Value::I32(10)],
                vec![Value::I32(20)],
                vec![Value::I32(30)],
            ],
        )],
        "total",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_agg_sum() {
    // Pure sum() aggregation: sum all values in `data`.
    assert_packed_jit_equivalence(
        r#"
            relation data(i32);
            relation total(i32);
            total(s) <-- agg s = sum(x) in data(x);
        "#,
        &[(
            "data",
            vec![
                vec![Value::I32(10)],
                vec![Value::I32(20)],
                vec![Value::I32(30)],
            ],
        )],
        "total",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_agg_max() {
    // Pure max() aggregation: max value in `data`.
    assert_packed_jit_equivalence(
        r#"
            relation data(i32);
            relation result(i32);
            result(m) <-- agg m = max(x) in data(x);
        "#,
        &[(
            "data",
            vec![
                vec![Value::I32(5)],
                vec![Value::I32(1)],
                vec![Value::I32(9)],
                vec![Value::I32(3)],
            ],
        )],
        "result",
    );
}

#[cfg(feature = "specialized")]
#[test]
fn test_packed_jit_agg_min() {
    // Pure min() aggregation: min value in `data`.
    assert_packed_jit_equivalence(
        r#"
            relation data(i32);
            relation result(i32);
            result(m) <-- agg m = min(x) in data(x);
        "#,
        &[(
            "data",
            vec![
                vec![Value::I32(5)],
                vec![Value::I32(1)],
                vec![Value::I32(9)],
                vec![Value::I32(3)],
            ],
        )],
        "result",
    );
}
