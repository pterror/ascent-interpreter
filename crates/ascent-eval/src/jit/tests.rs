//! Tests for the JIT compiler.

use crate::Engine;
use crate::value::Value;
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

/// Run a program with JIT and without, assert same results.
fn assert_jit_equivalence(input: &str, facts: &[(&str, Vec<Vec<Value>>)], query_rel: &str) {
    let ast: AscentProgram = syn::parse_str(input).expect("parse");
    let program = Program::from_ast(ast);

    // Run without JIT
    let mut engine_interp = Engine::new(&program);
    for (rel, tuples) in facts {
        for tuple in tuples {
            engine_interp.insert(rel, tuple.clone());
        }
    }
    engine_interp.run(&program);
    let mut interp_results: Vec<Vec<Value>> = engine_interp
        .relation(query_rel)
        .unwrap()
        .iter()
        .map(|t| t.to_vec())
        .collect();
    interp_results.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    // Run with JIT
    let mut engine_jit = Engine::new(&program);
    engine_jit.enable_jit();
    for (rel, tuples) in facts {
        for tuple in tuples {
            engine_jit.insert(rel, tuple.clone());
        }
    }
    engine_jit.run(&program);
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

/// These tests exercise the stratum meta-function JIT path, which compiles
/// a whole fixpoint loop per stratum into a single Cranelift function.

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

#[cfg(feature = "specialized")]
// ─── Stage 3 direct-insert tests ────────────────────────────────────

/// These tests specifically exercise the Stage 3 (direct-insert) path and verify
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
