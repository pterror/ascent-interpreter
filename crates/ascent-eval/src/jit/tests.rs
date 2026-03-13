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
