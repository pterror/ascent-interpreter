//! Tests for persisting engine state across runs.

use ascent_eval::Engine;
use ascent_eval::value::Value;
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

fn parse(input: &str) -> Program {
    let ast: AscentProgram = syn::parse_str(input).unwrap();
    Program::from_ast(ast)
}

fn collect_rel(engine: &Engine, name: &str) -> Vec<Vec<Value>> {
    let rel = engine.relation(name).expect("relation not found");
    let mut tuples: Vec<Vec<Value>> = rel.iter().map(|t| t.to_vec()).collect();
    tuples.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    tuples
}

#[test]
fn rerun_idempotent() {
    let src = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        edge(1, 2);
        edge(2, 3);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);
    engine.run(&program);
    let first = collect_rel(&engine, "path");

    // Run again on same engine — should produce identical results.
    engine.run(&program);
    let second = collect_rel(&engine, "path");
    assert_eq!(first, second);
}

#[test]
fn additive_facts() {
    let src = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);
    engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
    engine.run(&program);
    assert_eq!(collect_rel(&engine, "path").len(), 1); // (1,2)

    // Add a new fact and re-run.
    engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
    engine.run(&program);
    assert_eq!(collect_rel(&engine, "path").len(), 3); // (1,2), (2,3), (1,3)
}

#[test]
fn additive_rules() {
    // Start with only edge facts and a base rule.
    let src1 = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        edge(1, 2);
        edge(2, 3);
        path(x, y) <-- edge(x, y);
    "#;
    let prog1 = parse(src1);
    let mut engine = Engine::new(&prog1);
    engine.run(&prog1);
    assert_eq!(collect_rel(&engine, "path").len(), 2); // (1,2), (2,3)

    // Add transitive rule.
    let src2 = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        edge(1, 2);
        edge(2, 3);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    "#;
    let prog2 = parse(src2);
    engine.update_program(&prog2);
    engine.run(&prog2);
    assert_eq!(collect_rel(&engine, "path").len(), 3); // (1,2), (2,3), (1,3)
}

#[test]
fn clean_deltas_after_run() {
    let src = r#"
        relation node(i32);
        node(1);
        node(2);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);
    engine.run(&program);

    // After run, recent should be empty (no pending delta/recent data).
    let rel = engine.relation("node").unwrap();
    assert_eq!(
        rel.iter_recent().count(),
        0,
        "recent should be empty after run"
    );
}

#[test]
fn update_program_adds_new_relation() {
    let src1 = r#"
        relation a(i32);
        a(1);
    "#;
    let prog1 = parse(src1);
    let mut engine = Engine::new(&prog1);
    engine.run(&prog1);

    // Expand program with a new relation.
    let src2 = r#"
        relation a(i32);
        relation b(i32);
        a(1);
        b(2);
    "#;
    let prog2 = parse(src2);
    engine.update_program(&prog2);
    engine.run(&prog2);

    assert_eq!(collect_rel(&engine, "a"), vec![vec![Value::I32(1)]]);
    assert_eq!(collect_rel(&engine, "b"), vec![vec![Value::I32(2)]]);
}
