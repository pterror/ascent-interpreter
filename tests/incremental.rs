//! Tests for persisting engine state across runs.

use ascent_eval::value::Value;
use ascent_eval::{Engine, SourceId};
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

// --- Source-tagged fact tests ---

#[test]
fn source_insert_and_retract() {
    let src = r#"
        relation edge(i32, i32);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s1 = engine.intern_source("file_a");
    let s2 = engine.intern_source("file_b");

    engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], s1);
    engine.insert_with_source("edge", vec![Value::I32(3), Value::I32(4)], s2);
    engine.insert_with_source("edge", vec![Value::I32(5), Value::I32(6)], s1);
    assert_eq!(engine.relation("edge").unwrap().len(), 3);

    // Retract source s1 — should remove 2 tuples, leave 1
    let removed = engine.retract_source(s1);
    assert_eq!(removed, 2);
    assert_eq!(engine.relation("edge").unwrap().len(), 1);

    // The surviving tuple is from s2
    let tuples = collect_rel(&engine, "edge");
    assert_eq!(tuples, vec![vec![Value::I32(3), Value::I32(4)]]);
}

#[test]
fn source_retract_and_rederive() {
    let src = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s1 = engine.intern_source("file_a");
    let s2 = engine.intern_source("file_b");

    // file_a contributes edge(1,2), file_b contributes edge(2,3)
    engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], s1);
    engine.insert_with_source("edge", vec![Value::I32(2), Value::I32(3)], s2);
    engine.run(&program);

    assert_eq!(collect_rel(&engine, "path").len(), 3); // (1,2), (2,3), (1,3)

    // Retract file_a's facts
    engine.retract_source(s1);
    assert_eq!(engine.relation("edge").unwrap().len(), 1); // only (2,3)

    // Derived facts from the previous run are still around (anonymous).
    // After re-run, only facts derivable from remaining edges survive.
    // For now, clear derived relation and re-run (full re-derivation).
    // (Smarter invalidation is step 3.3)
    engine
        .relation_mut("path")
        .unwrap()
        .retract_source(SourceId::ANONYMOUS);
    engine.run(&program);
    assert_eq!(collect_rel(&engine, "path").len(), 1); // only (2,3)
}

#[test]
fn source_intern_idempotent() {
    let src = r#"
        relation node(i32);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s1 = engine.intern_source("my_source");
    let s2 = engine.intern_source("my_source");
    assert_eq!(s1, s2);

    let s3 = engine.intern_source("other_source");
    assert_ne!(s1, s3);
}

#[test]
fn source_from_empty_body_rules() {
    // Facts from empty-body rules should be tagged with the current source.
    let src = r#"
        relation node(i32);
        node(1);
        node(2);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s = engine.intern_source("file_x");
    engine.set_source(s);
    engine.run(&program);

    assert_eq!(engine.relation("node").unwrap().len(), 2);

    // Retract source — both facts should be removed
    let removed = engine.retract_source(s);
    assert_eq!(removed, 2);
    assert_eq!(engine.relation("node").unwrap().len(), 0);
}

#[test]
fn source_derived_facts_stay_anonymous() {
    // Derived facts (from rules with bodies) should NOT get source-tagged.
    let src = r#"
        relation edge(i32, i32);
        relation path(i32, i32);
        path(x, y) <-- edge(x, y);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s = engine.intern_source("my_file");
    engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], s);
    engine.set_source(s);
    engine.run(&program);

    // edge(1,2) is tagged with s, path(1,2) is derived (anonymous)
    assert_eq!(engine.relation("path").unwrap().len(), 1);

    // Retracting s removes edge(1,2) but NOT path(1,2)
    engine.retract_source(s);
    assert_eq!(engine.relation("edge").unwrap().len(), 0);
    assert_eq!(engine.relation("path").unwrap().len(), 1); // still there (anonymous)
}

#[test]
fn source_multiple_sources_independent() {
    let src = r#"
        relation fact(i32);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let a = engine.intern_source("a");
    let b = engine.intern_source("b");
    let c = engine.intern_source("c");

    engine.insert_with_source("fact", vec![Value::I32(1)], a);
    engine.insert_with_source("fact", vec![Value::I32(2)], b);
    engine.insert_with_source("fact", vec![Value::I32(3)], c);
    engine.insert_with_source("fact", vec![Value::I32(4)], a);

    assert_eq!(engine.relation("fact").unwrap().len(), 4);

    // Retract b — only its fact removed
    engine.retract_source(b);
    assert_eq!(engine.relation("fact").unwrap().len(), 3);

    // Retract a — removes 2 facts
    engine.retract_source(a);
    assert_eq!(engine.relation("fact").unwrap().len(), 1);

    let tuples = collect_rel(&engine, "fact");
    assert_eq!(tuples, vec![vec![Value::I32(3)]]);
}

#[test]
fn source_retract_then_reinsert() {
    let src = r#"
        relation edge(i32, i32);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s = engine.intern_source("file");
    engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], s);
    assert_eq!(engine.relation("edge").unwrap().len(), 1);

    // Retract and re-insert the same fact
    engine.retract_source(s);
    assert_eq!(engine.relation("edge").unwrap().len(), 0);

    engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], s);
    assert_eq!(engine.relation("edge").unwrap().len(), 1);
}

#[test]
fn source_untagged_facts_survive_retraction() {
    let src = r#"
        relation node(i32);
    "#;
    let program = parse(src);
    let mut engine = Engine::new(&program);

    let s = engine.intern_source("tagged");

    // Mix tagged and untagged inserts
    engine.insert("node", vec![Value::I32(1)]); // anonymous
    engine.insert_with_source("node", vec![Value::I32(2)], s); // tagged
    engine.insert("node", vec![Value::I32(3)]); // anonymous

    assert_eq!(engine.relation("node").unwrap().len(), 3);

    // Retract tagged source — only its fact removed
    engine.retract_source(s);
    assert_eq!(engine.relation("node").unwrap().len(), 2);

    let tuples = collect_rel(&engine, "node");
    assert_eq!(tuples, vec![vec![Value::I32(1)], vec![Value::I32(3)]]);
}
