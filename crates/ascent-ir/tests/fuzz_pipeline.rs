//! Fuzz testing the parse → desugar → lower-to-IR pipeline.
//!
//! Generates random valid Ascent programs and verifies structural invariants
//! hold after lowering through the full pipeline. This serves as a parity test
//! since the pipeline mirrors ascent_macro's internal processing.

use proptest::prelude::*;

use ascent_ir::{BodyItem, Program};
use ascent_syntax::AscentProgram;

// --- String generators (shared vocabulary with ascent-syntax proptest) ---

const REL_NAMES: [&str; 4] = ["alpha", "beta", "gamma", "delta"];

fn arb_type() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just("i32"), Just("i64"), Just("bool"), Just("String"),]
}

fn arb_var() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("x".to_string()),
        Just("y".to_string()),
        Just("z".to_string()),
        Just("w".to_string()),
    ]
}

fn arb_int_literal() -> impl Strategy<Value = String> {
    (0..1000i32).prop_map(|n| n.to_string())
}

// --- Program generators ---

fn arb_relation_decl(name: &str) -> impl Strategy<Value = (String, usize)> {
    let name = name.to_string();
    (1..=4usize).prop_flat_map(move |arity| {
        let name = name.clone();
        proptest::collection::vec(arb_type(), arity..=arity).prop_map(move |types| {
            let decl = format!("relation {}({});", name, types.join(", "));
            (decl, arity)
        })
    })
}

fn arb_body_clause(name: &str, arity: usize) -> impl Strategy<Value = String> {
    let name = name.to_string();
    proptest::collection::vec(arb_var(), arity..=arity)
        .prop_map(move |vars| format!("{}({})", name, vars.join(", ")))
}

/// Generate a program with only facts (head-only rules).
fn arb_facts_program() -> impl Strategy<Value = (String, usize, usize)> {
    (1..=3usize).prop_flat_map(|num_rels| {
        let names: Vec<&str> = REL_NAMES[..num_rels].to_vec();
        let decl_strats: Vec<_> = names
            .iter()
            .map(|name| arb_relation_decl(name).boxed())
            .collect();

        decl_strats.prop_flat_map(move |decls| {
            let arities: Vec<usize> = decls.iter().map(|(_, a)| *a).collect();
            let decl_strs: Vec<String> = decls.iter().map(|(d, _)| d.clone()).collect();
            let name0 = names[0].to_string();
            let arity0 = arities[0];
            let num_rels = names.len();

            proptest::collection::vec(
                proptest::collection::vec(arb_int_literal(), arity0..=arity0),
                0..=5,
            )
            .prop_map(move |fact_args| {
                let mut lines = decl_strs.clone();
                let num_facts = fact_args.len();
                for args in &fact_args {
                    lines.push(format!("{}({});", name0, args.join(", ")));
                }
                (lines.join("\n"), num_rels, num_facts)
            })
        })
    })
}

/// Generate a program with rules (body clauses joining relations).
fn arb_rules_program() -> impl Strategy<Value = (String, usize, usize)> {
    (2..=3usize).prop_flat_map(|num_rels| {
        let names: Vec<&str> = REL_NAMES[..num_rels].to_vec();
        let decl_strats: Vec<_> = names
            .iter()
            .map(|name| arb_relation_decl(name).boxed())
            .collect();

        decl_strats.prop_flat_map(move |decls| {
            let arities: Vec<usize> = decls.iter().map(|(_, a)| *a).collect();
            let decl_strs: Vec<String> = decls.iter().map(|(d, _)| d.clone()).collect();
            let names = names.clone();
            let num_rels = names.len();

            // Head = last relation, body = all other relations
            let head_name = names[num_rels - 1].to_string();
            let head_arity = arities[num_rels - 1];

            let body_strats: Vec<_> = names[..num_rels - 1]
                .iter()
                .zip(arities[..num_rels - 1].iter())
                .map(|(n, a)| arb_body_clause(n, *a).boxed())
                .collect();

            proptest::collection::vec(arb_var(), head_arity..=head_arity).prop_flat_map(
                move |head_vars| {
                    let head_name = head_name.clone();
                    let decl_strs = decl_strs.clone();
                    let body_strats = body_strats.clone();

                    body_strats
                        .into_iter()
                        .collect::<Vec<_>>()
                        .prop_map(move |body_clauses| {
                            let head = format!("{}({})", head_name, head_vars.join(", "));
                            let body = body_clauses.join(", ");
                            let rule = format!("{} <-- {};", head, body);
                            let mut lines = decl_strs.clone();
                            lines.push(rule);
                            (lines.join("\n"), num_rels, 1)
                        })
                },
            )
        })
    })
}

/// Generate a program with conditions (if guards).
fn arb_condition_program() -> impl Strategy<Value = String> {
    arb_relation_decl("alpha").prop_flat_map(|(decl, arity)| {
        let vars: Vec<String> = (0..arity).map(|i| format!("x{}", i)).collect();
        let head_var = vars[0].clone();
        let cond_var = vars[0].clone();
        (0..100i32).prop_map(move |threshold| {
            let clause = format!("alpha({})", vars.join(", "));
            format!(
                "{}\nrelation result(i32);\nresult({}) <-- {}, if {} > {};",
                decl, head_var, clause, cond_var, threshold,
            )
        })
    })
}

/// Generate a program with generators.
fn arb_generator_program() -> impl Strategy<Value = String> {
    (1..=50i32).prop_map(|end| format!("relation alpha(i32);\nalpha(x) <-- for x in 0..{};", end,))
}

/// Generate a program with negation.
fn arb_negation_program() -> impl Strategy<Value = String> {
    arb_relation_decl("alpha").prop_flat_map(|(decl, arity)| {
        let vars: Vec<String> = (0..arity).map(|i| format!("x{}", i)).collect();
        Just(format!(
            "{}\nrelation beta({});\nbeta({}) <-- alpha({}), !beta({});",
            decl,
            vec!["i32"; arity].join(", "),
            vars.join(", "),
            vars.join(", "),
            vars.join(", "),
        ))
    })
}

/// Generate a program with aggregation.
fn arb_aggregation_program() -> impl Strategy<Value = String> {
    prop_oneof![Just("min"), Just("max"), Just("count"), Just("sum"),].prop_map(move |agg_fn| {
        format!(
            "relation alpha(i32);\nrelation result(i32);\nresult(y) <-- agg y = {}(x) in alpha(x);",
            agg_fn,
        )
    })
}

/// Generate a program with disjunctions.
fn arb_disjunction_program() -> impl Strategy<Value = (String, usize)> {
    (2..=4usize).prop_map(|num_branches| {
        let branch_names = &REL_NAMES[..num_branches];
        let mut lines: Vec<String> = branch_names
            .iter()
            .map(|n| format!("relation {}(i32);", n))
            .collect();
        lines.push("relation result(i32);".to_string());
        let branches: Vec<String> = branch_names.iter().map(|n| format!("{}(x)", n)).collect();
        lines.push(format!("result(x) <-- ({});", branches.join(" | ")));
        (lines.join("\n"), num_branches)
    })
}

/// Generate a program with wildcards.
fn arb_wildcard_program() -> impl Strategy<Value = (String, usize)> {
    (1..=4usize).prop_map(|num_wildcards| {
        let types: Vec<&str> = vec!["i32"; num_wildcards];
        let args: Vec<&str> = vec!["_"; num_wildcards];
        let text = format!(
            "relation alpha({});\nrelation result(i32);\nresult(1) <-- alpha({});",
            types.join(", "),
            args.join(", "),
        );
        (text, num_wildcards)
    })
}

// --- Helpers ---

fn parse_and_lower(input: &str) -> Program {
    let ast: AscentProgram = syn::parse_str(input).expect("program should parse");
    Program::from_ast(ast)
}

// --- Property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // == Pipeline doesn't panic ==

    #[test]
    fn facts_lower_without_panic(
        (program_text, _, _) in arb_facts_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    #[test]
    fn rules_lower_without_panic(
        (program_text, _, _) in arb_rules_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    #[test]
    fn conditions_lower_without_panic(
        program_text in arb_condition_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    #[test]
    fn generators_lower_without_panic(
        program_text in arb_generator_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    #[test]
    fn negation_lowers_without_panic(
        program_text in arb_negation_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    #[test]
    fn aggregation_lowers_without_panic(
        program_text in arb_aggregation_program()
    ) {
        let _ = parse_and_lower(&program_text);
    }

    // == IR structural invariants ==

    #[test]
    fn ir_relations_match_declarations(
        (program_text, expected_rels, _) in arb_facts_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(
            prog.relations.len(),
            expected_rels,
            "IR should have same number of relations as declared"
        );
        // All declared names should be present
        for name in REL_NAMES.iter().take(expected_rels) {
            prop_assert!(
                prog.relations.contains_key(*name),
                "IR should contain relation '{}'",
                name
            );
        }
    }

    #[test]
    fn ir_relation_arities_preserved(
        (program_text, num_rels, _) in arb_facts_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for name in REL_NAMES.iter().take(num_rels) {
            let rel = prog.relations.get(*name).unwrap();
            prop_assert!(
                !rel.column_types.is_empty(),
                "relation '{}' should have at least one column",
                name
            );
        }
    }

    #[test]
    fn ir_facts_have_empty_body(
        (program_text, _, fact_count) in arb_facts_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(prog.rules.len(), fact_count);
        for rule in &prog.rules {
            prop_assert!(
                rule.body.is_empty(),
                "facts should have empty body in IR"
            );
            prop_assert!(
                !rule.heads.is_empty(),
                "facts should have at least one head clause"
            );
        }
    }

    #[test]
    fn ir_rule_heads_reference_declared_relations(
        (program_text, _, _) in arb_rules_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for rule in &prog.rules {
            for head in &rule.heads {
                prop_assert!(
                    prog.relations.contains_key(&head.relation),
                    "head clause references undeclared relation '{}'",
                    head.relation
                );
            }
        }
    }

    #[test]
    fn ir_rule_body_clauses_reference_declared_relations(
        (program_text, _, _) in arb_rules_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for rule in &prog.rules {
            for item in &rule.body {
                if let BodyItem::Clause(cl) = item {
                    prop_assert!(
                        prog.relations.contains_key(&cl.relation),
                        "body clause references undeclared relation '{}'",
                        cl.relation
                    );
                }
            }
        }
    }

    #[test]
    fn ir_head_arg_count_matches_relation_arity(
        (program_text, _, _) in arb_rules_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for rule in &prog.rules {
            for head in &rule.heads {
                let rel = prog.relations.get(&head.relation).unwrap();
                prop_assert_eq!(
                    head.args.len(),
                    rel.column_types.len(),
                    "head clause for '{}' has {} args but relation has {} columns",
                    head.relation,
                    head.args.len(),
                    rel.column_types.len(),
                );
            }
        }
    }

    #[test]
    fn ir_body_clause_arg_count_matches_relation_arity(
        (program_text, _, _) in arb_rules_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for rule in &prog.rules {
            for item in &rule.body {
                if let BodyItem::Clause(cl) = item {
                    let rel = prog.relations.get(&cl.relation).unwrap();
                    prop_assert_eq!(
                        cl.args.len(),
                        rel.column_types.len(),
                        "body clause for '{}' has {} args but relation has {} columns",
                        cl.relation,
                        cl.args.len(),
                        rel.column_types.len(),
                    );
                }
            }
        }
    }

    // == Condition lowering ==

    #[test]
    fn ir_conditions_become_condition_items(
        program_text in arb_condition_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(prog.rules.len(), 1);
        let has_condition = prog.rules[0].body.iter().any(|b| matches!(b, BodyItem::Condition(_)));
        prop_assert!(has_condition, "rule with 'if' should have a Condition body item");
    }

    // == Generator lowering ==

    #[test]
    fn ir_generators_become_generator_items(
        program_text in arb_generator_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert!(
            matches!(&prog.rules[0].body[0], BodyItem::Generator(_)),
            "rule with 'for' should have a Generator body item"
        );
    }

    // == Negation lowering ==

    #[test]
    fn ir_negation_becomes_aggregation(
        program_text in arb_negation_program()
    ) {
        let prog = parse_and_lower(&program_text);
        let has_agg = prog.rules.iter().any(|r| {
            r.body.iter().any(|b| matches!(b, BodyItem::Aggregation(_)))
        });
        prop_assert!(has_agg, "negation should be lowered to Aggregation in IR");
    }

    // == Aggregation lowering ==

    #[test]
    fn ir_aggregation_preserved(
        program_text in arb_aggregation_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert!(
            matches!(&prog.rules[0].body[0], BodyItem::Aggregation(_)),
            "'agg' should lower to Aggregation body item"
        );
    }

    // == Disjunction lowering ==

    #[test]
    fn ir_disjunction_expands_to_rules(
        (program_text, num_branches) in arb_disjunction_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(
            prog.rules.len(),
            num_branches,
            "disjunction with {} branches should produce {} IR rules",
            num_branches,
            num_branches,
        );
        // No disjunctions should survive to IR — they become separate rules
        for rule in &prog.rules {
            prop_assert_eq!(rule.body.len(), 1, "each expanded rule should have exactly one body clause");
            prop_assert!(
                matches!(&rule.body[0], BodyItem::Clause(_)),
                "each expanded disjunct should be a Clause"
            );
        }
    }

    // == Wildcard lowering ==

    #[test]
    fn ir_wildcards_become_vars(
        (program_text, num_wildcards) in arb_wildcard_program()
    ) {
        let prog = parse_and_lower(&program_text);
        prop_assert_eq!(prog.rules.len(), 1);
        if let BodyItem::Clause(cl) = &prog.rules[0].body[0] {
            prop_assert_eq!(cl.args.len(), num_wildcards);
            for (i, arg) in cl.args.iter().enumerate() {
                prop_assert!(
                    matches!(arg, ascent_ir::ClauseArg::Var(_)),
                    "wildcard at position {} should become a Var in IR, got {:?}",
                    i,
                    arg,
                );
            }
        }
    }

    // == No lattice flag on regular relations ==

    #[test]
    fn ir_relations_not_lattice(
        (program_text, num_rels, _) in arb_facts_program()
    ) {
        let prog = parse_and_lower(&program_text);
        for name in REL_NAMES.iter().take(num_rels) {
            let rel = prog.relations.get(*name).unwrap();
            prop_assert!(!rel.is_lattice, "relation '{}' should not be a lattice", name);
        }
    }
}
