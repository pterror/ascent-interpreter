use proptest::prelude::*;
use proptest::strategy::ValueTree;

use ascent_syntax::desugar::desugar_program;
use ascent_syntax::{AscentProgram, BodyClauseArg, BodyItemNode};

// --- String generation strategies ---

fn arb_type() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("i32"),
        Just("i64"),
        Just("bool"),
        Just("String"),
        Just("u32"),
        Just("u64"),
    ]
}

fn arb_lattice_type() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just("Dual<i32>"), Just("Dual<u32>"), Just("Dual<i64>"),]
}

fn arb_ident() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("alpha".to_string()),
        Just("beta".to_string()),
        Just("gamma".to_string()),
        Just("delta".to_string()),
        Just("epsilon".to_string()),
        Just("zeta".to_string()),
        Just("eta".to_string()),
        Just("theta".to_string()),
        Just("iota".to_string()),
        Just("kappa".to_string()),
    ]
}

fn arb_var() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("x".to_string()),
        Just("y".to_string()),
        Just("z".to_string()),
        Just("w".to_string()),
        Just("v".to_string()),
        Just("u".to_string()),
    ]
}

fn arb_int_literal() -> impl Strategy<Value = String> {
    (-1000i32..1000).prop_map(|n| {
        if n < 0 {
            format!("({})", n)
        } else {
            n.to_string()
        }
    })
}

fn arb_bool_literal() -> impl Strategy<Value = String> {
    prop_oneof![Just("true".to_string()), Just("false".to_string()),]
}

fn arb_string_literal() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(r#""hello""#.to_string()),
        Just(r#""world""#.to_string()),
        Just(r#""foo""#.to_string()),
        Just(r#""bar""#.to_string()),
        Just(r#""""#.to_string()),
    ]
}

#[allow(dead_code)]
fn arb_literal() -> impl Strategy<Value = String> {
    prop_oneof![arb_int_literal(), arb_bool_literal(), arb_string_literal(),]
}

// --- Relation declaration generators ---

fn arb_relation_decl(name: String) -> impl Strategy<Value = (String, usize)> {
    (1..=5usize).prop_flat_map(move |arity| {
        let name = name.clone();
        proptest::collection::vec(arb_type(), arity..=arity).prop_map(move |types| {
            let types_str = types.join(", ");
            let decl = format!("relation {}({});", name, types_str);
            (decl, arity)
        })
    })
}

fn arb_lattice_decl(name: String) -> impl Strategy<Value = (String, usize)> {
    (1..=3usize).prop_flat_map(move |key_arity| {
        let name = name.clone();
        proptest::collection::vec(arb_type(), key_arity..=key_arity).prop_flat_map(
            move |key_types| {
                let name = name.clone();
                let key_types = key_types.clone();
                arb_lattice_type().prop_map(move |lat_type| {
                    let mut all_types: Vec<&str> = key_types.clone();
                    all_types.push(lat_type);
                    let total = all_types.len();
                    let types_str = all_types.join(", ");
                    let decl = format!("lattice {}({});", name, types_str);
                    (decl, total)
                })
            },
        )
    })
}

// --- Fact generator ---

fn arb_fact(rel_name: String, arity: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(arb_int_literal(), arity..=arity).prop_map(move |args| {
        let args_str = args.join(", ");
        format!("{}({});", rel_name, args_str)
    })
}

// --- Rule generators ---

fn arb_body_clause(rel_name: String, arity: usize) -> impl Strategy<Value = String> {
    proptest::collection::vec(arb_var(), arity..=arity)
        .prop_map(move |vars| format!("{}({})", rel_name, vars.join(", ")))
}

fn arb_simple_rule(
    head_name: String,
    head_arity: usize,
    body_rels: Vec<(String, usize)>,
) -> impl Strategy<Value = (String, usize)> {
    let body_strats: Vec<_> = body_rels
        .into_iter()
        .map(|(name, arity)| arb_body_clause(name, arity).boxed())
        .collect();
    let body_count = body_strats.len();

    proptest::collection::vec(arb_var(), head_arity..=head_arity).prop_flat_map(move |head_vars| {
        let head_name = head_name.clone();
        let body_strats = body_strats.clone();
        body_strats
            .into_iter()
            .collect::<Vec<_>>()
            .prop_map(move |body_clauses| {
                let head = format!("{}({})", head_name, head_vars.join(", "));
                let body = body_clauses.join(", ");
                let rule = format!("{} <-- {};", head, body);
                (rule, body_count)
            })
    })
}

// --- Full program generator ---

const REL_NAMES: [&str; 4] = ["alpha", "beta", "gamma", "delta"];

fn arb_program_with_facts() -> impl Strategy<Value = (String, usize, usize, Vec<usize>)> {
    (1..=4usize).prop_flat_map(|num_rels| {
        let names: Vec<String> = REL_NAMES
            .iter()
            .take(num_rels)
            .map(|s| s.to_string())
            .collect();

        let decl_strats: Vec<_> = names
            .iter()
            .map(|name| arb_relation_decl(name.clone()).boxed())
            .collect();

        decl_strats.prop_flat_map(move |decls| {
            let arities: Vec<usize> = decls.iter().map(|(_, a)| *a).collect();
            let decl_strs: Vec<String> = decls.iter().map(|(d, _)| d.clone()).collect();
            let names = names.clone();
            let num_rels = names.len();

            let arity0 = arities[0];
            let name0 = names[0].clone();
            let arities_clone = arities.clone();
            proptest::collection::vec(arb_fact(name0, arity0), 0..=3).prop_map(move |facts| {
                let mut lines = decl_strs.clone();
                let fact_count = facts.len();
                lines.extend(facts);
                let program = lines.join("\n");
                (program, num_rels, fact_count, arities_clone.clone())
            })
        })
    })
}

fn arb_program_with_rules() -> impl Strategy<Value = (String, usize, usize)> {
    (2..=3usize).prop_flat_map(|num_rels| {
        let names: Vec<String> = REL_NAMES
            .iter()
            .take(num_rels)
            .map(|s| s.to_string())
            .collect();

        let decl_strats: Vec<_> = names
            .iter()
            .map(|name| arb_relation_decl(name.clone()).boxed())
            .collect();

        decl_strats.prop_flat_map(move |decls| {
            let arities: Vec<usize> = decls.iter().map(|(_, a)| *a).collect();
            let decl_strs: Vec<String> = decls.iter().map(|(d, _)| d.clone()).collect();
            let names = names.clone();
            let num_rels = names.len();

            let head_name = names[num_rels - 1].clone();
            let head_arity = arities[num_rels - 1];
            let body_rels: Vec<(String, usize)> = names[..num_rels - 1]
                .iter()
                .zip(arities[..num_rels - 1].iter())
                .map(|(n, a)| (n.clone(), *a))
                .collect();

            arb_simple_rule(head_name, head_arity, body_rels).prop_map(
                move |(rule_text, _body_count)| {
                    let mut lines = decl_strs.clone();
                    lines.push(rule_text);
                    let program = lines.join("\n");
                    (program, num_rels, 1)
                },
            )
        })
    })
}

// --- Helper ---

fn parse_program(input: &str) -> AscentProgram {
    syn::parse_str(input).expect("generated program should parse")
}

// --- Property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // == Category 1: Relation declaration properties ==

    #[test]
    fn relation_count_matches(
        (program_text, expected_rels, _, _) in arb_program_with_facts()
    ) {
        let prog = parse_program(&program_text);
        prop_assert_eq!(prog.relations.len(), expected_rels);
    }

    #[test]
    fn relation_arities_match(
        (program_text, _, _, expected_arities) in arb_program_with_facts()
    ) {
        let prog = parse_program(&program_text);
        for (rel, expected_arity) in prog.relations.iter().zip(expected_arities.iter()) {
            prop_assert_eq!(
                rel.field_types.len(),
                *expected_arity,
                "relation {} has wrong arity",
                rel.name
            );
        }
    }

    #[test]
    fn relations_are_not_lattice(
        (program_text, _, _, _) in arb_program_with_facts()
    ) {
        let prog = parse_program(&program_text);
        for rel in &prog.relations {
            prop_assert!(!rel.is_lattice, "relation {} should not be a lattice", rel.name);
        }
    }

    #[test]
    fn lattice_flag_is_set(name in arb_ident()) {
        let (decl_text, arity) = arb_lattice_decl(name)
            .new_tree(&mut proptest::test_runner::TestRunner::default())
            .unwrap()
            .current();
        let prog: AscentProgram = syn::parse_str(&decl_text).unwrap();
        prop_assert_eq!(prog.relations.len(), 1);
        prop_assert!(prog.relations[0].is_lattice);
        prop_assert_eq!(prog.relations[0].field_types.len(), arity);
    }

    // == Category 2: Fact properties ==

    #[test]
    fn facts_have_empty_body(
        (program_text, _, fact_count, _) in arb_program_with_facts()
    ) {
        let prog = parse_program(&program_text);
        prop_assert_eq!(prog.rules.len(), fact_count);
        for rule in &prog.rules {
            prop_assert!(
                rule.body_items.is_empty(),
                "fact should have empty body"
            );
        }
    }

    #[test]
    fn fact_head_arg_count_matches_arity(
        (program_text, _, _, arities) in arb_program_with_facts()
    ) {
        if arities.is_empty() {
            return Ok(());
        }
        let prog = parse_program(&program_text);
        let first_arity = arities[0];
        for rule in &prog.rules {
            let head = rule.head_clauses.first().unwrap().clause();
            prop_assert_eq!(
                head.args.len(),
                first_arity,
                "fact head arg count should match relation arity"
            );
        }
    }

    // == Category 3: Rule properties ==

    #[test]
    fn rule_body_count_preserved(
        (program_text, _, rule_count) in arb_program_with_rules()
    ) {
        let prog = parse_program(&program_text);
        prop_assert_eq!(prog.rules.len(), rule_count);
    }

    #[test]
    fn rule_relation_names_preserved(
        (program_text, num_rels, _) in arb_program_with_rules()
    ) {
        let prog = parse_program(&program_text);
        let head_name = REL_NAMES[num_rels - 1];
        for rule in &prog.rules {
            let head = rule.head_clauses.first().unwrap().clause();
            prop_assert_eq!(head.rel.to_string(), head_name);
        }
        for rule in &prog.rules {
            for bitem in &rule.body_items {
                if let BodyItemNode::Clause(cl) = bitem {
                    let rel_name = cl.rel.to_string();
                    prop_assert!(
                        REL_NAMES[..num_rels - 1].contains(&rel_name.as_str()),
                        "body clause relation '{}' should be one of the declared body relations",
                        rel_name
                    );
                }
            }
        }
    }

    #[test]
    fn rule_head_has_clauses(
        (program_text, _, _) in arb_program_with_rules()
    ) {
        let prog = parse_program(&program_text);
        for rule in &prog.rules {
            prop_assert!(!rule.head_clauses.is_empty(), "rule head should not be empty");
            prop_assert!(!rule.body_items.is_empty(), "rule body should not be empty");
        }
    }

    // == Category 4: Desugaring invariants ==

    #[test]
    fn disjunction_desugared_away(
        num_disjuncts in 2..=4usize,
    ) {
        let branches: Vec<String> = REL_NAMES
            .iter()
            .take(num_disjuncts)
            .map(|name| format!("{}(x)", name))
            .collect();
        let mut decls: Vec<String> = branches.iter().enumerate().map(|(i, _)| {
            format!("relation {}(i32);", REL_NAMES[i])
        }).collect();
        decls.push("relation result(i32);".to_string());
        let disjunction = branches.join(" | ");
        let rule = format!("result(x) <-- ({});", disjunction);
        decls.push(rule);
        let program_text = decls.join("\n");

        let prog = parse_program(&program_text);
        let desugared = desugar_program(prog);

        prop_assert_eq!(
            desugared.rules.len(),
            num_disjuncts,
            "disjunction with {} branches should produce {} rules",
            num_disjuncts,
            num_disjuncts,
        );

        for rule in &desugared.rules {
            for bitem in &rule.body_items {
                prop_assert!(
                    !matches!(bitem, BodyItemNode::Disjunction(_)),
                    "no disjunction nodes should remain after desugaring"
                );
            }
        }
    }

    #[test]
    fn wildcards_desugared_away(num_wildcards in 1..=4usize) {
        let args: Vec<String> = (0..num_wildcards).map(|_| "_".to_string()).collect();
        let types: Vec<&str> = (0..num_wildcards).map(|_| "i32").collect();
        let program_text = format!(
            "relation alpha({});\nrelation result(i32);\nresult(1) <-- alpha({});",
            types.join(", "),
            args.join(", "),
        );

        let prog = parse_program(&program_text);
        let desugared = desugar_program(prog);

        prop_assert_eq!(desugared.rules.len(), 1);
        if let BodyItemNode::Clause(cl) = &desugared.rules[0].body_items[0] {
            for arg in cl.args.iter() {
                if let BodyClauseArg::Expr(expr) = arg {
                    prop_assert!(
                        !ascent_syntax::is_wild_card(expr),
                        "no wildcards should remain after desugaring"
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_vars_get_conditions(
        arity in 2..=4usize,
    ) {
        let args: Vec<&str> = (0..arity).map(|_| "x").collect();
        let types: Vec<&str> = (0..arity).map(|_| "i32").collect();
        let program_text = format!(
            "relation alpha({});\nrelation result(i32);\nresult(x) <-- alpha({});",
            types.join(", "),
            args.join(", "),
        );

        let prog = parse_program(&program_text);
        let desugared = desugar_program(prog);

        prop_assert_eq!(desugared.rules.len(), 1);
        if let BodyItemNode::Clause(cl) = &desugared.rules[0].body_items[0] {
            prop_assert_eq!(
                cl.cond_clauses.len(),
                arity - 1,
                "repeated var x appearing {} times should produce {} equality conditions",
                arity,
                arity - 1,
            );
        }
    }

    // == Category 5: Parser robustness ==

    #[test]
    fn empty_program_parses(_dummy in 0..1i32) {
        let prog: AscentProgram = syn::parse_str("").unwrap();
        prop_assert!(prog.relations.is_empty());
        prop_assert!(prog.rules.is_empty());
    }

    #[test]
    fn single_column_relation_parses(ty in arb_type()) {
        let program_text = format!("relation alpha({});", ty);
        let prog: AscentProgram = syn::parse_str(&program_text).unwrap();
        prop_assert_eq!(prog.relations.len(), 1);
        prop_assert_eq!(prog.relations[0].field_types.len(), 1);
    }

    #[test]
    fn many_rules_parse(num_facts in 1..=20usize) {
        let mut lines = vec!["relation alpha(i32);".to_string()];
        for i in 0..num_facts {
            lines.push(format!("alpha({});", i));
        }
        let program_text = lines.join("\n");
        let prog: AscentProgram = syn::parse_str(&program_text).unwrap();
        prop_assert_eq!(prog.relations.len(), 1);
        prop_assert_eq!(prog.rules.len(), num_facts);
    }

    #[test]
    fn condition_rules_parse(
        threshold in 0..100i32,
    ) {
        let program_text = format!(
            "relation alpha(i32);\nrelation result(i32);\nresult(x) <-- alpha(x), if x > {};",
            threshold,
        );
        let prog: AscentProgram = syn::parse_str(&program_text).unwrap();
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert_eq!(prog.rules[0].body_items.len(), 2);
        prop_assert!(matches!(&prog.rules[0].body_items[1], BodyItemNode::Cond(_)));
    }

    #[test]
    fn generator_rules_parse(
        end in 1..=100i32,
    ) {
        let program_text = format!(
            "relation alpha(i32);\nalpha(x) <-- for x in 0..{};",
            end,
        );
        let prog: AscentProgram = syn::parse_str(&program_text).unwrap();
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert!(matches!(&prog.rules[0].body_items[0], BodyItemNode::Generator(_)));
    }

    #[test]
    fn negation_parses(
        _dummy in 0..1i32,
    ) {
        let program_text =
            "relation alpha(i32);\nrelation beta(i32);\nbeta(x) <-- alpha(x), !beta(x);";
        let prog: AscentProgram = syn::parse_str(program_text).unwrap();
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert!(matches!(&prog.rules[0].body_items[1], BodyItemNode::Negation(_)));
    }

    #[test]
    fn aggregation_parses(
        agg_fn in prop_oneof![
            Just("min"),
            Just("max"),
            Just("count"),
            Just("sum"),
        ]
    ) {
        let program_text = format!(
            "relation alpha(i32);\nrelation result(i32);\nresult(y) <-- agg y = {}(x) in alpha(x);",
            agg_fn,
        );
        let prog: AscentProgram = syn::parse_str(&program_text).unwrap();
        prop_assert_eq!(prog.rules.len(), 1);
        prop_assert!(matches!(&prog.rules[0].body_items[0], BodyItemNode::Agg(_)));
    }

    #[test]
    fn desugar_preserves_relation_count(
        (program_text, expected_rels, _, _) in arb_program_with_facts()
    ) {
        let prog = parse_program(&program_text);
        let desugared = desugar_program(prog);
        prop_assert_eq!(
            desugared.relations.len(),
            expected_rels,
            "desugaring should not change relation count"
        );
    }

    #[test]
    fn full_program_roundtrip(
        (program_text, num_rels, _rule_count) in arb_program_with_rules()
    ) {
        let prog = parse_program(&program_text);
        let desugared = desugar_program(prog);
        prop_assert_eq!(desugared.relations.len(), num_rels);
        for rule in &desugared.rules {
            prop_assert!(!rule.head_clauses.is_empty());
        }
    }
}
