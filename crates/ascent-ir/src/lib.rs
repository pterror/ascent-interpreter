//! Intermediate representation for the Ascent interpreter.
//!
//! This IR is a simplified form of the AST, designed for efficient interpretation.
//! It removes syntactic sugar and normalizes the program structure.

use std::collections::HashMap;

use ascent_syntax::{
    AggClauseNode, AscentProgram, BodyClauseArg, BodyItemNode, CondClause, GeneratorNode,
    HeadClauseNode, HeadItemNode, RuleNode, desugar::desugar_program,
};
use syn::{Expr, Type};

/// A complete Ascent program in IR form.
#[derive(Debug)]
pub struct Program {
    /// All relations in the program.
    pub relations: HashMap<String, Relation>,
    /// All rules, grouped by strongly connected components.
    pub rules: Vec<Rule>,
}

/// A relation declaration.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Relation name.
    pub name: String,
    /// Column types (as syn::Type for display).
    pub column_types: Vec<Type>,
    /// Whether this is a lattice.
    pub is_lattice: bool,
    /// Initial values (if any).
    pub initialization: Option<Expr>,
}

/// A rule: head <- body.
#[derive(Debug)]
pub struct Rule {
    /// Head clauses (what to insert).
    pub heads: Vec<HeadClause>,
    /// Body items (what to match).
    pub body: Vec<BodyItem>,
}

/// A clause in the rule head.
#[derive(Debug, Clone)]
pub struct HeadClause {
    /// Relation to insert into.
    pub relation: String,
    /// Expressions for each column.
    pub args: Vec<Expr>,
}

/// A body item in a rule.
#[derive(Debug)]
pub enum BodyItem {
    /// Match against a relation: `rel(x, y)`
    Clause(Clause),
    /// Generate values: `for x in expr`
    Generator(Generator),
    /// Filter condition: `if expr`
    Condition(Condition),
    /// Aggregation: `agg x = f(y) in rel(z)`
    Aggregation(Aggregation),
}

/// A relation lookup in the body.
#[derive(Debug, Clone)]
pub struct Clause {
    /// Relation to look up.
    pub relation: String,
    /// Patterns for each column (variable names or expressions).
    pub args: Vec<ClauseArg>,
    /// Additional conditions on this clause.
    pub conditions: Vec<Condition>,
}

/// An argument in a body clause.
#[derive(Debug, Clone)]
pub enum ClauseArg {
    /// A variable to bind.
    Var(String),
    /// An expression to match.
    Expr(Expr),
}

/// A generator: `for pattern in expr`.
#[derive(Debug, Clone)]
pub struct Generator {
    /// Variables to bind.
    pub vars: Vec<String>,
    /// Expression to iterate.
    pub expr: Expr,
    /// The full pattern (for complex patterns).
    pub pattern: syn::Pat,
}

/// A filter condition.
#[derive(Debug, Clone)]
pub enum Condition {
    /// `if expr`
    If(Expr),
    /// `if let pattern = expr`
    IfLet { pattern: syn::Pat, expr: Expr },
    /// `let pattern = expr`
    Let { pattern: syn::Pat, expr: Expr },
}

/// An aggregation clause.
#[derive(Debug, Clone)]
pub struct Aggregation {
    /// Variables to bind from the aggregation result.
    pub result_vars: Vec<String>,
    /// The aggregator expression/path.
    pub aggregator: Expr,
    /// Variables to aggregate over.
    pub bound_vars: Vec<String>,
    /// Relation to aggregate from.
    pub relation: String,
    /// Arguments to the relation.
    pub args: Vec<Expr>,
}

impl Program {
    /// Lower an AST program to IR.
    pub fn from_ast(ast: AscentProgram) -> Self {
        // First desugar
        let ast = desugar_program(ast);

        // Collect relations
        let mut relations = HashMap::new();
        for rel in ast.relations {
            relations.insert(
                rel.name.to_string(),
                Relation {
                    name: rel.name.to_string(),
                    column_types: rel.field_types.into_iter().collect(),
                    is_lattice: rel.is_lattice,
                    initialization: rel.initialization,
                },
            );
        }

        // Lower rules
        let rules = ast.rules.into_iter().map(Rule::from_ast).collect();

        Program { relations, rules }
    }
}

impl Rule {
    fn from_ast(rule: RuleNode) -> Self {
        let heads = rule
            .head_clauses
            .into_iter()
            .filter_map(|h| match h {
                HeadItemNode::HeadClause(hc) => Some(HeadClause::from_ast(hc)),
                HeadItemNode::MacroInvocation(_) => None, // Should be expanded by desugar
            })
            .collect();

        let body = rule
            .body_items
            .into_iter()
            .map(BodyItem::from_ast)
            .collect();

        Rule { heads, body }
    }
}

impl HeadClause {
    fn from_ast(hc: HeadClauseNode) -> Self {
        HeadClause {
            relation: hc.rel.to_string(),
            args: hc.args.into_iter().collect(),
        }
    }
}

impl BodyItem {
    fn from_ast(bi: BodyItemNode) -> Self {
        match bi {
            BodyItemNode::Clause(cl) => BodyItem::Clause(Clause::from_ast(cl)),
            BodyItemNode::Generator(generator) => {
                BodyItem::Generator(Generator::from_ast(generator))
            }
            BodyItemNode::Cond(cond) => BodyItem::Condition(Condition::from_ast(cond)),
            BodyItemNode::Agg(agg) => BodyItem::Aggregation(Aggregation::from_ast(agg)),
            BodyItemNode::Negation(_) => {
                panic!("negation should be desugared to aggregation")
            }
            BodyItemNode::Disjunction(_) => {
                panic!("disjunction should be desugared to multiple rules")
            }
            BodyItemNode::MacroInvocation(_) => {
                panic!("macro invocations should be expanded")
            }
        }
    }
}

impl Clause {
    fn from_ast(cl: ascent_syntax::BodyClauseNode) -> Self {
        let args = cl
            .args
            .into_iter()
            .map(|arg| match arg {
                BodyClauseArg::Expr(e) => {
                    // Check if it's a simple variable
                    if let syn::Expr::Path(p) = &e
                        && let Some(ident) = p.path.get_ident()
                    {
                        return ClauseArg::Var(ident.to_string());
                    }
                    ClauseArg::Expr(e)
                }
                BodyClauseArg::Pat(_) => {
                    panic!("pattern args should be desugared")
                }
            })
            .collect();

        let conditions = cl
            .cond_clauses
            .into_iter()
            .map(Condition::from_ast)
            .collect();

        Clause {
            relation: cl.rel.to_string(),
            args,
            conditions,
        }
    }
}

impl Generator {
    fn from_ast(generator: GeneratorNode) -> Self {
        let vars = ascent_syntax::pattern_get_vars(&generator.pattern)
            .into_iter()
            .map(|i| i.to_string())
            .collect();

        Generator {
            vars,
            expr: generator.expr,
            pattern: generator.pattern,
        }
    }
}

impl Condition {
    fn from_ast(cond: CondClause) -> Self {
        match cond {
            CondClause::If(c) => Condition::If(c.cond),
            CondClause::IfLet(c) => Condition::IfLet {
                pattern: c.pattern,
                expr: c.exp,
            },
            CondClause::Let(c) => Condition::Let {
                pattern: c.pattern,
                expr: c.exp,
            },
        }
    }
}

impl Aggregation {
    fn from_ast(agg: AggClauseNode) -> Self {
        let result_vars = ascent_syntax::pattern_get_vars(&agg.pat)
            .into_iter()
            .map(|i| i.to_string())
            .collect();

        let bound_vars = agg.bound_args.into_iter().map(|i| i.to_string()).collect();

        let aggregator = match agg.aggregator {
            ascent_syntax::AggregatorNode::Path(p) => syn::parse2(quote::quote! { #p }).unwrap(),
            ascent_syntax::AggregatorNode::Expr(e) => e,
        };

        Aggregation {
            result_vars,
            aggregator,
            bound_vars,
            relation: agg.rel.to_string(),
            args: agg.rel_args.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_lower(input: &str) -> Program {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        Program::from_ast(ast)
    }

    #[test]
    fn test_lower_simple_program() {
        let prog = parse_and_lower(
            r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        );

        assert_eq!(prog.relations.len(), 2);
        assert!(prog.relations.contains_key("edge"));
        assert!(prog.relations.contains_key("path"));
        assert_eq!(prog.rules.len(), 2);
    }

    #[test]
    fn test_lower_with_condition() {
        let prog = parse_and_lower(
            r#"
            relation number(i32);
            relation even(i32);
            even(x) <-- number(x), if x % 2 == 0;
        "#,
        );

        assert_eq!(prog.rules.len(), 1);
        assert_eq!(prog.rules[0].body.len(), 2);

        // Second body item should be a condition
        assert!(matches!(prog.rules[0].body[1], BodyItem::Condition(_)));
    }

    #[test]
    fn test_lower_with_generator() {
        let prog = parse_and_lower(
            r#"
            relation number(i32);
            number(x) <-- for x in 0..10;
        "#,
        );

        assert_eq!(prog.rules.len(), 1);
        assert!(matches!(prog.rules[0].body[0], BodyItem::Generator(_)));
    }

    #[test]
    fn test_lower_disjunction_expands() {
        let prog = parse_and_lower(
            r#"
            relation a(i32);
            relation b(i32);
            relation c(i32);
            c(x) <-- (a(x) | b(x));
        "#,
        );

        // Disjunction should expand to 2 rules
        assert_eq!(prog.rules.len(), 2);
    }

    #[test]
    fn test_lower_aggregation() {
        let prog = parse_and_lower(
            r#"
            relation number(i32);
            relation total(i32);
            total(s) <-- agg s = sum(x) in number(x);
        "#,
        );

        assert_eq!(prog.rules.len(), 1);
        assert!(matches!(prog.rules[0].body[0], BodyItem::Aggregation(_)));
    }

    #[test]
    fn test_clause_args() {
        let prog = parse_and_lower(
            r#"
            relation edge(i32, i32);
            relation self_loop(i32);
            self_loop(x) <-- edge(x, x);
        "#,
        );

        // After desugaring, repeated var becomes var + condition
        if let BodyItem::Clause(cl) = &prog.rules[0].body[0] {
            // Should have a condition for equality check
            assert!(!cl.conditions.is_empty());
        }
    }
}
