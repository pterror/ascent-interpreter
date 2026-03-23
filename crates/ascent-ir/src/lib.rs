//! Intermediate representation for the Ascent interpreter.
//!
//! This IR is a simplified form of the AST, designed for efficient interpretation.
//! It removes syntactic sugar and normalizes the program structure.
//!
//! # Example
//!
//! ```no_run
//! use ascent_syntax::AscentProgram;
//! use ascent_ir::Program;
//!
//! let src = "relation edge(i32, i32); relation path(i32, i32); path(x,y) <-- edge(x,y);";
//! let ast: AscentProgram = syn::parse_str(src).unwrap();
//! let program = Program::from_ast(ast).unwrap();
//! assert_eq!(program.rules.len(), 1);
//! ```

use std::collections::HashMap;

use ascent_syntax::{
    AggClauseNode, AscentProgram, BodyClauseArg, BodyItemNode, CondClause, GeneratorNode,
    HeadClauseNode, HeadItemNode, RuleNode, desugar::desugar_program,
};
use quote::ToTokens;

// ─── IR-native types ────────────────────────────────────────────────

/// IR-native expression, replacing `syn::Expr` in public type signatures.
#[derive(Debug, Clone)]
pub enum IrExpr {
    Lit(IrLit),
    Var(String),
    Binary(IrBinOp, Box<IrExpr>, Box<IrExpr>),
    Unary(IrUnOp, Box<IrExpr>),
    Range {
        start: Box<IrExpr>,
        end: Box<IrExpr>,
        inclusive: bool,
    },
    Tuple(Vec<IrExpr>),
    Call(String, Vec<IrExpr>),
    MethodCall(Box<IrExpr>, String, Vec<IrExpr>),
    Cast(Box<IrExpr>, String),
    Array(Vec<IrExpr>),
    /// Opaque expression serialized as tokens. Re-parsed at eval time.
    Raw(String),
}

/// IR-native literal value.
#[derive(Debug, Clone)]
pub enum IrLit {
    Int(i128),
    Float(f64),
    Bool(bool),
    Char(char),
    String(String),
}

/// IR-native binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// IR-native unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrUnOp {
    Neg,
    Not,
    Deref,
}

/// IR-native type representation, replacing `syn::Type` in public type signatures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// A simple named type (e.g., `i32`, `String`).
    Named(String),
    /// A complex type serialized as a string (e.g., `Vec<i32>`).
    Complex(String),
}

/// IR-native pattern, replacing `syn::Pat` in public type signatures.
#[derive(Debug, Clone)]
pub enum IrPattern {
    Wild,
    Var(String, Option<Box<IrPattern>>),
    Lit(IrLit),
    Tuple(Vec<IrPattern>),
    TupleStruct(String, Vec<IrPattern>),
    Path(String),
    Or(Vec<IrPattern>),
    Ref(Box<IrPattern>),
}

/// IR-native attribute, replacing `syn::Attribute` in public type signatures.
#[derive(Debug, Clone)]
pub struct IrAttribute {
    pub name: String,
    pub args: Option<String>,
}

// ─── Conversion functions ───────────────────────────────────────────

/// Lower a `syn::Expr` to `IrExpr`.
pub fn lower_expr(expr: syn::Expr) -> IrExpr {
    match expr {
        syn::Expr::Lit(ref lit) => match lower_syn_lit(&lit.lit) {
            Some(ir_lit) => IrExpr::Lit(ir_lit),
            None => IrExpr::Raw(expr.to_token_stream().to_string()),
        },
        syn::Expr::Path(ref p) => {
            if let Some(ident) = p.path.get_ident() {
                let name = ident.to_string();
                match name.as_str() {
                    "true" => IrExpr::Lit(IrLit::Bool(true)),
                    "false" => IrExpr::Lit(IrLit::Bool(false)),
                    _ => IrExpr::Var(name),
                }
            } else {
                IrExpr::Raw(expr.to_token_stream().to_string())
            }
        }
        syn::Expr::Binary(bin) => match lower_binop(&bin.op) {
            Some(op) => {
                let left = lower_expr(*bin.left);
                let right = lower_expr(*bin.right);
                IrExpr::Binary(op, Box::new(left), Box::new(right))
            }
            None => IrExpr::Raw(syn::Expr::Binary(bin).to_token_stream().to_string()),
        },
        syn::Expr::Unary(u) => match lower_unop(&u.op) {
            Some(op) => IrExpr::Unary(op, Box::new(lower_expr(*u.expr))),
            None => IrExpr::Raw(syn::Expr::Unary(u).to_token_stream().to_string()),
        },
        syn::Expr::Paren(p) => lower_expr(*p.expr),
        syn::Expr::Reference(r) => lower_expr(*r.expr),
        syn::Expr::Range(r) => {
            if r.start.is_some() && r.end.is_some() {
                let start = Box::new(lower_expr(*r.start.unwrap()));
                let end = Box::new(lower_expr(*r.end.unwrap()));
                IrExpr::Range {
                    start,
                    end,
                    inclusive: matches!(r.limits, syn::RangeLimits::Closed(_)),
                }
            } else {
                // Half-open ranges (..5, 3.., ..): preserve the original expression
                IrExpr::Raw(syn::Expr::Range(r).to_token_stream().to_string())
            }
        }
        syn::Expr::Tuple(t) => IrExpr::Tuple(t.elems.into_iter().map(lower_expr).collect()),
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = &*call.func {
                let name = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let args: Vec<IrExpr> = call.args.into_iter().map(lower_expr).collect();
                IrExpr::Call(name, args)
            } else {
                IrExpr::Raw(syn::Expr::Call(call).to_token_stream().to_string())
            }
        }
        syn::Expr::MethodCall(mc) => {
            let receiver = lower_expr(*mc.receiver);
            let method = mc.method.to_string();
            let args: Vec<IrExpr> = mc.args.into_iter().map(lower_expr).collect();
            IrExpr::MethodCall(Box::new(receiver), method, args)
        }
        syn::Expr::Cast(cast) => {
            if let syn::Type::Path(tp) = &*cast.ty {
                let target = tp
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                IrExpr::Cast(Box::new(lower_expr(*cast.expr)), target)
            } else {
                IrExpr::Raw(syn::Expr::Cast(cast).to_token_stream().to_string())
            }
        }
        syn::Expr::Array(arr) => IrExpr::Array(arr.elems.into_iter().map(lower_expr).collect()),
        other => IrExpr::Raw(other.to_token_stream().to_string()),
    }
}

fn lower_syn_lit(lit: &syn::Lit) -> Option<IrLit> {
    match lit {
        syn::Lit::Int(i) => i.base10_parse::<i128>().ok().map(IrLit::Int),
        syn::Lit::Float(f) => f.base10_parse::<f64>().ok().map(IrLit::Float),
        syn::Lit::Bool(b) => Some(IrLit::Bool(b.value)),
        syn::Lit::Str(s) => Some(IrLit::String(s.value())),
        syn::Lit::Char(c) => Some(IrLit::Char(c.value())),
        _ => None,
    }
}

fn lower_binop(op: &syn::BinOp) -> Option<IrBinOp> {
    Some(match op {
        syn::BinOp::Add(_) => IrBinOp::Add,
        syn::BinOp::Sub(_) => IrBinOp::Sub,
        syn::BinOp::Mul(_) => IrBinOp::Mul,
        syn::BinOp::Div(_) => IrBinOp::Div,
        syn::BinOp::Rem(_) => IrBinOp::Rem,
        syn::BinOp::BitAnd(_) => IrBinOp::BitAnd,
        syn::BinOp::BitOr(_) => IrBinOp::BitOr,
        syn::BinOp::BitXor(_) => IrBinOp::BitXor,
        syn::BinOp::Shl(_) => IrBinOp::Shl,
        syn::BinOp::Shr(_) => IrBinOp::Shr,
        syn::BinOp::Eq(_) => IrBinOp::Eq,
        syn::BinOp::Ne(_) => IrBinOp::Ne,
        syn::BinOp::Lt(_) => IrBinOp::Lt,
        syn::BinOp::Le(_) => IrBinOp::Le,
        syn::BinOp::Gt(_) => IrBinOp::Gt,
        syn::BinOp::Ge(_) => IrBinOp::Ge,
        syn::BinOp::And(_) => IrBinOp::And,
        syn::BinOp::Or(_) => IrBinOp::Or,
        _ => return None,
    })
}

fn lower_unop(op: &syn::UnOp) -> Option<IrUnOp> {
    Some(match op {
        syn::UnOp::Neg(_) => IrUnOp::Neg,
        syn::UnOp::Not(_) => IrUnOp::Not,
        syn::UnOp::Deref(_) => IrUnOp::Deref,
        _ => return None,
    })
}

/// Lower a `syn::Type` to `IrType`.
pub fn lower_type(ty: syn::Type) -> IrType {
    if let syn::Type::Path(tp) = &ty
        && let Some(ident) = tp.path.get_ident()
    {
        return IrType::Named(ident.to_string());
    }
    IrType::Complex(ty.to_token_stream().to_string())
}

/// Lower a `syn::Pat` to `IrPattern`.
pub fn lower_pattern(pat: syn::Pat) -> IrPattern {
    match pat {
        syn::Pat::Wild(_) => IrPattern::Wild,
        syn::Pat::Ident(ident) => {
            let name = ident.ident.to_string();
            if name == "_" {
                return IrPattern::Wild;
            }
            let sub = ident
                .subpat
                .map(|(_, sub)| Box::new(lower_pattern(*sub)));
            IrPattern::Var(name, sub)
        }
        syn::Pat::Lit(lit) => {
            // PatLit is an alias for ExprLit in syn 2.0 — fields are `attrs` and `lit`
            if let Some(ir_lit) = lower_syn_lit(&lit.lit) {
                IrPattern::Lit(ir_lit)
            } else {
                IrPattern::Lit(IrLit::String(lit.lit.to_token_stream().to_string()))
            }
        }
        syn::Pat::Tuple(tuple) => {
            IrPattern::Tuple(tuple.elems.into_iter().map(lower_pattern).collect())
        }
        syn::Pat::TupleStruct(ts) => {
            let path_str = ts
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            let fields: Vec<IrPattern> = ts.elems.into_iter().map(lower_pattern).collect();
            IrPattern::TupleStruct(path_str, fields)
        }
        syn::Pat::Path(p) => {
            let path_str = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            IrPattern::Path(path_str)
        }
        syn::Pat::Or(or_pat) => {
            IrPattern::Or(or_pat.cases.into_iter().map(lower_pattern).collect())
        }
        syn::Pat::Reference(r) => IrPattern::Ref(Box::new(lower_pattern(*r.pat))),
        syn::Pat::Paren(p) => lower_pattern(*p.pat),
        // Unknown patterns — encode as a Path with the token string
        other => IrPattern::Path(other.to_token_stream().to_string()),
    }
}

/// Lower a `syn::Attribute` to `IrAttribute`.
pub fn lower_attribute(attr: syn::Attribute) -> IrAttribute {
    let name = attr
        .path()
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    let args = match &attr.meta {
        syn::Meta::Path(_) => None,
        syn::Meta::List(list) => Some(list.tokens.to_string()),
        syn::Meta::NameValue(nv) => Some(nv.value.to_token_stream().to_string()),
    };
    IrAttribute { name, args }
}

// ─── Public IR types ────────────────────────────────────────────────

/// A complete Ascent program in IR form.
#[derive(Debug, Clone)]
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
    /// Column types.
    pub column_types: Vec<IrType>,
    /// Whether this is a lattice.
    pub is_lattice: bool,
    /// Initial values (if any).
    pub initialization: Option<IrExpr>,
    /// Attributes on the relation declaration (e.g., `#[ds(btree)]`).
    pub attrs: Vec<IrAttribute>,
}

/// A rule: head <- body.
#[derive(Debug, Clone)]
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
    pub args: Vec<IrExpr>,
}

/// A body item in a rule.
#[derive(Debug, Clone)]
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
    Expr(IrExpr),
}

/// A generator: `for pattern in expr`.
#[derive(Debug, Clone)]
pub struct Generator {
    /// Variables to bind.
    pub vars: Vec<String>,
    /// Expression to iterate.
    pub expr: IrExpr,
    /// The full pattern (for complex patterns).
    pub pattern: IrPattern,
}

/// A filter condition.
#[derive(Debug, Clone)]
pub enum Condition {
    /// `if expr`
    If(IrExpr),
    /// `if let pattern = expr`
    IfLet { pattern: IrPattern, expr: IrExpr },
    /// `let pattern = expr`
    Let { pattern: IrPattern, expr: IrExpr },
}

/// An aggregation clause.
#[derive(Debug, Clone)]
pub struct Aggregation {
    /// Variables to bind from the aggregation result.
    pub result_vars: Vec<String>,
    /// The aggregator expression/path.
    pub aggregator: IrExpr,
    /// Variables to aggregate over.
    pub bound_vars: Vec<String>,
    /// Relation to aggregate from.
    pub relation: String,
    /// Arguments to the relation.
    pub args: Vec<IrExpr>,
}

// ─── from_ast implementations ───────────────────────────────────────

impl Program {
    /// Lower an AST program to IR.
    pub fn from_ast(ast: AscentProgram) -> Result<Self, String> {
        // First desugar
        let ast = desugar_program(ast);

        // Collect relations
        let mut relations = HashMap::new();
        for rel in ast.relations {
            relations.insert(
                rel.name.to_string(),
                Relation {
                    name: rel.name.to_string(),
                    column_types: rel.column_types.into_iter().map(lower_type).collect(),
                    is_lattice: rel.is_lattice,
                    initialization: rel.initialization.map(lower_expr),
                    attrs: rel.attrs.into_iter().map(lower_attribute).collect(),
                },
            );
        }

        // Lower rules
        let rules = ast
            .rules
            .into_iter()
            .map(Rule::from_ast)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Program { relations, rules })
    }
}

impl Rule {
    fn from_ast(rule: RuleNode) -> Result<Self, String> {
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
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Rule { heads, body })
    }
}

impl HeadClause {
    fn from_ast(hc: HeadClauseNode) -> Self {
        HeadClause {
            relation: hc.rel.to_string(),
            args: hc.args.into_iter().map(lower_expr).collect(),
        }
    }
}

impl BodyItem {
    fn from_ast(bi: BodyItemNode) -> Result<Self, String> {
        match bi {
            BodyItemNode::Clause(cl) => Ok(BodyItem::Clause(Clause::from_ast(cl)?)),
            BodyItemNode::Generator(generator) => {
                Ok(BodyItem::Generator(Generator::from_ast(generator)))
            }
            BodyItemNode::Cond(cond) => Ok(BodyItem::Condition(Condition::from_ast(cond))),
            BodyItemNode::Agg(agg) => Ok(BodyItem::Aggregation(Aggregation::from_ast(agg))),
            BodyItemNode::Negation(neg) => {
                // Lower negation directly to aggregation with "not"
                Ok(BodyItem::Aggregation(Aggregation {
                    result_vars: vec![],
                    aggregator: IrExpr::Var("not".to_string()),
                    bound_vars: vec![],
                    relation: neg.rel.to_string(),
                    args: neg.args.into_iter().map(lower_expr).collect(),
                }))
            }
            BodyItemNode::Disjunction(_) => {
                Err("disjunction should be desugared to multiple rules".to_string())
            }
            BodyItemNode::MacroInvocation(_) => {
                Err("macro invocations should be expanded".to_string())
            }
        }
    }
}

impl Clause {
    fn from_ast(cl: ascent_syntax::BodyClauseNode) -> Result<Self, String> {
        let args = cl
            .args
            .into_iter()
            .map(|arg| match arg {
                BodyClauseArg::Expr(e) => {
                    // Check if it's a simple variable
                    if let syn::Expr::Path(p) = &e
                        && let Some(ident) = p.path.get_ident()
                    {
                        return Ok(ClauseArg::Var(ident.to_string()));
                    }
                    Ok(ClauseArg::Expr(lower_expr(e)))
                }
                BodyClauseArg::Pat(_) => Err("pattern args should be desugared".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let conditions = cl
            .cond_clauses
            .into_iter()
            .map(Condition::from_ast)
            .collect();

        Ok(Clause {
            relation: cl.rel.to_string(),
            args,
            conditions,
        })
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
            expr: lower_expr(generator.expr),
            pattern: lower_pattern(generator.pattern),
        }
    }
}

impl Condition {
    fn from_ast(cond: CondClause) -> Self {
        match cond {
            CondClause::If(c) => Condition::If(lower_expr(c.cond)),
            CondClause::IfLet(c) => Condition::IfLet {
                pattern: lower_pattern(c.pattern),
                expr: lower_expr(c.expr),
            },
            CondClause::Let(c) => Condition::Let {
                pattern: lower_pattern(c.pattern),
                expr: lower_expr(c.expr),
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
            ascent_syntax::AggregatorNode::Path(p) => {
                let expr: syn::Expr = syn::parse2(quote::quote! { #p })
                    .expect("internal: aggregator path must parse as expression");
                lower_expr(expr)
            }
            ascent_syntax::AggregatorNode::Expr(e) => lower_expr(e),
        };

        Aggregation {
            result_vars,
            aggregator,
            bound_vars,
            relation: agg.rel.to_string(),
            args: agg.rel_args.into_iter().map(lower_expr).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_lower(input: &str) -> Program {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        Program::from_ast(ast).expect("lowering should succeed")
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

    #[test]
    fn test_attributes_preserved() {
        let prog = parse_and_lower(
            r#"
            #[ds(btree)]
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
        "#,
        );

        let edge = prog.relations.get("edge").unwrap();
        assert_eq!(edge.attrs.len(), 1);

        let path = prog.relations.get("path").unwrap();
        assert!(path.attrs.is_empty());
    }
}
