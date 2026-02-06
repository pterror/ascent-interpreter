//! Core AST types and parsing for Ascent syntax.

use derive_syn_parse::Parse;
use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    Attribute, Expr, Generics, Ident, ImplGenerics, Pat, Result, Token, Type, TypeGenerics,
    Visibility, WhereClause, braced, parenthesized,
};

use crate::syn_utils::pattern_get_vars;
use crate::utils::{expr_to_ident, join_spans};

mod kw {
    syn::custom_keyword!(relation);
    syn::custom_keyword!(lattice);
    syn::custom_keyword!(agg);
    syn::custom_keyword!(ident);
    syn::custom_keyword!(expr);
}

/// The `<--` token used to separate rule heads from bodies.
#[derive(Parse, Clone)]
pub struct LongLeftArrow(Token![<], Token![-], Token![-]);

impl LongLeftArrow {
    pub fn span(&self) -> Span {
        join_spans([self.0.span, self.1.span, self.2.span])
    }
}

/// Type and impl signatures for the generated Ascent struct.
#[derive(Clone, Debug)]
pub struct Signatures {
    pub declaration: TypeSignature,
    pub implementation: Option<ImplSignature>,
}

impl Signatures {
    pub fn split_ty_generics_for_impl(
        &self,
    ) -> (ImplGenerics<'_>, TypeGenerics<'_>, Option<&'_ WhereClause>) {
        self.declaration.generics.split_for_impl()
    }

    pub fn split_impl_generics_for_impl(
        &self,
    ) -> (ImplGenerics<'_>, TypeGenerics<'_>, Option<&'_ WhereClause>) {
        let Some(signature) = &self.implementation else {
            return self.split_ty_generics_for_impl();
        };

        let (impl_generics, _, _) = signature.impl_generics.split_for_impl();
        let (_, ty_generics, where_clause) = signature.generics.split_for_impl();

        (impl_generics, ty_generics, where_clause)
    }
}

impl Parse for Signatures {
    fn parse(input: ParseStream) -> Result<Self> {
        let declaration = TypeSignature::parse(input)?;
        let implementation = if input.peek(Token![impl]) {
            Some(ImplSignature::parse(input)?)
        } else {
            None
        };
        Ok(Signatures {
            declaration,
            implementation,
        })
    }
}

/// Parse generics including where clause.
fn parse_generics_with_where_clause(input: ParseStream) -> Result<Generics> {
    let mut res = Generics::parse(input)?;
    if input.peek(Token![where]) {
        res.where_clause = Some(input.parse()?);
    }
    Ok(res)
}

/// Type signature: `struct Name<T> where T: Clone;`
#[derive(Clone, Parse, Debug)]
pub struct TypeSignature {
    #[call(Attribute::parse_outer)]
    pub attrs: Vec<Attribute>,
    pub visibility: Visibility,
    pub struct_kw: Token![struct],
    pub ident: Ident,
    #[call(parse_generics_with_where_clause)]
    pub generics: Generics,
    pub semi: Token![;],
}

/// Impl signature: `impl<T: Clone> Name<T>;`
#[derive(Clone, Parse, Debug)]
pub struct ImplSignature {
    pub impl_kw: Token![impl],
    pub impl_generics: Generics,
    pub ident: Ident,
    #[call(parse_generics_with_where_clause)]
    pub generics: Generics,
    pub semi: Token![;],
}

/// A relation or lattice declaration.
#[derive(Clone, PartialEq, Eq)]
pub struct RelationNode {
    pub attrs: Vec<Attribute>,
    pub name: Ident,
    pub field_types: Punctuated<Type, Token![,]>,
    pub initialization: Option<Expr>,
    pub is_lattice: bool,
}

impl Parse for RelationNode {
    fn parse(input: ParseStream) -> Result<Self> {
        let is_lattice = input.peek(kw::lattice);
        if is_lattice {
            input.parse::<kw::lattice>()?;
        } else {
            input.parse::<kw::relation>()?;
        }
        let name: Ident = input.parse()?;
        let content;
        parenthesized!(content in input);
        let field_types = content.parse_terminated(Type::parse, Token![,])?;
        let initialization = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Some(input.parse::<Expr>()?)
        } else {
            None
        };

        input.parse::<Token![;]>()?;
        if is_lattice && field_types.empty_or_trailing() {
            return Err(input.error("empty lattice is not allowed"));
        }
        Ok(RelationNode {
            attrs: vec![],
            name,
            field_types,
            is_lattice,
            initialization,
        })
    }
}

impl std::fmt::Debug for RelationNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelationNode")
            .field("name", &self.name.to_string())
            .field("is_lattice", &self.is_lattice)
            .field("field_types", &self.field_types.len())
            .finish()
    }
}

fn peek_macro_invocation(parse_stream: ParseStream) -> bool {
    parse_stream.peek(Ident) && parse_stream.peek2(Token![!])
}

fn peek_if_or_let(parse_stream: ParseStream) -> bool {
    parse_stream.peek(Token![if]) || parse_stream.peek(Token![let])
}

fn peek_expr_arg(_parse_stream: ParseStream) -> bool {
    true // fallback case
}

/// A body item in a rule (clause, generator, condition, etc.).
#[derive(Parse, Clone)]
pub enum BodyItemNode {
    #[peek(Token![for], name = "generative clause")]
    Generator(GeneratorNode),
    #[peek(kw::agg, name = "aggregate clause")]
    Agg(AggClauseNode),
    #[peek_with(peek_macro_invocation, name = "macro invocation")]
    MacroInvocation(syn::ExprMacro),
    #[peek(Ident, name = "body clause")]
    Clause(BodyClauseNode),
    #[peek(Token![!], name = "negation clause")]
    Negation(NegationClauseNode),
    #[peek(syn::token::Paren, name = "disjunction node")]
    Disjunction(DisjunctionNode),
    #[peek_with(peek_if_or_let, name = "if condition or let binding")]
    Cond(CondClause),
}

impl std::fmt::Debug for BodyItemNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generator(g) => f.debug_tuple("Generator").field(&g.pattern).finish(),
            Self::Agg(a) => f.debug_tuple("Agg").field(&a.rel.to_string()).finish(),
            Self::MacroInvocation(m) => {
                f.debug_tuple("MacroInvocation").field(&m.mac.path).finish()
            }
            Self::Clause(c) => f.debug_tuple("Clause").field(&c.rel.to_string()).finish(),
            Self::Negation(n) => f.debug_tuple("Negation").field(&n.rel.to_string()).finish(),
            Self::Disjunction(d) => f
                .debug_tuple("Disjunction")
                .field(&d.disjuncts.len())
                .finish(),
            Self::Cond(c) => f.debug_tuple("Cond").field(c).finish(),
        }
    }
}

/// Disjunction separator token.
#[derive(Parse, Clone)]
pub enum DisjunctionToken {
    #[peek(Token![||], name = "||")]
    OrOr(#[allow(dead_code)] Token![||]),
    #[peek(Token![|], name = "|")]
    Or(#[allow(dead_code)] Token![|]),
}

/// Disjunction: `(clause1 | clause2)`
#[derive(Clone)]
pub struct DisjunctionNode {
    pub paren: syn::token::Paren,
    pub disjuncts: Punctuated<Punctuated<BodyItemNode, Token![,]>, DisjunctionToken>,
}

impl Parse for DisjunctionNode {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        let paren = parenthesized!(content in input);
        let res: Punctuated<Punctuated<BodyItemNode, Token![,]>, DisjunctionToken> =
            Punctuated::<Punctuated<BodyItemNode, Token![,]>, DisjunctionToken>::parse_terminated_with(
                &content,
                Punctuated::<BodyItemNode, Token![,]>::parse_separated_nonempty,
            )?;
        if res
            .pairs()
            .any(|pair| matches!(pair.punct(), Some(DisjunctionToken::OrOr(_))))
        {
            eprintln!("WARNING: In Ascent rules, use `|` as the disjunction token instead of `||`")
        }
        Ok(DisjunctionNode {
            paren,
            disjuncts: res,
        })
    }
}

/// Generator: `for pattern in expr`
#[derive(Parse, Clone)]
pub struct GeneratorNode {
    pub for_keyword: Token![for],
    #[call(Pat::parse_multi)]
    pub pattern: Pat,
    pub in_keyword: Token![in],
    pub expr: Expr,
}

/// A clause in the rule body.
#[derive(Clone)]
pub struct BodyClauseNode {
    pub rel: Ident,
    pub args: Punctuated<BodyClauseArg, Token![,]>,
    pub cond_clauses: Vec<CondClause>,
}

impl Parse for BodyClauseNode {
    fn parse(input: ParseStream) -> Result<Self> {
        let rel: Ident = input.parse()?;
        let args_content;
        parenthesized!(args_content in input);
        let args = args_content.parse_terminated(BodyClauseArg::parse, Token![,])?;
        let mut cond_clauses = vec![];
        while let Ok(cl) = input.parse() {
            cond_clauses.push(cl);
        }
        Ok(BodyClauseNode {
            rel,
            args,
            cond_clauses,
        })
    }
}

/// Argument to a body clause: either an expression or a pattern with `?`.
#[derive(Parse, Clone, PartialEq, Eq, Debug)]
pub enum BodyClauseArg {
    #[peek(Token![?], name = "Pattern arg")]
    Pat(ClauseArgPattern),
    #[peek_with(peek_expr_arg, name = "Expression arg")]
    Expr(Expr),
}

impl BodyClauseArg {
    pub fn unwrap_expr(self) -> Expr {
        match self {
            Self::Expr(exp) => exp,
            Self::Pat(_) => panic!("unwrap_expr(): BodyClauseArg is not an expr"),
        }
    }

    pub fn unwrap_expr_ref(&self) -> &Expr {
        match self {
            Self::Expr(exp) => exp,
            Self::Pat(_) => panic!("unwrap_expr(): BodyClauseArg is not an expr"),
        }
    }

    pub fn get_vars(&self) -> Vec<Ident> {
        match self {
            BodyClauseArg::Pat(p) => pattern_get_vars(&p.pattern),
            BodyClauseArg::Expr(e) => expr_to_ident(e).into_iter().collect(),
        }
    }
}

impl ToTokens for BodyClauseArg {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            BodyClauseArg::Pat(pat) => {
                pat.huh_token.to_tokens(tokens);
                pat.pattern.to_tokens(tokens);
            }
            BodyClauseArg::Expr(exp) => exp.to_tokens(tokens),
        }
    }
}

/// Pattern argument with `?` prefix.
#[derive(Parse, Clone, PartialEq, Eq, Debug)]
pub struct ClauseArgPattern {
    pub huh_token: Token![?],
    #[call(Pat::parse_multi)]
    pub pattern: Pat,
}

/// If-let clause: `if let pattern = expr`
#[derive(Parse, Clone, PartialEq, Eq, Hash, Debug)]
pub struct IfLetClause {
    pub if_keyword: Token![if],
    pub let_keyword: Token![let],
    #[call(Pat::parse_multi)]
    pub pattern: Pat,
    pub eq_symbol: Token![=],
    pub exp: syn::Expr,
}

/// If clause: `if condition`
#[derive(Parse, Clone, PartialEq, Eq, Hash, Debug)]
pub struct IfClause {
    pub if_keyword: Token![if],
    pub cond: Expr,
}

/// Let clause: `let pattern = expr`
#[derive(Parse, Clone, PartialEq, Eq, Hash, Debug)]
pub struct LetClause {
    pub let_keyword: Token![let],
    #[call(Pat::parse_multi)]
    pub pattern: Pat,
    pub eq_symbol: Token![=],
    pub exp: syn::Expr,
}

/// Condition clause variants.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CondClause {
    IfLet(IfLetClause),
    If(IfClause),
    Let(LetClause),
}

impl CondClause {
    pub fn bound_vars(&self) -> Vec<Ident> {
        match self {
            CondClause::IfLet(cl) => pattern_get_vars(&cl.pattern),
            CondClause::If(_) => vec![],
            CondClause::Let(cl) => pattern_get_vars(&cl.pattern),
        }
    }

    pub fn expr(&self) -> &Expr {
        match self {
            CondClause::IfLet(cl) => &cl.exp,
            CondClause::If(cl) => &cl.cond,
            CondClause::Let(cl) => &cl.exp,
        }
    }
}

impl Parse for CondClause {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![if]) {
            if input.peek2(Token![let]) {
                let cl: IfLetClause = input.parse()?;
                Ok(Self::IfLet(cl))
            } else {
                let cl: IfClause = input.parse()?;
                Ok(Self::If(cl))
            }
        } else if input.peek(Token![let]) {
            let cl: LetClause = input.parse()?;
            Ok(Self::Let(cl))
        } else {
            Err(input.error("expected either if clause or if let clause"))
        }
    }
}

/// Negation clause: `!rel(args)`
#[derive(Parse, Clone)]
pub struct NegationClauseNode {
    pub neg_token: Token![!],
    pub rel: Ident,
    #[paren]
    pub rel_arg_paren: syn::token::Paren,
    #[inside(rel_arg_paren)]
    #[call(Punctuated::parse_terminated)]
    pub args: Punctuated<Expr, Token![,]>,
}

/// Head item: either a clause or a macro invocation.
#[derive(Clone, Parse)]
pub enum HeadItemNode {
    #[peek_with(peek_macro_invocation, name = "macro invocation")]
    MacroInvocation(syn::ExprMacro),
    #[peek(Ident, name = "head clause")]
    HeadClause(HeadClauseNode),
}

impl HeadItemNode {
    pub fn clause(&self) -> &HeadClauseNode {
        match self {
            HeadItemNode::HeadClause(cl) => cl,
            HeadItemNode::MacroInvocation(_) => panic!("unexpected macro invocation"),
        }
    }
}

impl std::fmt::Debug for HeadItemNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MacroInvocation(m) => {
                f.debug_tuple("MacroInvocation").field(&m.mac.path).finish()
            }
            Self::HeadClause(c) => f
                .debug_tuple("HeadClause")
                .field(&c.rel.to_string())
                .finish(),
        }
    }
}

/// Head clause: `rel(args)`
#[derive(Clone)]
pub struct HeadClauseNode {
    pub rel: Ident,
    pub args: Punctuated<Expr, Token![,]>,
}

impl Parse for HeadClauseNode {
    fn parse(input: ParseStream) -> Result<Self> {
        let rel: Ident = input.parse()?;
        let args_content;
        parenthesized!(args_content in input);
        let args = args_content.parse_terminated(Expr::parse, Token![,])?;
        Ok(HeadClauseNode { rel, args })
    }
}

impl ToTokens for HeadClauseNode {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        self.rel.to_tokens(tokens);
        self.args.to_tokens(tokens);
    }
}

impl std::fmt::Debug for HeadClauseNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadClauseNode")
            .field("rel", &self.rel.to_string())
            .field("args", &self.args.len())
            .finish()
    }
}

/// Aggregation clause: `agg pat = aggregator(bound) in rel(args)`
#[derive(Parse, Clone)]
pub struct AggClauseNode {
    pub agg_kw: kw::agg,
    #[call(Pat::parse_multi)]
    pub pat: Pat,
    pub eq_token: Token![=],
    pub aggregator: AggregatorNode,
    #[paren]
    pub agg_arg_paren: syn::token::Paren,
    #[inside(agg_arg_paren)]
    #[call(Punctuated::parse_terminated)]
    pub bound_args: Punctuated<Ident, Token![,]>,
    pub in_kw: Token![in],
    pub rel: Ident,
    #[paren]
    pub rel_arg_paren: syn::token::Paren,
    #[inside(rel_arg_paren)]
    #[call(Punctuated::parse_terminated)]
    pub rel_args: Punctuated<Expr, Token![,]>,
}

/// Aggregator: either a path or a parenthesized expression.
#[derive(Clone)]
pub enum AggregatorNode {
    Path(syn::Path),
    Expr(Expr),
}

impl Parse for AggregatorNode {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(syn::token::Paren) {
            let inside_parens;
            parenthesized!(inside_parens in input);
            Ok(AggregatorNode::Expr(inside_parens.parse()?))
        } else {
            Ok(AggregatorNode::Path(input.parse()?))
        }
    }
}

impl std::fmt::Debug for AggregatorNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(p) => f.debug_tuple("Path").field(p).finish(),
            Self::Expr(e) => f.debug_tuple("Expr").field(e).finish(),
        }
    }
}

/// A rule: `head <-- body;` or `head;` (for facts).
pub struct RuleNode {
    pub head_clauses: Punctuated<HeadItemNode, Token![,]>,
    pub body_items: Vec<BodyItemNode>,
}

impl Parse for RuleNode {
    fn parse(input: ParseStream) -> Result<Self> {
        let head_clauses = if input.peek(syn::token::Brace) {
            let content;
            braced!(content in input);
            Punctuated::<HeadItemNode, Token![,]>::parse_terminated(&content)?
        } else {
            Punctuated::<HeadItemNode, Token![,]>::parse_separated_nonempty(input)?
        };

        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            Ok(RuleNode {
                head_clauses,
                body_items: vec![],
            })
        } else {
            input.parse::<LongLeftArrow>()?;
            let body_items =
                Punctuated::<BodyItemNode, Token![,]>::parse_separated_nonempty(input)?;
            input.parse::<Token![;]>()?;
            Ok(RuleNode {
                head_clauses,
                body_items: body_items.into_iter().collect(),
            })
        }
    }
}

impl std::fmt::Debug for RuleNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleNode")
            .field("head_clauses", &self.head_clauses.len())
            .field("body_items", &self.body_items)
            .finish()
    }
}

/// Macro definition parameter.
#[derive(Parse, Clone)]
pub struct MacroDefParam {
    pub dollar: Token![$],
    pub name: Ident,
    pub colon: Token![:],
    pub kind: MacroParamKind,
}

/// Macro parameter kind.
#[derive(Parse, Clone)]
pub enum MacroParamKind {
    #[peek(kw::ident, name = "ident")]
    Expr(Ident),
    #[peek(kw::expr, name = "expr")]
    Ident(Ident),
}

/// Macro definition: `macro name($param: kind) { body }`
#[derive(Parse, Clone)]
pub struct MacroDefNode {
    pub mac: Token![macro],
    pub name: Ident,
    #[paren]
    pub arg_paren: syn::token::Paren,
    #[inside(arg_paren)]
    #[call(Punctuated::parse_terminated)]
    pub params: Punctuated<MacroDefParam, Token![,]>,
    #[brace]
    pub body_brace: syn::token::Brace,
    #[inside(body_brace)]
    pub body: TokenStream,
}

impl std::fmt::Debug for MacroDefNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacroDefNode")
            .field("name", &self.name.to_string())
            .finish()
    }
}

/// A complete Ascent program.
#[derive(Debug)]
pub struct AscentProgram {
    pub attributes: Vec<syn::Attribute>,
    pub signatures: Option<Signatures>,
    pub relations: Vec<RelationNode>,
    pub rules: Vec<RuleNode>,
    pub macros: Vec<MacroDefNode>,
}

impl Parse for AscentProgram {
    fn parse(input: ParseStream) -> Result<Self> {
        let attributes = Attribute::parse_inner(input)?;
        let mut struct_attrs = Attribute::parse_outer(input)?;
        let signatures = if input.peek(Token![pub]) || input.peek(Token![struct]) {
            let mut signatures = Signatures::parse(input)?;
            signatures.declaration.attrs = std::mem::take(&mut struct_attrs);
            Some(signatures)
        } else {
            None
        };

        let mut rules = vec![];
        let mut relations = vec![];
        let mut macros = vec![];

        while !input.is_empty() {
            let attrs = if !struct_attrs.is_empty() {
                std::mem::take(&mut struct_attrs)
            } else {
                Attribute::parse_outer(input)?
            };

            if input.peek(kw::relation) || input.peek(kw::lattice) {
                let mut relation_node = RelationNode::parse(input)?;
                relation_node.attrs = attrs;
                relations.push(relation_node);
            } else if input.peek(Token![macro]) {
                if !attrs.is_empty() {
                    return Err(syn::Error::new(attrs[0].span(), "unexpected attribute(s)"));
                }
                macros.push(MacroDefNode::parse(input)?);
            } else {
                if !attrs.is_empty() {
                    return Err(syn::Error::new(attrs[0].span(), "unexpected attribute(s)"));
                }
                rules.push(RuleNode::parse(input)?);
            }
        }

        Ok(AscentProgram {
            attributes,
            signatures,
            relations,
            rules,
            macros,
        })
    }
}

/// Identity of a relation (name + types + lattice flag).
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct RelationIdentity {
    pub name: Ident,
    pub field_types: Vec<Type>,
    pub is_lattice: bool,
}

impl From<&RelationNode> for RelationIdentity {
    fn from(relation_node: &RelationNode) -> Self {
        RelationIdentity {
            name: relation_node.name.clone(),
            field_types: relation_node.field_types.iter().cloned().collect(),
            is_lattice: relation_node.is_lattice,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_program() {
        let input = r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.relations.len(), 2);
        assert_eq!(program.rules.len(), 2);
    }

    #[test]
    fn test_parse_lattice() {
        let input = r#"
            lattice shortest(i32, i32, Dual<u32>);
            shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.relations.len(), 1);
        assert!(program.relations[0].is_lattice);
    }

    #[test]
    fn test_parse_aggregation() {
        let input = r#"
            relation number(i32);
            relation lowest(i32);
            lowest(y) <-- agg y = min(x) in number(x);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
    }

    #[test]
    fn test_parse_negation() {
        let input = r#"
            relation a(i32);
            relation b(i32);
            a(x) <-- b(x), !a(x);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
    }

    #[test]
    fn test_parse_generator() {
        let input = r#"
            relation number(i32);
            number(x) <-- for x in 0..10;
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
    }

    #[test]
    fn test_parse_conditions() {
        let input = r#"
            relation number(i32);
            relation even(i32);
            even(x) <-- number(x), if x % 2 == 0;
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
    }

    #[test]
    fn test_parse_disjunction() {
        let input = r#"
            relation a(i32);
            relation b(i32);
            relation c(i32);
            c(x) <-- (a(x) | b(x));
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
    }

    #[test]
    fn test_parse_fact() {
        let input = r#"
            relation edge(i32, i32);
            edge(1, 2);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.rules.len(), 1);
        assert!(program.rules[0].body_items.is_empty());
    }

    #[test]
    fn test_parse_macro() {
        let input = r#"
            relation shared(i32);
            macro wrap($x: expr) { shared($x) }
            wrap!(42);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert_eq!(program.macros.len(), 1);
    }

    #[test]
    fn test_parse_with_struct() {
        let input = r#"
            struct MyProgram;
            relation edge(i32, i32);
        "#;

        let program: AscentProgram = syn::parse_str(input).unwrap();
        assert!(program.signatures.is_some());
        assert_eq!(
            program.signatures.unwrap().declaration.ident.to_string(),
            "MyProgram"
        );
    }
}
