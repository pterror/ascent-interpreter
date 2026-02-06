//! Desugaring passes for Ascent syntax.
//!
//! Transforms syntactic sugar into normalized forms:
//! - Disjunctions → multiple rules
//! - Pattern arguments → if-let conditions
//! - Repeated variables → equality checks
//! - Wildcards → fresh variables
//! - Negation → aggregation (optional, for interpreter use)

use std::collections::HashMap;

use proc_macro2::{Ident, Span};
use quote::{quote, quote_spanned};
use syn::parse2;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;

use crate::syn_utils::{expr_get_vars, pattern_get_vars};
use crate::syntax::{
    AggClauseNode, AscentProgram, BodyClauseArg, BodyClauseNode, BodyItemNode, CondClause, RuleNode,
};
use crate::utils::{expr_to_ident, is_wild_card};

/// Generator for unique identifiers.
#[derive(Clone)]
pub struct GenSym {
    counters: HashMap<String, u32>,
    transformer: fn(&str) -> String,
}

impl Default for GenSym {
    fn default() -> Self {
        Self {
            counters: HashMap::new(),
            transformer: |s| format!("{}_", s),
        }
    }
}

impl GenSym {
    pub fn new(transformer: fn(&str) -> String) -> Self {
        Self {
            counters: HashMap::new(),
            transformer,
        }
    }

    pub fn next(&mut self, prefix: &str) -> String {
        match self.counters.get_mut(prefix) {
            Some(n) => {
                *n += 1;
                format!("{}{}", (self.transformer)(prefix), *n - 1)
            }
            None => {
                self.counters.insert(prefix.into(), 1);
                (self.transformer)(prefix)
            }
        }
    }

    pub fn next_ident(&mut self, prefix: &str, span: Span) -> Ident {
        Ident::new(&self.next(prefix), span)
    }
}

/// Desugar an entire program.
pub fn desugar_program(mut prog: AscentProgram) -> AscentProgram {
    prog.rules = prog
        .rules
        .into_iter()
        .flat_map(desugar_disjunctions)
        .map(desugar_pattern_args)
        .map(desugar_wildcards)
        .map(desugar_repeated_vars)
        .collect();

    prog
}

/// Expand disjunctions into multiple rules.
///
/// `head <-- a, (b | c), d;` becomes:
/// - `head <-- a, b, d;`
/// - `head <-- a, c, d;`
pub fn desugar_disjunctions(rule: RuleNode) -> Vec<RuleNode> {
    fn bitem_desugar(bitem: &BodyItemNode) -> Vec<Vec<BodyItemNode>> {
        match bitem {
            BodyItemNode::Disjunction(d) => {
                let mut res = vec![];
                for disjunct in d.disjuncts.iter() {
                    for conjunction in bitems_desugar(&disjunct.iter().cloned().collect::<Vec<_>>())
                    {
                        res.push(conjunction);
                    }
                }
                res
            }
            _ => vec![vec![bitem.clone()]],
        }
    }

    fn bitems_desugar(bitems: &[BodyItemNode]) -> Vec<Vec<BodyItemNode>> {
        if bitems.is_empty() {
            return vec![vec![]];
        }

        let mut res = vec![];
        let sub_res = bitems_desugar(&bitems[0..bitems.len() - 1]);
        let last_desugared = bitem_desugar(&bitems[bitems.len() - 1]);

        for sub_res_item in sub_res {
            for last_item in &last_desugared {
                let mut res_item = sub_res_item.clone();
                res_item.extend(last_item.clone());
                res.push(res_item);
            }
        }

        res
    }

    bitems_desugar(&rule.body_items)
        .into_iter()
        .map(|body_items| RuleNode {
            body_items,
            head_clauses: rule.head_clauses.clone(),
        })
        .collect()
}

/// Replace pattern arguments with temporary variables and if-let conditions.
///
/// `rel(?Some(x))` becomes `rel(__arg), if let Some(x) = __arg`
pub fn desugar_pattern_args(rule: RuleNode) -> RuleNode {
    let mut gensym = GenSym::default();

    fn clause_desugar(body_clause: BodyClauseNode, gensym: &mut GenSym) -> BodyClauseNode {
        let mut new_args = Punctuated::new();
        let mut new_cond_clauses = vec![];

        for pair in body_clause.args.into_pairs() {
            let (arg, punc) = pair.into_tuple();
            let new_arg = match arg {
                BodyClauseArg::Expr(_) => arg,
                BodyClauseArg::Pat(pat) => {
                    let pattern = pat.pattern;
                    let ident = gensym.next_ident("__pat", pattern.span());
                    let new_cond_clause: CondClause =
                        parse2(quote! { if let #pattern = #ident }).unwrap();
                    new_cond_clauses.push(new_cond_clause);
                    BodyClauseArg::Expr(parse2(quote! { #ident }).unwrap())
                }
            };
            new_args.push_value(new_arg);
            if let Some(punc) = punc {
                new_args.push_punct(punc);
            }
        }

        new_cond_clauses.extend(body_clause.cond_clauses);
        BodyClauseNode {
            args: new_args,
            cond_clauses: new_cond_clauses,
            rel: body_clause.rel,
        }
    }

    RuleNode {
        body_items: rule
            .body_items
            .into_iter()
            .map(|bi| match bi {
                BodyItemNode::Clause(cl) => BodyItemNode::Clause(clause_desugar(cl, &mut gensym)),
                _ => bi,
            })
            .collect(),
        head_clauses: rule.head_clauses,
    }
}

/// Replace wildcards with fresh variables.
///
/// `rel(_, x, _)` becomes `rel(__wild0, x, __wild1)`
pub fn desugar_wildcards(mut rule: RuleNode) -> RuleNode {
    let mut gensym = GenSym::default();
    gensym.next("_"); // Skip past "_" itself

    for bi in &mut rule.body_items {
        if let BodyItemNode::Clause(bcl) = bi {
            for arg in bcl.args.iter_mut() {
                if let BodyClauseArg::Expr(expr) = arg
                    && is_wild_card(expr)
                {
                    let new_ident = gensym.next_ident("_", expr.span());
                    *expr = parse2(quote! { #new_ident }).unwrap();
                }
            }
        }
    }

    rule
}

/// Add equality checks for repeated variables in the same clause.
///
/// `rel(x, x)` becomes `rel(x, __x0), if x == __x0`
pub fn desugar_repeated_vars(mut rule: RuleNode) -> RuleNode {
    let mut grounded_vars = HashMap::<Ident, usize>::new();

    for i in 0..rule.body_items.len() {
        let bitem = &mut rule.body_items[i];
        match bitem {
            BodyItemNode::Clause(cl) => {
                let mut new_cond_clauses = vec![];

                for arg_ind in 0..cl.args.len() {
                    let expr = cl.args[arg_ind].unwrap_expr_ref();
                    let expr_has_vars_from_same_clause = expr_get_vars(expr).iter().any(|var| {
                        if let Some(cl_ind) = grounded_vars.get(var) {
                            *cl_ind == i
                        } else {
                            false
                        }
                    });

                    if expr_has_vars_from_same_clause {
                        let new_ident = fresh_ident(
                            &expr_to_ident(expr)
                                .map(|e| e.to_string())
                                .unwrap_or_else(|| "expr".to_string()),
                            expr.span(),
                        );
                        let check: CondClause =
                            parse2(quote_spanned! {expr.span()=> if #new_ident == #expr }).unwrap();
                        new_cond_clauses.push(check);
                        cl.args[arg_ind] =
                            BodyClauseArg::Expr(parse2(quote! { #new_ident }).unwrap());
                    } else if let Some(ident) = expr_to_ident(expr) {
                        grounded_vars.entry(ident).or_insert(i);
                    }
                }

                for new_cond in new_cond_clauses.into_iter().rev() {
                    cl.cond_clauses.insert(0, new_cond);
                }
            }
            BodyItemNode::Generator(generator) => {
                for ident in pattern_get_vars(&generator.pattern) {
                    grounded_vars.entry(ident).or_insert(i);
                }
            }
            BodyItemNode::Cond(cond_cl @ CondClause::IfLet(_))
            | BodyItemNode::Cond(cond_cl @ CondClause::Let(_)) => {
                for ident in cond_cl.bound_vars() {
                    grounded_vars.entry(ident).or_insert(i);
                }
            }
            BodyItemNode::Cond(CondClause::If(_)) => {}
            BodyItemNode::Agg(agg) => {
                for ident in pattern_get_vars(&agg.pat) {
                    grounded_vars.entry(ident).or_insert(i);
                }
            }
            BodyItemNode::Negation(_) => {}
            BodyItemNode::Disjunction(_) => {
                panic!("disjunctions should be desugared before repeated vars")
            }
            BodyItemNode::MacroInvocation(m) => {
                panic!("macro invocations should be expanded: {:?}", m.mac.path)
            }
        }
    }

    rule
}

/// Convert negation to aggregation.
///
/// `!rel(x)` becomes `agg () = not() in rel(x)`
///
/// Note: This requires the `ascent::aggregators::not` aggregator at runtime.
pub fn desugar_negation(mut rule: RuleNode) -> RuleNode {
    for bi in &mut rule.body_items {
        if let BodyItemNode::Negation(neg) = bi {
            let rel = &neg.rel;
            let args = &neg.args;
            let span = neg.neg_token.span;
            let replacement: AggClauseNode = parse2(quote_spanned! {span=>
                agg () = ::ascent::aggregators::not() in #rel(#args)
            })
            .unwrap();
            *bi = BodyItemNode::Agg(replacement);
        }
    }
    rule
}

use std::sync::atomic::{AtomicU32, Ordering};

static IDENT_COUNTER: AtomicU32 = AtomicU32::new(0);

fn fresh_ident(prefix: &str, span: Span) -> Ident {
    let counter = IDENT_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ident::new(&format!("__{prefix}_{counter}"), span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_program(input: &str) -> AscentProgram {
        syn::parse_str(input).unwrap()
    }

    #[test]
    fn test_desugar_disjunction() {
        let prog = parse_program(
            r#"
            relation a(i32);
            relation b(i32);
            relation c(i32);
            c(x) <-- (a(x) | b(x));
        "#,
        );

        let desugared = desugar_program(prog);
        assert_eq!(desugared.rules.len(), 2);
    }

    #[test]
    fn test_desugar_nested_disjunction() {
        let prog = parse_program(
            r#"
            relation a(i32);
            relation b(i32);
            relation c(i32);
            relation d(i32);
            d(x) <-- (a(x) | b(x)), (b(x) | c(x));
        "#,
        );

        let desugared = desugar_program(prog);
        // (a|b) × (b|c) = 4 combinations
        assert_eq!(desugared.rules.len(), 4);
    }

    #[test]
    fn test_desugar_wildcards() {
        let prog = parse_program(
            r#"
            relation edge(i32, i32);
            relation node(i32);
            node(x) <-- edge(x, _);
        "#,
        );

        let desugared = desugar_program(prog);
        assert_eq!(desugared.rules.len(), 1);

        // The wildcard should be replaced with a fresh variable
        if let BodyItemNode::Clause(cl) = &desugared.rules[0].body_items[0] {
            assert_eq!(cl.args.len(), 2);
            // Second arg should not be a wildcard anymore
            if let BodyClauseArg::Expr(e) = &cl.args[1] {
                assert!(!is_wild_card(e));
            }
        }
    }

    #[test]
    fn test_desugar_repeated_vars() {
        let prog = parse_program(
            r#"
            relation rel(i32, i32);
            relation same(i32);
            same(x) <-- rel(x, x);
        "#,
        );

        let desugared = desugar_program(prog);
        assert_eq!(desugared.rules.len(), 1);

        // Should have an equality condition added
        if let BodyItemNode::Clause(cl) = &desugared.rules[0].body_items[0] {
            assert!(!cl.cond_clauses.is_empty());
        }
    }
}
