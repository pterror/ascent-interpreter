//! Utilities for working with syn AST nodes.

use std::collections::HashSet;

use proc_macro2::Ident;
use syn::{Expr, Pat, Path};

use crate::utils::into_set;

/// Extract all variables bound by a pattern.
pub fn pattern_get_vars(pat: &Pat) -> Vec<Ident> {
    let mut res = vec![];
    match pat {
        Pat::Ident(pat_ident) => {
            res.push(pat_ident.ident.clone());
            if let Some(subpat) = &pat_ident.subpat {
                res.extend(pattern_get_vars(&subpat.1))
            }
        }
        Pat::Lit(_) => {}
        Pat::Macro(_) => {}
        Pat::Or(or_pat) => {
            let cases_vars = or_pat.cases.iter().map(pattern_get_vars).map(into_set);
            let intersection = cases_vars
                .reduce(|case_vars, accu| case_vars.intersection(&accu).cloned().collect());
            if let Some(intersection) = intersection {
                res.extend(intersection);
            }
        }
        Pat::Path(_) => {}
        Pat::Range(_) => {}
        Pat::Reference(ref_pat) => res.extend(pattern_get_vars(&ref_pat.pat)),
        Pat::Rest(_) => {}
        Pat::Slice(slice_pat) => {
            for sub_pat in slice_pat.elems.iter() {
                res.extend(pattern_get_vars(sub_pat));
            }
        }
        Pat::Struct(struct_pat) => {
            for field_pat in struct_pat.fields.iter() {
                res.extend(pattern_get_vars(&field_pat.pat));
            }
        }
        Pat::Tuple(tuple_pat) => {
            for elem_pat in tuple_pat.elems.iter() {
                res.extend(pattern_get_vars(elem_pat));
            }
        }
        Pat::TupleStruct(tuple_struct_pat) => {
            for elem_pat in tuple_struct_pat.elems.iter() {
                res.extend(pattern_get_vars(elem_pat));
            }
        }
        Pat::Type(type_pat) => {
            res.extend(pattern_get_vars(&type_pat.pat));
        }
        Pat::Verbatim(_) => {}
        Pat::Wild(_) => {}
        _ => {}
    }
    res
}

/// Visit all variables in a pattern, allowing mutation.
pub fn pattern_visit_vars_mut(pat: &mut Pat, visitor: &mut dyn FnMut(&mut Ident)) {
    match pat {
        Pat::Ident(pat_ident) => {
            visitor(&mut pat_ident.ident);
            if let Some(subpat) = &mut pat_ident.subpat {
                pattern_visit_vars_mut(&mut subpat.1, visitor);
            }
        }
        Pat::Lit(_) => {}
        Pat::Macro(_) => {}
        Pat::Or(or_pat) => {
            for case in or_pat.cases.iter_mut() {
                pattern_visit_vars_mut(case, visitor)
            }
        }
        Pat::Path(_) => {}
        Pat::Range(_) => {}
        Pat::Reference(ref_pat) => pattern_visit_vars_mut(&mut ref_pat.pat, visitor),
        Pat::Rest(_) => {}
        Pat::Slice(slice_pat) => {
            for sub_pat in slice_pat.elems.iter_mut() {
                pattern_visit_vars_mut(sub_pat, visitor);
            }
        }
        Pat::Struct(struct_pat) => {
            for field_pat in struct_pat.fields.iter_mut() {
                pattern_visit_vars_mut(&mut field_pat.pat, visitor);
            }
        }
        Pat::Tuple(tuple_pat) => {
            for elem_pat in tuple_pat.elems.iter_mut() {
                pattern_visit_vars_mut(elem_pat, visitor);
            }
        }
        Pat::TupleStruct(tuple_struct_pat) => {
            for elem_pat in tuple_struct_pat.elems.iter_mut() {
                pattern_visit_vars_mut(elem_pat, visitor);
            }
        }
        Pat::Type(type_pat) => {
            pattern_visit_vars_mut(&mut type_pat.pat, visitor);
        }
        Pat::Verbatim(_) => {}
        Pat::Wild(_) => {}
        _ => {}
    }
}

/// If the expression is a let expression, return its bound variables.
pub fn expr_get_let_bound_vars(expr: &Expr) -> Vec<Ident> {
    match expr {
        Expr::Let(l) => pattern_get_vars(&l.pat),
        _ => vec![],
    }
}

/// Visit free variables in an expression.
pub fn expr_visit_free_vars(expr: &Expr, visitor: &mut dyn FnMut(&Ident)) {
    macro_rules! visit {
        ($e:expr) => {
            expr_visit_free_vars($e, visitor)
        };
    }
    macro_rules! visitor_except {
        ($excluded:expr) => {
            &mut |ident| {
                if !$excluded.contains(ident) {
                    visitor(ident)
                }
            }
        };
    }
    macro_rules! visit_except {
        ($e:expr, $excluded:expr) => {
            expr_visit_free_vars($e, visitor_except!($excluded))
        };
    }

    match expr {
        Expr::Array(arr) => {
            for elem in arr.elems.iter() {
                visit!(elem);
            }
        }
        Expr::Assign(assign) => {
            visit!(&assign.left);
            visit!(&assign.right);
        }
        Expr::Async(a) => block_visit_free_vars(&a.block, visitor),
        Expr::Await(a) => visit!(&a.base),
        Expr::Binary(b) => {
            visit!(&b.left);
            visit!(&b.right);
        }
        Expr::Block(b) => block_visit_free_vars(&b.block, visitor),
        Expr::Break(b) => {
            if let Some(e) = &b.expr {
                visit!(e);
            }
        }
        Expr::Call(c) => {
            visit!(&c.func);
            for arg in c.args.iter() {
                visit!(arg);
            }
        }
        Expr::Cast(c) => visit!(&c.expr),
        Expr::Closure(c) => {
            let input_vars: HashSet<_> = c.inputs.iter().flat_map(pattern_get_vars).collect();
            visit_except!(&c.body, input_vars);
        }
        Expr::Continue(_) => {}
        Expr::Field(f) => visit!(&f.base),
        Expr::ForLoop(f) => {
            let pat_vars: HashSet<_> = pattern_get_vars(&f.pat).into_iter().collect();
            visit!(&f.expr);
            block_visit_free_vars(&f.body, visitor_except!(pat_vars));
        }
        Expr::Group(g) => visit!(&g.expr),
        Expr::If(e) => {
            let bound_vars = expr_get_let_bound_vars(&e.cond)
                .into_iter()
                .collect::<HashSet<_>>();
            visit!(&e.cond);
            block_visit_free_vars(&e.then_branch, visitor_except!(bound_vars));
            if let Some(eb) = &e.else_branch {
                visit!(&eb.1);
            }
        }
        Expr::Index(i) => {
            visit!(&i.expr);
            visit!(&i.index);
        }
        Expr::Let(l) => visit!(&l.expr),
        Expr::Lit(_) => {}
        Expr::Loop(l) => block_visit_free_vars(&l.body, visitor),
        Expr::Macro(_) => {
            // Cannot determine free variables in macros
        }
        Expr::Match(m) => {
            visit!(&m.expr);
            for arm in m.arms.iter() {
                if let Some(g) = &arm.guard {
                    visit!(&g.1);
                }
                let arm_vars = pattern_get_vars(&arm.pat)
                    .into_iter()
                    .collect::<HashSet<_>>();
                visit_except!(&arm.body, arm_vars);
            }
        }
        Expr::MethodCall(c) => {
            visit!(&c.receiver);
            for arg in c.args.iter() {
                visit!(arg);
            }
        }
        Expr::Paren(p) => visit!(&p.expr),
        Expr::Path(p) => {
            if let Some(ident) = p.path.get_ident() {
                visitor(ident);
            }
        }
        Expr::Range(r) => {
            if let Some(start) = &r.start {
                visit!(start);
            }
            if let Some(end) = &r.end {
                visit!(end);
            }
        }
        Expr::Reference(r) => visit!(&r.expr),
        Expr::Repeat(r) => {
            visit!(&r.expr);
            visit!(&r.len);
        }
        Expr::Return(r) => {
            if let Some(e) = &r.expr {
                visit!(e);
            }
        }
        Expr::Struct(s) => {
            for f in s.fields.iter() {
                visit!(&f.expr);
            }
            if let Some(rest) = &s.rest {
                visit!(rest);
            }
        }
        Expr::Try(t) => visit!(&t.expr),
        Expr::TryBlock(t) => block_visit_free_vars(&t.block, visitor),
        Expr::Tuple(t) => {
            for e in t.elems.iter() {
                visit!(e);
            }
        }
        Expr::Unary(u) => visit!(&u.expr),
        Expr::Unsafe(u) => block_visit_free_vars(&u.block, visitor),
        Expr::Verbatim(_) => {}
        Expr::While(w) => {
            let bound_vars = expr_get_let_bound_vars(&w.cond)
                .into_iter()
                .collect::<HashSet<_>>();
            visit!(&w.cond);
            block_visit_free_vars(&w.body, visitor_except!(bound_vars));
        }
        Expr::Yield(y) => {
            if let Some(e) = &y.expr {
                visit!(e);
            }
        }
        _ => {}
    }
}

/// Visit free variables in a block.
pub fn block_visit_free_vars(block: &syn::Block, visitor: &mut dyn FnMut(&Ident)) {
    let mut bound_vars = HashSet::new();
    for stmt in block.stmts.iter() {
        let (stmt_bound_vars, _) = stmt_get_vars(stmt);
        stmt_visit_free_vars(stmt, &mut |ident| {
            if !bound_vars.contains(ident) {
                visitor(ident)
            }
        });
        bound_vars.extend(stmt_bound_vars);
    }
}

/// Get bound and used variables from a statement.
fn stmt_get_vars(stmt: &syn::Stmt) -> (Vec<Ident>, Vec<Ident>) {
    let mut bound_vars = vec![];
    let mut used_vars = vec![];
    match stmt {
        syn::Stmt::Local(l) => {
            bound_vars.extend(pattern_get_vars(&l.pat));
            if let Some(init) = &l.init {
                used_vars.extend(expr_get_vars(&init.expr));
                if let Some(diverge) = &init.diverge {
                    used_vars.extend(expr_get_let_bound_vars(&diverge.1));
                }
            }
        }
        syn::Stmt::Item(_) => {}
        syn::Stmt::Expr(e, _) => used_vars.extend(expr_get_vars(e)),
        syn::Stmt::Macro(_) => {}
    }
    (bound_vars, used_vars)
}

/// Visit free variables in a statement.
fn stmt_visit_free_vars(stmt: &syn::Stmt, visitor: &mut dyn FnMut(&Ident)) {
    match stmt {
        syn::Stmt::Local(l) => {
            if let Some(init) = &l.init {
                expr_visit_free_vars(&init.expr, visitor);
                if let Some(diverge) = &init.diverge {
                    expr_visit_free_vars(&diverge.1, visitor);
                }
            }
        }
        syn::Stmt::Item(_) => {}
        syn::Stmt::Expr(e, _) => expr_visit_free_vars(e, visitor),
        syn::Stmt::Macro(_) => {}
    }
}

/// Get all free variables in an expression.
pub fn expr_get_vars(expr: &Expr) -> Vec<Ident> {
    let mut res = vec![];
    expr_visit_free_vars(expr, &mut |ident| res.push(ident.clone()));
    res
}

/// Get mutable reference to ident if path is a single ident.
pub fn path_get_ident_mut(path: &mut Path) -> Option<&mut Ident> {
    if path.segments.len() != 1 || path.leading_colon.is_some() {
        return None;
    }
    let res = path.segments.first_mut()?;
    if res.arguments.is_empty() {
        Some(&mut res.ident)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_pattern_get_vars() {
        use syn::parse::Parser;

        let pattern = quote::quote! {
            SomePair(x, (y, z))
        };
        let pat = Pat::parse_single.parse2(pattern).unwrap();
        assert_eq!(
            ["x", "y", "z"]
                .iter()
                .map(ToString::to_string)
                .collect::<HashSet<_>>(),
            pattern_get_vars(&pat)
                .into_iter()
                .map(|id| id.to_string())
                .collect()
        );
    }

    #[test]
    fn test_expr_get_vars() {
        let expr: Expr = syn::parse_str("x + y * z").unwrap();
        let vars: HashSet<_> = expr_get_vars(&expr)
            .into_iter()
            .map(|i| i.to_string())
            .collect();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }
}
