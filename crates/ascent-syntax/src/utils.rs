//! General utility functions.

use std::collections::HashSet;
use std::hash::Hash;

use proc_macro2::{Ident, Span};
use syn::{Expr, Pat};

use crate::syn_utils::path_get_ident_mut;

/// Convert an iterator into a HashSet.
pub fn into_set<T: Eq + Hash>(iter: impl IntoIterator<Item = T>) -> HashSet<T> {
    iter.into_iter().collect()
}

/// Extract identifier from expression if it's a simple path.
pub fn expr_to_ident(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Path(p) => p.path.get_ident().cloned(),
        _ => None,
    }
}

/// Get mutable reference to identifier if expression is a simple path.
#[allow(dead_code)]
pub fn expr_to_ident_mut(expr: &mut Expr) -> Option<&mut Ident> {
    match expr {
        Expr::Path(p) => path_get_ident_mut(&mut p.path),
        _ => None,
    }
}

/// Extract identifier from pattern if it's a simple ident pattern.
pub fn pat_to_ident(pat: &Pat) -> Option<Ident> {
    match pat {
        Pat::Ident(ident) => Some(ident.ident.clone()),
        _ => None,
    }
}

/// Check if expression is a wildcard (`_`).
pub fn is_wild_card(expr: &Expr) -> bool {
    match expr {
        Expr::Infer(_) => true,
        Expr::Verbatim(ts) => ts.to_string() == "_",
        _ => false,
    }
}

/// Join multiple spans into one.
pub fn join_spans(spans: impl IntoIterator<Item = Span>) -> Span {
    let mut spans = spans.into_iter();
    let fst = spans.next().unwrap_or(Span::call_site());
    spans
        .try_fold(fst, |acc, next| acc.join(next))
        .unwrap_or(fst)
}

/// Check if two spans are equal (by debug representation).
#[allow(dead_code)]
pub fn spans_eq(span1: &Span, span2: &Span) -> bool {
    format!("{:?}", span1) == format!("{:?}", span2)
}
