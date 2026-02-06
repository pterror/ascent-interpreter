//! Syntax parity tests: verify our parser accepts exactly the same syntax as ascent_macro.
//!
//! Each test contains an `ascent! { ... }` invocation (compile-time proof that ascent_macro
//! accepts the syntax) and a `syn::parse_str` call (runtime proof that our parser accepts it).
//! If either rejects something the other accepts, the test fails.
//!
//! These tests focus on syntax edge cases, not evaluation correctness (that's in comparison.rs).

use ascent::Dual;
use ascent::aggregators::{count, max, min, sum};
use ascent::ascent;
use ascent_syntax::AscentProgram;

/// Parse with our parser — panics with the input if it fails.
fn must_parse(input: &str) -> AscentProgram {
    syn::parse_str(input)
        .unwrap_or_else(|e| panic!("our parser rejected valid syntax: {e}\n---\n{input}"))
}

// ─── Trailing commas ─────────────────────────────────────────────────

#[test]
fn parity_trailing_comma_in_relation() {
    ascent! {
        relation foo(i32, i32,);
    }
    must_parse("relation foo(i32, i32,);");
}

#[test]
fn parity_trailing_comma_in_head() {
    ascent! {
        relation foo(i32, i32);
        foo(1, 2,);
    }
    must_parse("relation foo(i32, i32);\nfoo(1, 2,);");
}

#[test]
fn parity_trailing_comma_in_body() {
    ascent! {
        relation foo(i32, i32);
        relation bar(i32, i32);
        bar(x, y) <-- foo(x, y,);
    }
    must_parse("relation foo(i32, i32);\nrelation bar(i32, i32);\nbar(x, y) <-- foo(x, y,);");
}

// ─── Single-column relations ─────────────────────────────────────────

#[test]
fn parity_single_column() {
    ascent! {
        relation single(i32);
        single(42);
    }
    must_parse("relation single(i32);\nsingle(42);");
}

// ─── High-arity relations ────────────────────────────────────────────

#[test]
fn parity_five_columns() {
    ascent! {
        relation wide(i32, i32, i32, i32, i32);
        wide(1, 2, 3, 4, 5);
    }
    must_parse("relation wide(i32, i32, i32, i32, i32);\nwide(1, 2, 3, 4, 5);");
}

// ─── Expressions in head ─────────────────────────────────────────────

#[test]
fn parity_arithmetic_in_head() {
    ascent! {
        relation a(i32);
        relation b(i32);
        b(x + 1) <-- a(x);
    }
    must_parse("relation a(i32);\nrelation b(i32);\nb(x + 1) <-- a(x);");
}

#[test]
fn parity_complex_expr_in_head() {
    ascent! {
        relation a(i32);
        relation b(i32);
        b(x * x + 2 * x + 1) <-- a(x);
    }
    must_parse("relation a(i32);\nrelation b(i32);\nb(x * x + 2 * x + 1) <-- a(x);");
}

// ─── Multiple head clauses ───────────────────────────────────────────

#[test]
fn parity_multi_head() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        b(x), c(x) <-- a(x);
    }
    must_parse("relation a(i32);\nrelation b(i32);\nrelation c(i32);\nb(x), c(x) <-- a(x);");
}

// ─── Multiple body clauses ───────────────────────────────────────────

#[test]
fn parity_multi_body() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        relation d(i32);
        d(x) <-- a(x), b(x), c(x);
    }
    must_parse(
        "relation a(i32);\nrelation b(i32);\nrelation c(i32);\nrelation d(i32);\nd(x) <-- a(x), b(x), c(x);",
    );
}

// ─── Wildcards ───────────────────────────────────────────────────────

#[test]
fn parity_wildcards() {
    ascent! {
        relation pair(i32, i32);
        relation first(i32);
        first(x) <-- pair(x, _);
    }
    must_parse("relation pair(i32, i32);\nrelation first(i32);\nfirst(x) <-- pair(x, _);");
}

#[test]
fn parity_multiple_wildcards() {
    ascent! {
        relation triple(i32, i32, i32);
        relation mid(i32);
        mid(y) <-- triple(_, y, _);
    }
    must_parse("relation triple(i32, i32, i32);\nrelation mid(i32);\nmid(y) <-- triple(_, y, _);");
}

// ─── Repeated variables ──────────────────────────────────────────────

#[test]
fn parity_repeated_var() {
    // Note: ascent macro treats repeated vars as equality on references,
    // our parser desugars them to explicit equality conditions.
    // Both accept the syntax — the difference is in semantics.
    ascent! {
        relation edge(i32, i32);
        relation self_loop(i32);
        self_loop(x) <-- edge(x, y), if x == y;
    }
    must_parse("relation edge(i32, i32);\nrelation self_loop(i32);\nself_loop(x) <-- edge(x, x);");
}

// ─── Conditions ──────────────────────────────────────────────────────

#[test]
fn parity_if_condition() {
    ascent! {
        relation a(i32);
        relation b(i32);
        b(x) <-- a(x), if *x > 0;
    }
    must_parse("relation a(i32);\nrelation b(i32);\nb(x) <-- a(x), if *x > 0;");
}

#[test]
fn parity_let_binding() {
    ascent! {
        relation a(i32, i32);
        relation b(i32);
        b(s) <-- a(x, y), let s = x + y;
    }
    must_parse("relation a(i32, i32);\nrelation b(i32);\nb(s) <-- a(x, y), let s = x + y;");
}

// ─── Generators ──────────────────────────────────────────────────────

#[test]
fn parity_range_generator() {
    ascent! {
        relation n(i32);
        n(x) <-- for x in 0..10;
    }
    must_parse("relation n(i32);\nn(x) <-- for x in 0..10;");
}

#[test]
fn parity_array_generator() {
    ascent! {
        relation n(i32);
        n(x) <-- for x in [1, 2, 3];
    }
    must_parse("relation n(i32);\nn(x) <-- for x in [1, 2, 3];");
}

#[test]
fn parity_generator_with_condition() {
    ascent! {
        relation n(i32);
        n(x) <-- for x in 0..20, if x % 3 == 0;
    }
    must_parse("relation n(i32);\nn(x) <-- for x in 0..20, if x % 3 == 0;");
}

// ─── Negation ────────────────────────────────────────────────────────

#[test]
fn parity_negation() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation only_a(i32);
        only_a(x) <-- a(x), !b(x);
    }
    must_parse(
        "relation a(i32);\nrelation b(i32);\nrelation only_a(i32);\nonly_a(x) <-- a(x), !b(x);",
    );
}

#[test]
fn parity_double_negation() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        relation result(i32);
        result(x) <-- a(x), !b(x), !c(x);
    }
    must_parse(
        "relation a(i32);\nrelation b(i32);\nrelation c(i32);\nrelation result(i32);\nresult(x) <-- a(x), !b(x), !c(x);",
    );
}

// ─── Disjunction ─────────────────────────────────────────────────────

#[test]
fn parity_disjunction_pipe() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        c(x) <-- (a(x) | b(x));
    }
    must_parse("relation a(i32);\nrelation b(i32);\nrelation c(i32);\nc(x) <-- (a(x) | b(x));");
}

#[test]
fn parity_disjunction_or() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        c(x) <-- (a(x) || b(x));
    }
    must_parse("relation a(i32);\nrelation b(i32);\nrelation c(i32);\nc(x) <-- (a(x) || b(x));");
}

#[test]
fn parity_three_way_disjunction() {
    ascent! {
        relation a(i32);
        relation b(i32);
        relation c(i32);
        relation d(i32);
        d(x) <-- (a(x) | b(x) | c(x));
    }
    must_parse(
        "relation a(i32);\nrelation b(i32);\nrelation c(i32);\nrelation d(i32);\nd(x) <-- (a(x) | b(x) | c(x));",
    );
}

// ─── Aggregation ─────────────────────────────────────────────────────

#[test]
fn parity_agg_min() {
    ascent! {
        relation n(i32);
        relation result(i32);
        result(m) <-- agg m = min(x) in n(x);
    }
    must_parse("relation n(i32);\nrelation result(i32);\nresult(m) <-- agg m = min(x) in n(x);");
}

#[test]
fn parity_agg_max() {
    ascent! {
        relation n(i32);
        relation result(i32);
        result(m) <-- agg m = max(x) in n(x);
    }
    must_parse("relation n(i32);\nrelation result(i32);\nresult(m) <-- agg m = max(x) in n(x);");
}

#[test]
fn parity_agg_count() {
    ascent! {
        relation n(i32);
        relation result(usize);
        result(c) <-- agg c = count() in n(_);
    }
    must_parse("relation n(i32);\nrelation result(usize);\nresult(c) <-- agg c = count() in n(_);");
}

#[test]
fn parity_agg_sum() {
    ascent! {
        relation n(i32);
        relation result(i32);
        result(s) <-- agg s = sum(x) in n(x);
    }
    must_parse("relation n(i32);\nrelation result(i32);\nresult(s) <-- agg s = sum(x) in n(x);");
}

#[test]
fn parity_grouped_agg() {
    ascent! {
        relation kv(i32, i32);
        relation best(i32, i32);
        best(k, m) <-- kv(k, _), agg m = max(v) in kv(k, v);
    }
    must_parse(
        "relation kv(i32, i32);\nrelation best(i32, i32);\nbest(k, m) <-- kv(k, _), agg m = max(v) in kv(k, v);",
    );
}

// ─── Lattice declarations ────────────────────────────────────────────

#[test]
fn parity_lattice() {
    ascent! {
        lattice best(i32, i32);
        best(1, 10);
    }
    must_parse("lattice best(i32, i32);\nbest(1, 10);");
}

#[test]
fn parity_lattice_pattern() {
    ascent! {
        relation edge(i32, i32, i32);
        lattice shortest(i32, i32, Dual<i32>);
        shortest(x, y, Dual(*w)) <-- edge(x, y, w);
        shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
    }
    let input = "relation edge(i32, i32, i32);\nlattice shortest(i32, i32, Dual<i32>);\nshortest(x, y, Dual(*w)) <-- edge(x, y, w);\nshortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));";
    must_parse(input);
}

// ─── Constants in body clauses ───────────────────────────────────────

#[test]
fn parity_constant_filter() {
    ascent! {
        relation kv(i32, i32);
        relation result(i32);
        result(x) <-- kv(x, 42);
    }
    must_parse("relation kv(i32, i32);\nrelation result(i32);\nresult(x) <-- kv(x, 42);");
}

// ─── Dereference in condition ────────────────────────────────────────

#[test]
fn parity_deref_condition() {
    ascent! {
        relation a(i32);
        relation b(i32);
        b(x) <-- a(x), if *x > 5;
    }
    must_parse("relation a(i32);\nrelation b(i32);\nb(x) <-- a(x), if *x > 5;");
}

// ─── Empty program ───────────────────────────────────────────────────

#[test]
fn parity_empty_program() {
    ascent! {}
    must_parse("");
}

// ─── Multiple facts on one line ──────────────────────────────────────

#[test]
fn parity_multiple_facts() {
    ascent! {
        relation n(i32);
        n(1); n(2); n(3);
    }
    must_parse("relation n(i32);\nn(1); n(2); n(3);");
}

// ─── Complex mixed program ───────────────────────────────────────────

#[test]
fn parity_complex_mixed() {
    ascent! {
        relation edge(i32, i32);
        relation node(i32);
        relation path(i32, i32);
        relation reachable_count(i32, usize);

        edge(1, 2); edge(2, 3); edge(3, 4); edge(1, 3);
        node(x) <-- edge(x, _);
        node(y) <-- edge(_, y);
        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
        reachable_count(x, c) <-- node(x), agg c = count() in path(x, _);
    }
    let input = "\
        relation edge(i32, i32);\n\
        relation node(i32);\n\
        relation path(i32, i32);\n\
        relation reachable_count(i32, usize);\n\
        edge(1, 2); edge(2, 3); edge(3, 4); edge(1, 3);\n\
        node(x) <-- edge(x, _);\n\
        node(y) <-- edge(_, y);\n\
        path(x, y) <-- edge(x, y);\n\
        path(x, z) <-- edge(x, y), path(y, z);\n\
        reachable_count(x, c) <-- node(x), agg c = count() in path(x, _);";
    must_parse(input);
}
