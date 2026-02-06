//! Semi-naive evaluation engine.

use std::collections::HashMap;

use ascent_ir::{Aggregation, BodyItem, Clause, ClauseArg, Condition, Program, Rule};
use petgraph::algo::{condensation, toposort};
use petgraph::graph::DiGraph;

use crate::aggregators::apply_aggregator;
use crate::expr::{eval_expr, expand_range};
use crate::relation::RelationStorage;
use crate::value::{Tuple, Value};

/// Variable bindings during rule evaluation.
pub type Bindings = HashMap<String, Value>;

/// The evaluation engine state.
#[derive(Debug)]
pub struct Engine {
    /// Storage for each relation.
    relations: HashMap<String, RelationStorage>,
}

impl Engine {
    /// Create a new engine from a program.
    pub fn new(program: &Program) -> Self {
        let mut relations = HashMap::new();

        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            relations.insert(name.clone(), RelationStorage::new(arity));
        }

        Engine { relations }
    }

    /// Get a relation by name.
    pub fn relation(&self, name: &str) -> Option<&RelationStorage> {
        self.relations.get(name)
    }

    /// Get a mutable relation by name.
    pub fn relation_mut(&mut self, name: &str) -> Option<&mut RelationStorage> {
        self.relations.get_mut(name)
    }

    /// Insert a tuple into a relation.
    pub fn insert(&mut self, relation: &str, tuple: Tuple) -> bool {
        if let Some(rel) = self.relations.get_mut(relation) {
            rel.insert(tuple)
        } else {
            false
        }
    }

    /// Run the program to fixpoint using semi-naive evaluation with SCC-based stratification.
    ///
    /// Rules are grouped into strongly connected components and processed in
    /// topological order. Each SCC runs to fixpoint before dependent SCCs begin.
    pub fn run(&mut self, program: &Program) {
        let sccs = compute_rule_sccs(program);
        for scc_indices in &sccs {
            let rules: Vec<&Rule> = scc_indices.iter().map(|&i| &program.rules[i]).collect();
            self.run_stratum(&rules);
        }
    }

    /// Run a set of rules to fixpoint.
    fn run_stratum(&mut self, rules: &[&Rule]) {
        if rules.is_empty() {
            return;
        }

        // Initial iteration: evaluate all rules once
        for rule in rules {
            self.evaluate_rule(rule, false);
        }

        // Advance all relations
        let mut changed = false;
        for rel in self.relations.values_mut() {
            if rel.advance() {
                changed = true;
            }
        }

        // Semi-naive loop
        while changed {
            changed = false;

            for rule in rules {
                self.evaluate_rule(rule, true);
            }

            for rel in self.relations.values_mut() {
                if rel.advance() {
                    changed = true;
                }
            }
        }
    }

    /// Evaluate a single rule.
    fn evaluate_rule(&mut self, rule: &Rule, use_recent: bool) {
        if use_recent && !self.rule_has_recent_data(rule) {
            return;
        }

        let derived = self.derive_tuples(rule, use_recent);

        for (relation, tuple) in derived {
            self.insert(&relation, tuple);
        }
    }

    /// Check if any relation referenced in the rule body has recent tuples.
    fn rule_has_recent_data(&self, rule: &Rule) -> bool {
        for item in &rule.body {
            let rel_name = match item {
                BodyItem::Clause(clause) => &clause.relation,
                BodyItem::Aggregation(agg) => &agg.relation,
                _ => continue,
            };
            if let Some(rel) = self.relations.get(rel_name)
                && rel.iter_recent().next().is_some()
            {
                return true;
            }
        }
        false
    }

    /// Derive all tuples from a rule.
    ///
    /// In semi-naive mode, for each clause position that has recent tuples,
    /// evaluate the rule with that clause using recent and all others using full.
    /// This ensures we discover all new derivations regardless of which clause
    /// received new data.
    fn derive_tuples(&self, rule: &Rule, use_recent: bool) -> Vec<(String, Tuple)> {
        let mut results = Vec::new();

        if !use_recent {
            // Initial iteration: all clauses use full
            let final_bindings = self.process_body(&rule.body, vec![Bindings::new()], None);
            self.collect_head_tuples(rule, &final_bindings, &mut results);
            return results;
        }

        // Semi-naive: try each clause position with recent data
        for (idx, item) in rule.body.iter().enumerate() {
            let rel_name = match item {
                BodyItem::Clause(c) => &c.relation,
                _ => continue,
            };
            if let Some(rel) = self.relations.get(rel_name)
                && rel.iter_recent().next().is_some()
            {
                let final_bindings =
                    self.process_body(&rule.body, vec![Bindings::new()], Some(idx));
                self.collect_head_tuples(rule, &final_bindings, &mut results);
            }
        }

        results
    }

    /// Collect head tuples from a set of bindings.
    fn collect_head_tuples(
        &self,
        rule: &Rule,
        bindings: &[Bindings],
        results: &mut Vec<(String, Tuple)>,
    ) {
        for binding in bindings {
            for head in &rule.heads {
                if let Some(tuple) = self.eval_head_tuple(head, binding) {
                    results.push((head.relation.clone(), tuple));
                }
            }
        }
    }

    /// Process body items, producing all valid bindings.
    ///
    /// `recent_clause_idx`: if Some(i), clause at position i uses recent tuples,
    /// all other clauses use full. If None, all use full (initial iteration).
    fn process_body(
        &self,
        body: &[BodyItem],
        bindings: Vec<Bindings>,
        recent_clause_idx: Option<usize>,
    ) -> Vec<Bindings> {
        let mut current = bindings;

        for (i, item) in body.iter().enumerate() {
            if current.is_empty() {
                break;
            }

            current = match item {
                BodyItem::Clause(clause) => {
                    let use_recent = recent_clause_idx == Some(i);
                    self.process_clause(clause, current, use_recent)
                }
                BodyItem::Generator(generator) => self.process_generator(generator, current),
                BodyItem::Condition(cond) => self.process_condition(cond, current),
                BodyItem::Aggregation(agg) => self.process_aggregation(agg, current),
            };
        }

        current
    }

    /// Process a clause against the relation using index lookups when possible.
    fn process_clause(
        &self,
        clause: &Clause,
        bindings: Vec<Bindings>,
        use_recent: bool,
    ) -> Vec<Bindings> {
        let Some(rel) = self.relations.get(&clause.relation) else {
            return vec![];
        };

        let mut results = Vec::new();

        for binding in &bindings {
            // Find the best column to index on: a bound variable or evaluable expression
            let index_col = self.find_index_column(clause, binding);

            if let Some((col, value)) = index_col {
                // Use index lookup: only check matching tuples
                let indices = rel.lookup(col, &value);
                for &idx in indices {
                    if use_recent && !rel.is_recent(idx) {
                        continue;
                    }
                    let tuple = rel.get(idx);
                    if let Some(new_binding) = self.match_clause(clause, tuple, binding.clone())
                        && let Some(final_binding) =
                            self.check_clause_conditions(clause, new_binding)
                    {
                        results.push(final_binding);
                    }
                }
            } else {
                // No bound columns: full scan
                let iter: Box<dyn Iterator<Item = &Tuple>> = if use_recent {
                    Box::new(rel.iter_recent())
                } else {
                    Box::new(rel.iter_full())
                };
                for tuple in iter {
                    if let Some(new_binding) = self.match_clause(clause, tuple, binding.clone())
                        && let Some(final_binding) =
                            self.check_clause_conditions(clause, new_binding)
                    {
                        results.push(final_binding);
                    }
                }
            }
        }

        results
    }

    /// Find the best column to use for index lookup.
    ///
    /// Returns Some((column_index, value)) if a clause arg is already bound.
    fn find_index_column(&self, clause: &Clause, binding: &Bindings) -> Option<(usize, Value)> {
        for (col, arg) in clause.args.iter().enumerate() {
            match arg {
                ClauseArg::Var(var) => {
                    if let Some(val) = binding.get(var) {
                        return Some((col, val.clone()));
                    }
                }
                ClauseArg::Expr(expr) => {
                    if let Some(val) = eval_expr(expr, binding) {
                        return Some((col, val));
                    }
                }
            }
        }
        None
    }

    /// Try to match a clause against a tuple, extending bindings.
    fn match_clause(
        &self,
        clause: &Clause,
        tuple: &Tuple,
        mut bindings: Bindings,
    ) -> Option<Bindings> {
        if clause.args.len() != tuple.len() {
            return None;
        }

        for (arg, value) in clause.args.iter().zip(tuple.iter()) {
            match arg {
                ClauseArg::Var(var) => {
                    if let Some(existing) = bindings.get(var) {
                        if existing != value {
                            return None;
                        }
                    } else {
                        bindings.insert(var.clone(), value.clone());
                    }
                }
                ClauseArg::Expr(expr) => {
                    if let Some(evaluated) = eval_expr(expr, &bindings)
                        && evaluated != *value
                    {
                        return None;
                    }
                }
            }
        }

        Some(bindings)
    }

    /// Check additional conditions on a clause, potentially extending bindings.
    fn check_clause_conditions(&self, clause: &Clause, mut bindings: Bindings) -> Option<Bindings> {
        for cond in &clause.conditions {
            bindings = self.eval_condition(cond, bindings)?;
        }
        Some(bindings)
    }

    /// Process a generator.
    fn process_generator(
        &self,
        generator: &ascent_ir::Generator,
        bindings: Vec<Bindings>,
    ) -> Vec<Bindings> {
        let mut results = Vec::new();

        for binding in bindings {
            if let Some(range_val) = eval_expr(&generator.expr, &binding)
                && let Some(values) = expand_range(&range_val)
            {
                for value in values {
                    let mut new_binding = binding.clone();
                    if let Some(var) = generator.vars.first() {
                        new_binding.insert(var.clone(), value);
                    }
                    results.push(new_binding);
                }
            }
        }

        results
    }

    /// Process an aggregation clause.
    fn process_aggregation(&self, agg: &Aggregation, bindings: Vec<Bindings>) -> Vec<Bindings> {
        let Some(rel) = self.relations.get(&agg.relation) else {
            return vec![];
        };

        // Resolve aggregator name from the expression
        let agg_name = resolve_aggregator_name(&agg.aggregator);

        let mut results = Vec::new();

        for binding in &bindings {
            // Find all matching tuples and extract bound variables
            let mut collected: Vec<Vec<Value>> = Vec::new();

            for tuple in rel.iter_full() {
                if let Some(match_binding) = self.match_agg_args(agg, tuple, binding.clone()) {
                    // Extract bound variable values
                    let bound_vals: Vec<Value> = agg
                        .bound_vars
                        .iter()
                        .filter_map(|var| match_binding.get(var).cloned())
                        .collect();
                    collected.push(bound_vals);
                }
            }

            // Apply aggregator
            let agg_results = apply_aggregator(&agg_name, collected);

            // Bind results to output pattern variables
            for result_tuple in agg_results {
                let mut new_binding = binding.clone();
                for (var, val) in agg.result_vars.iter().zip(result_tuple) {
                    new_binding.insert(var.clone(), val);
                }
                results.push(new_binding);
            }
        }

        results
    }

    /// Match aggregation relation arguments against a tuple.
    fn match_agg_args(
        &self,
        agg: &Aggregation,
        tuple: &Tuple,
        mut bindings: Bindings,
    ) -> Option<Bindings> {
        if agg.args.len() != tuple.len() {
            return None;
        }

        for (arg_expr, value) in agg.args.iter().zip(tuple.iter()) {
            // Check if the argument is a simple variable (bound var)
            if let syn::Expr::Path(p) = arg_expr
                && let Some(ident) = p.path.get_ident()
            {
                let name = ident.to_string();
                if let Some(existing) = bindings.get(&name) {
                    if existing != value {
                        return None;
                    }
                } else {
                    bindings.insert(name, value.clone());
                }
                continue;
            }

            // Otherwise evaluate and compare
            if let Some(evaluated) = eval_expr(arg_expr, &bindings)
                && evaluated != *value
            {
                return None;
            }
        }

        Some(bindings)
    }

    /// Process a condition, potentially binding new variables.
    fn process_condition(&self, cond: &Condition, bindings: Vec<Bindings>) -> Vec<Bindings> {
        bindings
            .into_iter()
            .filter_map(|b| self.eval_condition(cond, b))
            .collect()
    }

    /// Evaluate a condition. Returns updated bindings on success, None on failure.
    fn eval_condition(&self, cond: &Condition, bindings: Bindings) -> Option<Bindings> {
        match cond {
            Condition::If(expr) => {
                if eval_expr(expr, &bindings)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    Some(bindings)
                } else {
                    None
                }
            }
            Condition::IfLet { pattern, expr } => {
                let value = eval_expr(expr, &bindings)?;
                let mut new_bindings = bindings;
                if match_pattern(pattern, &value, &mut new_bindings) {
                    Some(new_bindings)
                } else {
                    None
                }
            }
            Condition::Let { pattern, expr } => {
                let value = eval_expr(expr, &bindings)?;
                let mut new_bindings = bindings;
                if match_pattern(pattern, &value, &mut new_bindings) {
                    Some(new_bindings)
                } else {
                    None
                }
            }
        }
    }

    /// Evaluate a head clause to produce a tuple.
    fn eval_head_tuple(&self, head: &ascent_ir::HeadClause, bindings: &Bindings) -> Option<Tuple> {
        let mut tuple = Vec::with_capacity(head.args.len());

        for arg in &head.args {
            if let Some(value) = eval_expr(arg, bindings) {
                tuple.push(value);
            } else {
                return None;
            }
        }

        Some(tuple)
    }
}

/// Compute strongly connected components of the rule dependency graph.
///
/// Returns rule indices grouped by SCC, in topological order (dependencies first).
fn compute_rule_sccs(program: &Program) -> Vec<Vec<usize>> {
    let n = program.rules.len();
    if n == 0 {
        return vec![];
    }

    // Map: relation_name → rule indices that produce it (appear in head)
    let mut producers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, rule) in program.rules.iter().enumerate() {
        for head in &rule.heads {
            producers.entry(&head.relation).or_default().push(i);
        }
    }

    // Build directed graph: edge from producer → consumer
    let mut graph = DiGraph::<usize, ()>::new();
    let nodes: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

    for (consumer_idx, rule) in program.rules.iter().enumerate() {
        for item in &rule.body {
            let rel_name = match item {
                BodyItem::Clause(c) => Some(c.relation.as_str()),
                BodyItem::Aggregation(a) => Some(a.relation.as_str()),
                _ => None,
            };
            if let Some(rel) = rel_name
                && let Some(prod_indices) = producers.get(rel)
            {
                for &prod_idx in prod_indices {
                    graph.add_edge(nodes[prod_idx], nodes[consumer_idx], ());
                }
            }
        }
    }

    // Condensation: collapse SCCs into single nodes in a DAG
    let condensed = condensation(graph, true);

    // Topological sort of the condensation DAG
    let order = toposort(&condensed, None).expect("condensation is always a DAG");

    // Extract rule indices for each SCC in topological order
    order.iter().map(|&idx| condensed[idx].clone()).collect()
}

/// Match a pattern against a value, extending bindings on success.
fn match_pattern(pat: &syn::Pat, value: &Value, bindings: &mut Bindings) -> bool {
    match pat {
        // Wildcard: always matches, binds nothing
        syn::Pat::Wild(_) => true,

        // Variable binding: `x` or `mut x`
        syn::Pat::Ident(ident) => {
            let name = ident.ident.to_string();
            // Handle `_` identifiers as wildcards
            if name == "_" {
                return true;
            }
            // If there's a subpattern (`name @ pattern`), match it too
            if let Some((_, sub_pat)) = &ident.subpat
                && !match_pattern(sub_pat, value, bindings)
            {
                return false;
            }
            bindings.insert(name, value.clone());
            true
        }

        // Literal pattern: `42`, `true`, `'a'`
        syn::Pat::Lit(expr_lit) => {
            let expr = syn::Expr::Lit(expr_lit.clone());
            if let Some(lit_val) = eval_expr(&expr, bindings) {
                lit_val == *value
            } else {
                false
            }
        }

        // Reference pattern: `&x` — in Datalog context, match the inner pattern
        syn::Pat::Reference(r) => match_pattern(&r.pat, value, bindings),

        // Tuple pattern: `(a, b, c)`
        syn::Pat::Tuple(tuple) => {
            if let Value::Tuple(vals) = value {
                if vals.len() != tuple.elems.len() {
                    return false;
                }
                for (pat_elem, val) in tuple.elems.iter().zip(vals.iter()) {
                    if !match_pattern(pat_elem, val, bindings) {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        // Tuple struct pattern: `Some(x)`, `Ok(v)`
        syn::Pat::TupleStruct(ts) => {
            let path_str = path_to_string(&ts.path);
            match path_str.as_str() {
                "Some" => {
                    if let Value::Option(Some(inner)) = value
                        && ts.elems.len() == 1
                    {
                        match_pattern(&ts.elems[0], inner, bindings)
                    } else {
                        false
                    }
                }
                "None" => matches!(value, Value::Option(None)),
                _ => false,
            }
        }

        // Path pattern: `None`, `true`, `false`
        syn::Pat::Path(p) => {
            let path_str = path_to_string(&p.path);
            match path_str.as_str() {
                "None" => matches!(value, Value::Option(None)),
                "true" => matches!(value, Value::Bool(true)),
                "false" => matches!(value, Value::Bool(false)),
                _ => false,
            }
        }

        // Or pattern: `A | B`
        syn::Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                let mut trial = bindings.clone();
                if match_pattern(case, value, &mut trial) {
                    *bindings = trial;
                    return true;
                }
            }
            false
        }

        // Parenthesized: `(pattern)`
        syn::Pat::Paren(p) => match_pattern(&p.pat, value, bindings),

        _ => false,
    }
}

/// Convert a syn::Path to a simple string (last segment).
fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default()
}

/// Extract the aggregator name from an expression.
/// Handles both simple paths (`min`) and qualified paths (`ascent::aggregators::min`).
fn resolve_aggregator_name(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Path(p) => {
            // Use the last segment of the path
            p.path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default()
        }
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ascent_ir::Program;
    use ascent_syntax::AscentProgram;

    fn run_program(input: &str) -> Engine {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.run(&program);
        engine
    }

    fn run_with_facts(input: &str, facts: Vec<(&str, Vec<Tuple>)>) -> Engine {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        for (rel, tuples) in facts {
            for tuple in tuples {
                engine.insert(rel, tuple);
            }
        }

        engine.run(&program);
        engine
    }

    #[test]
    fn test_fact_with_literals() {
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            edge(1, 2);
            edge(2, 3);
        "#,
        );

        let edge = engine.relation("edge").unwrap();
        assert!(edge.contains(&vec![Value::I32(1), Value::I32(2)]));
        assert!(edge.contains(&vec![Value::I32(2), Value::I32(3)]));
        assert_eq!(edge.len(), 2);
    }

    #[test]
    fn test_transitive_closure() {
        let engine = run_with_facts(
            r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
            vec![(
                "edge",
                vec![
                    vec![Value::I32(1), Value::I32(2)],
                    vec![Value::I32(2), Value::I32(3)],
                    vec![Value::I32(3), Value::I32(4)],
                ],
            )],
        );

        let path = engine.relation("path").unwrap();
        assert!(path.contains(&vec![Value::I32(1), Value::I32(2)]));
        assert!(path.contains(&vec![Value::I32(1), Value::I32(3)]));
        assert!(path.contains(&vec![Value::I32(1), Value::I32(4)]));
        assert!(path.contains(&vec![Value::I32(2), Value::I32(3)]));
        assert!(path.contains(&vec![Value::I32(2), Value::I32(4)]));
        assert!(path.contains(&vec![Value::I32(3), Value::I32(4)]));
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_transitive_closure_from_facts() {
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            edge(1, 2);
            edge(2, 3);
            edge(3, 4);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#,
        );

        let path = engine.relation("path").unwrap();
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_join() {
        let engine = run_with_facts(
            r#"
            relation r(i32, i32);
            relation s(i32, i32);
            relation joined(i32, i32, i32);
            joined(a, b, c) <-- r(a, b), s(b, c);
        "#,
            vec![
                (
                    "r",
                    vec![
                        vec![Value::I32(1), Value::I32(2)],
                        vec![Value::I32(3), Value::I32(4)],
                    ],
                ),
                (
                    "s",
                    vec![
                        vec![Value::I32(2), Value::I32(5)],
                        vec![Value::I32(4), Value::I32(6)],
                    ],
                ),
            ],
        );

        let joined = engine.relation("joined").unwrap();
        assert!(joined.contains(&vec![Value::I32(1), Value::I32(2), Value::I32(5)]));
        assert!(joined.contains(&vec![Value::I32(3), Value::I32(4), Value::I32(6)]));
        assert_eq!(joined.len(), 2);
    }

    #[test]
    fn test_condition_filter() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation even(i32);
            number(1);
            number(2);
            number(3);
            number(4);
            number(5);
            even(x) <-- number(x), if x % 2 == 0;
        "#,
        );

        let even = engine.relation("even").unwrap();
        assert!(even.contains(&vec![Value::I32(2)]));
        assert!(even.contains(&vec![Value::I32(4)]));
        assert_eq!(even.len(), 2);
    }

    #[test]
    fn test_generator() {
        let engine = run_program(
            r#"
            relation number(i32);
            number(x) <-- for x in 0..5;
        "#,
        );

        let number = engine.relation("number").unwrap();
        assert_eq!(number.len(), 5);
        assert!(number.contains(&vec![Value::I32(0)]));
        assert!(number.contains(&vec![Value::I32(4)]));
    }

    #[test]
    fn test_arithmetic_in_head() {
        let engine = run_program(
            r#"
            relation input(i32);
            relation doubled(i32);
            input(1);
            input(2);
            input(3);
            doubled(x * 2) <-- input(x);
        "#,
        );

        let doubled = engine.relation("doubled").unwrap();
        assert!(doubled.contains(&vec![Value::I32(2)]));
        assert!(doubled.contains(&vec![Value::I32(4)]));
        assert!(doubled.contains(&vec![Value::I32(6)]));
        assert_eq!(doubled.len(), 3);
    }

    #[test]
    fn test_generator_with_condition() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation small_even(i32);
            number(x) <-- for x in 0..20;
            small_even(x) <-- number(x), if x % 2 == 0, if x < 10;
        "#,
        );

        let small_even = engine.relation("small_even").unwrap();
        assert_eq!(small_even.len(), 5); // 0, 2, 4, 6, 8
        assert!(small_even.contains(&vec![Value::I32(0)]));
        assert!(small_even.contains(&vec![Value::I32(8)]));
        assert!(!small_even.contains(&vec![Value::I32(10)]));
    }

    #[test]
    fn test_fibonacci_like() {
        let engine = run_program(
            r#"
            relation fib(i32, i32);
            fib(0, 1);
            fib(1, 1);
            fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if n < 10;
        "#,
        );

        let fib = engine.relation("fib").unwrap();
        assert!(fib.contains(&vec![Value::I32(0), Value::I32(1)]));
        assert!(fib.contains(&vec![Value::I32(1), Value::I32(1)]));
        assert!(fib.contains(&vec![Value::I32(2), Value::I32(2)]));
        assert!(fib.contains(&vec![Value::I32(3), Value::I32(3)]));
        assert!(fib.contains(&vec![Value::I32(4), Value::I32(5)]));
        assert!(fib.contains(&vec![Value::I32(5), Value::I32(8)]));
    }

    #[test]
    fn test_aggregation_min() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation lowest(i32);
            number(5);
            number(3);
            number(8);
            number(1);
            number(7);
            lowest(y) <-- agg y = min(x) in number(x);
        "#,
        );

        let lowest = engine.relation("lowest").unwrap();
        assert_eq!(lowest.len(), 1);
        assert!(lowest.contains(&vec![Value::I32(1)]));
    }

    #[test]
    fn test_aggregation_max() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation highest(i32);
            number(5);
            number(3);
            number(8);
            highest(y) <-- agg y = max(x) in number(x);
        "#,
        );

        let highest = engine.relation("highest").unwrap();
        assert_eq!(highest.len(), 1);
        assert!(highest.contains(&vec![Value::I32(8)]));
    }

    #[test]
    fn test_aggregation_sum() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation total(i32);
            number(1);
            number(2);
            number(3);
            total(s) <-- agg s = sum(x) in number(x);
        "#,
        );

        let total = engine.relation("total").unwrap();
        assert_eq!(total.len(), 1);
        assert!(total.contains(&vec![Value::I32(6)]));
    }

    #[test]
    fn test_aggregation_count() {
        let engine = run_program(
            r#"
            relation number(i32);
            relation card(i32);
            number(10);
            number(20);
            number(30);
            card(n) <-- agg n = count() in number(_);
        "#,
        );

        let card = engine.relation("card").unwrap();
        assert_eq!(card.len(), 1);
        assert!(card.contains(&vec![Value::I32(3)]));
    }

    #[test]
    fn test_negation_as_aggregation() {
        let engine = run_program(
            r#"
            relation a(i32);
            relation b(i32);
            relation only_a(i32);
            a(1);
            a(2);
            a(3);
            b(2);
            only_a(x) <-- a(x), !b(x);
        "#,
        );

        let only_a = engine.relation("only_a").unwrap();
        assert!(only_a.contains(&vec![Value::I32(1)]));
        assert!(only_a.contains(&vec![Value::I32(3)]));
        assert!(!only_a.contains(&vec![Value::I32(2)]));
        assert_eq!(only_a.len(), 2);
    }

    #[test]
    fn test_stratification_agg_after_recursion() {
        // Aggregation over a recursively-computed relation.
        // fib_max should see the complete fib relation, not partial results.
        let engine = run_program(
            r#"
            relation fib(i32, i32);
            relation fib_max(i32);
            fib(0, 1);
            fib(1, 1);
            fib(n + 1, a + b) <-- fib(n, a), fib(n - 1, b), if n < 10;
            fib_max(m) <-- agg m = max(v) in fib(_, v);
        "#,
        );

        let fib_max = engine.relation("fib_max").unwrap();
        assert_eq!(fib_max.len(), 1);
        assert!(fib_max.contains(&vec![Value::I32(89)]));
    }

    #[test]
    fn test_negation_with_recursive_input() {
        // Negation (which uses aggregation internally) should see complete
        // recursive results thanks to stratification.
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            relation unreachable(i32);
            relation node(i32);
            edge(1, 2);
            edge(2, 3);
            node(1);
            node(2);
            node(3);
            node(4);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
            unreachable(x) <-- node(x), !path(1, x);
        "#,
        );

        let unreachable = engine.relation("unreachable").unwrap();
        // Node 4 is unreachable from node 1; node 1 is unreachable from itself
        assert!(unreachable.contains(&vec![Value::I32(4)]));
        assert!(unreachable.contains(&vec![Value::I32(1)]));
        assert_eq!(unreachable.len(), 2);
    }

    // ─── SCC stratification tests ────────────────────────────────────

    #[test]
    fn test_scc_multi_stratum_chain() {
        // Three strata: base facts → recursive path → aggregation on path
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            relation path_count(i32);
            relation max_dest(i32, i32);

            edge(1, 2);
            edge(2, 3);
            edge(3, 4);

            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);

            path_count(c) <-- agg c = count() in path(_, _);
            max_dest(src, m) <-- path(src, _), agg m = max(d) in path(src, d);
        "#,
        );

        // 6 paths: (1,2),(1,3),(1,4),(2,3),(2,4),(3,4)
        assert_eq!(engine.relation("path").unwrap().len(), 6);
        assert!(
            engine
                .relation("path_count")
                .unwrap()
                .contains(&vec![Value::I32(6)])
        );
        // max_dest: 1→4, 2→4, 3→4
        let max_dest = engine.relation("max_dest").unwrap();
        assert!(max_dest.contains(&vec![Value::I32(1), Value::I32(4)]));
        assert!(max_dest.contains(&vec![Value::I32(2), Value::I32(4)]));
        assert!(max_dest.contains(&vec![Value::I32(3), Value::I32(4)]));
        assert_eq!(max_dest.len(), 3);
    }

    #[test]
    fn test_scc_independent_sccs() {
        // Two independent recursive computations that don't depend on each other
        let engine = run_program(
            r#"
            relation a_edge(i32, i32);
            relation a_path(i32, i32);
            relation b_edge(i32, i32);
            relation b_path(i32, i32);

            a_edge(1, 2);
            a_edge(2, 3);
            a_path(x, y) <-- a_edge(x, y);
            a_path(x, z) <-- a_edge(x, y), a_path(y, z);

            b_edge(10, 20);
            b_edge(20, 30);
            b_edge(30, 40);
            b_path(x, y) <-- b_edge(x, y);
            b_path(x, z) <-- b_edge(x, y), b_path(y, z);
        "#,
        );

        // a_path: (1,2),(1,3),(2,3)
        assert_eq!(engine.relation("a_path").unwrap().len(), 3);
        // b_path: (10,20),(10,30),(10,40),(20,30),(20,40),(30,40)
        assert_eq!(engine.relation("b_path").unwrap().len(), 6);
    }

    #[test]
    fn test_scc_aggregation_feeds_downstream() {
        // Aggregation result used by a subsequent non-aggregation rule
        let engine = run_program(
            r#"
            relation score(i32, i32);
            relation total(i32);
            relation is_high(i32);

            score(1, 10);
            score(2, 20);
            score(3, 30);

            total(s) <-- agg s = sum(x) in score(_, x);
            is_high(t) <-- total(t), if *t > 50;
        "#,
        );

        assert!(
            engine
                .relation("total")
                .unwrap()
                .contains(&vec![Value::I32(60)])
        );
        // 60 > 50, so is_high should have (60)
        assert!(
            engine
                .relation("is_high")
                .unwrap()
                .contains(&vec![Value::I32(60)])
        );
    }

    #[test]
    fn test_scc_cascading_aggregations() {
        // Multiple layers of aggregation: data → grouped agg → global agg
        let engine = run_program(
            r#"
            relation score(i32, i32);
            relation best(i32, i32);
            relation overall_best(i32);

            score(1, 10);
            score(1, 20);
            score(2, 30);
            score(2, 15);

            best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
            overall_best(m) <-- agg m = max(s) in best(_, s);
        "#,
        );

        let best = engine.relation("best").unwrap();
        assert!(best.contains(&vec![Value::I32(1), Value::I32(20)]));
        assert!(best.contains(&vec![Value::I32(2), Value::I32(30)]));
        assert_eq!(best.len(), 2);

        // overall_best: max(20, 30) = 30
        assert!(
            engine
                .relation("overall_best")
                .unwrap()
                .contains(&vec![Value::I32(30)])
        );
    }

    #[test]
    fn test_scc_negation_after_recursion() {
        // Negation (desugared to aggregation) depends on recursive relation
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            relation node(i32);
            relation reach(i32, i32);
            relation isolated(i32);

            node(1); node(2); node(3); node(4); node(5);
            edge(1, 2); edge(2, 3);

            reach(x, y) <-- edge(x, y);
            reach(x, z) <-- edge(x, y), reach(y, z);

            isolated(x) <-- node(x), !reach(x, _), !reach(_, x);
        "#,
        );

        let reach = engine.relation("reach").unwrap();
        // reach: (1,2),(1,3),(2,3)
        assert_eq!(reach.len(), 3);

        let isolated = engine.relation("isolated").unwrap();
        // Nodes 4 and 5 have no edges at all
        assert!(isolated.contains(&vec![Value::I32(4)]));
        assert!(isolated.contains(&vec![Value::I32(5)]));
        assert_eq!(isolated.len(), 2);
    }
}
