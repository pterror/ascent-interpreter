//! Semi-naive evaluation engine.

use std::collections::HashMap;

use ascent_ir::{Aggregation, BodyItem, Clause, ClauseArg, Condition, Program, Rule};

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

    /// Run the program to fixpoint using semi-naive evaluation.
    pub fn run(&mut self, program: &Program) {
        // Initial iteration: evaluate all rules once
        for rule in &program.rules {
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

            for rule in &program.rules {
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
        if use_recent && !self.rule_has_recent_clause(rule) {
            return;
        }

        let derived = self.derive_tuples(rule, use_recent);

        for (relation, tuple) in derived {
            self.insert(&relation, tuple);
        }
    }

    /// Check if any clause in the rule has recent tuples.
    fn rule_has_recent_clause(&self, rule: &Rule) -> bool {
        for item in &rule.body {
            if let BodyItem::Clause(clause) = item
                && let Some(rel) = self.relations.get(&clause.relation)
                && rel.iter_recent().next().is_some()
            {
                return true;
            }
        }
        false
    }

    /// Derive all tuples from a rule.
    fn derive_tuples(&self, rule: &Rule, use_recent: bool) -> Vec<(String, Tuple)> {
        let mut results = Vec::new();

        let initial_bindings = vec![Bindings::new()];
        let final_bindings = self.process_body(&rule.body, initial_bindings, use_recent);

        for bindings in final_bindings {
            for head in &rule.heads {
                if let Some(tuple) = self.eval_head_tuple(head, &bindings) {
                    results.push((head.relation.clone(), tuple));
                }
            }
        }

        results
    }

    /// Process body items, producing all valid bindings.
    fn process_body(
        &self,
        body: &[BodyItem],
        bindings: Vec<Bindings>,
        use_recent: bool,
    ) -> Vec<Bindings> {
        let mut current = bindings;

        for (i, item) in body.iter().enumerate() {
            if current.is_empty() {
                break;
            }

            current = match item {
                BodyItem::Clause(clause) => {
                    let use_recent_for_this = use_recent && i == 0;
                    self.process_clause(clause, current, use_recent_for_this)
                }
                BodyItem::Generator(generator) => self.process_generator(generator, current),
                BodyItem::Condition(cond) => self.process_condition(cond, current),
                BodyItem::Aggregation(agg) => self.process_aggregation(agg, current),
            };
        }

        current
    }

    /// Process a clause against the relation.
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

        let tuples: Vec<_> = if use_recent {
            rel.iter_recent().collect()
        } else {
            rel.iter_full().collect()
        };

        for binding in bindings {
            for tuple in &tuples {
                if let Some(new_binding) = self.match_clause(clause, tuple, binding.clone())
                    && self.check_clause_conditions(clause, &new_binding)
                {
                    results.push(new_binding);
                }
            }
        }

        results
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

    /// Check additional conditions on a clause.
    fn check_clause_conditions(&self, clause: &Clause, bindings: &Bindings) -> bool {
        for cond in &clause.conditions {
            if !self.eval_condition(cond, bindings) {
                return false;
            }
        }
        true
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

    /// Process a condition filter.
    fn process_condition(&self, cond: &Condition, bindings: Vec<Bindings>) -> Vec<Bindings> {
        bindings
            .into_iter()
            .filter(|b| self.eval_condition(cond, b))
            .collect()
    }

    /// Evaluate a condition.
    fn eval_condition(&self, cond: &Condition, bindings: &Bindings) -> bool {
        match cond {
            Condition::If(expr) => eval_expr(expr, bindings)
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            Condition::IfLet { .. } => {
                // TODO: implement pattern matching
                true
            }
            Condition::Let { .. } => {
                // TODO: implement let binding
                true
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
}
