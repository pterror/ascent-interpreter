//! Semi-naive evaluation engine.

use std::collections::HashMap;

use ascent_ir::{BodyItem, Clause, ClauseArg, Condition, Program, Rule};

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
                // For semi-naive, we need at least one clause to use recent tuples
                self.evaluate_rule(rule, true);
            }

            // Advance all relations
            for rel in self.relations.values_mut() {
                if rel.advance() {
                    changed = true;
                }
            }
        }
    }

    /// Evaluate a single rule.
    fn evaluate_rule(&mut self, rule: &Rule, use_recent: bool) {
        // If using semi-naive and no clauses use recent, skip
        if use_recent && !self.rule_has_recent_clause(rule) {
            return;
        }

        // Collect all derived tuples first to avoid borrow issues
        let derived = self.derive_tuples(rule, use_recent);

        // Insert derived tuples
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

        // Start with empty bindings
        let initial_bindings = vec![Bindings::new()];

        // Process each body item
        let final_bindings = self.process_body(&rule.body, initial_bindings, use_recent);

        // For each complete binding, produce head tuples
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
                    // For semi-naive: at least one clause should use recent
                    let use_recent_for_this = use_recent && i == 0;
                    self.process_clause(clause, current, use_recent_for_this)
                }
                BodyItem::Generator(generator) => self.process_generator(generator, current),
                BodyItem::Condition(cond) => self.process_condition(cond, current),
                BodyItem::Aggregation(_agg) => {
                    // TODO: implement aggregation
                    current
                }
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

        // Choose iterator based on semi-naive flag
        let tuples: Vec<_> = if use_recent {
            rel.iter_recent().collect()
        } else {
            rel.iter_full().collect()
        };

        for binding in bindings {
            for tuple in &tuples {
                if let Some(new_binding) = self.match_clause(clause, tuple, binding.clone()) {
                    // Check additional conditions on the clause
                    if self.check_clause_conditions(clause, &new_binding) {
                        results.push(new_binding);
                    }
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
                ClauseArg::Expr(_expr) => {
                    // TODO: evaluate expression and compare
                    // For now, just skip
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
            // TODO: properly evaluate generator expression
            // For now, handle simple range patterns
            if let Some(values) = self.eval_generator_values(generator, &binding) {
                for value in values {
                    let mut new_binding = binding.clone();
                    // Bind the first variable to the value
                    if let Some(var) = generator.vars.first() {
                        new_binding.insert(var.clone(), value);
                    }
                    results.push(new_binding);
                }
            }
        }

        results
    }

    /// Evaluate generator to produce values.
    fn eval_generator_values(
        &self,
        _generator: &ascent_ir::Generator,
        _bindings: &Bindings,
    ) -> Option<Vec<Value>> {
        // TODO: implement proper expression evaluation
        // For now, return None (generator not supported yet)
        None
    }

    /// Process a condition filter.
    fn process_condition(&self, cond: &Condition, bindings: Vec<Bindings>) -> Vec<Bindings> {
        bindings
            .into_iter()
            .filter(|b| self.eval_condition(cond, b))
            .collect()
    }

    /// Evaluate a condition.
    fn eval_condition(&self, cond: &Condition, _bindings: &Bindings) -> bool {
        match cond {
            Condition::If(_expr) => {
                // TODO: evaluate expression
                true
            }
            Condition::IfLet {
                pattern: _,
                expr: _,
            } => {
                // TODO: evaluate pattern match
                true
            }
            Condition::Let {
                pattern: _,
                expr: _,
            } => {
                // TODO: evaluate let binding
                true
            }
        }
    }

    /// Evaluate a head clause to produce a tuple.
    fn eval_head_tuple(&self, head: &ascent_ir::HeadClause, bindings: &Bindings) -> Option<Tuple> {
        let mut tuple = Vec::with_capacity(head.args.len());

        for arg in &head.args {
            // Try to evaluate the expression
            if let Some(value) = self.eval_expr_simple(arg, bindings) {
                tuple.push(value);
            } else {
                return None;
            }
        }

        Some(tuple)
    }

    /// Simple expression evaluation (variables only for now).
    fn eval_expr_simple(&self, expr: &syn::Expr, bindings: &Bindings) -> Option<Value> {
        match expr {
            syn::Expr::Path(p) => {
                if let Some(ident) = p.path.get_ident() {
                    bindings.get(&ident.to_string()).cloned()
                } else {
                    None
                }
            }
            syn::Expr::Lit(lit) => self.eval_lit(&lit.lit),
            _ => None,
        }
    }

    /// Evaluate a literal.
    fn eval_lit(&self, lit: &syn::Lit) -> Option<Value> {
        match lit {
            syn::Lit::Int(i) => {
                let v: i64 = i.base10_parse().ok()?;
                Some(Value::I64(v))
            }
            syn::Lit::Bool(b) => Some(Value::Bool(b.value)),
            syn::Lit::Str(s) => Some(Value::string(s.value())),
            syn::Lit::Char(c) => Some(Value::Char(c.value())),
            _ => None,
        }
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

    #[test]
    fn test_simple_fact() {
        let mut engine = run_program(
            r#"
            relation edge(i32, i32);
            edge(1, 2);
        "#,
        );

        // Manually insert the fact since we don't evaluate literals in facts yet
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);

        assert!(
            engine
                .relation("edge")
                .unwrap()
                .contains(&vec![Value::I32(1), Value::I32(2)])
        );
    }

    #[test]
    fn test_transitive_closure() {
        let input = r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#;

        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        // Insert initial edges
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
        engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
        engine.insert("edge", vec![Value::I32(3), Value::I32(4)]);

        // Run to fixpoint
        engine.run(&program);

        // Check paths
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
    fn test_join() {
        let input = r#"
            relation r(i32, i32);
            relation s(i32, i32);
            relation joined(i32, i32, i32);
            joined(a, b, c) <-- r(a, b), s(b, c);
        "#;

        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        engine.insert("r", vec![Value::I32(1), Value::I32(2)]);
        engine.insert("r", vec![Value::I32(3), Value::I32(4)]);
        engine.insert("s", vec![Value::I32(2), Value::I32(5)]);
        engine.insert("s", vec![Value::I32(4), Value::I32(6)]);

        engine.run(&program);

        let joined = engine.relation("joined").unwrap();
        assert!(joined.contains(&vec![Value::I32(1), Value::I32(2), Value::I32(5)]));
        assert!(joined.contains(&vec![Value::I32(3), Value::I32(4), Value::I32(6)]));
        assert_eq!(joined.len(), 2);
    }
}
