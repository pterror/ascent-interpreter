//! Semi-naive evaluation engine.

use std::cell::RefCell;
use std::fmt;

use ascent_ir::{Aggregation, BodyItem, Clause, ClauseArg, Condition, Program, Rule};
use petgraph::algo::{condensation, toposort};
use petgraph::graph::DiGraph;
use rustc_hash::FxHashMap;

use crate::aggregators::apply_aggregator;
use crate::expr::{eval_expr, eval_expr_with_registry, expand_range};
use crate::relation::RelationStorage;
use crate::value::{DynValue, Tuple, Value};

/// Interned variable identifier (u32 index instead of String).
pub type VarId = u32;

/// Variable bindings during rule evaluation (u32-keyed with FxHash for fast clone/lookup).
pub type Bindings = FxHashMap<VarId, Value>;

/// Undo log for rolling back in-place binding modifications.
///
/// Each entry records the VarId that was modified and its previous value
/// (`None` if the key was freshly inserted, `Some(old)` if it was overwritten).
type UndoLog = Vec<(VarId, Option<Value>)>;

/// Rollback bindings to a checkpoint, undoing all insertions since that point.
fn rollback(bindings: &mut Bindings, undo: &mut UndoLog, checkpoint: usize) {
    for (id, old_val) in undo.drain(checkpoint..).rev() {
        match old_val {
            Some(v) => {
                bindings.insert(id, v);
            }
            None => {
                bindings.remove(&id);
            }
        }
    }
}

/// Maps variable name strings to compact u32 identifiers.
///
/// Uses interior mutability so `intern` can be called from `&self` methods
/// without conflicting with other Engine borrows.
#[derive(Debug, Default)]
pub struct VarInterner {
    ids: RefCell<FxHashMap<String, VarId>>,
    next_id: std::cell::Cell<VarId>,
}

impl VarInterner {
    /// Intern a variable name, returning its stable u32 id.
    pub fn intern(&self, name: &str) -> VarId {
        let ids = self.ids.borrow();
        if let Some(&id) = ids.get(name) {
            return id;
        }
        drop(ids);
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        self.ids.borrow_mut().insert(name.to_string(), id);
        id
    }
}

/// Constructor function for a custom type: takes arguments, returns a Value.
pub type ValueConstructor = Box<dyn Fn(&[Value]) -> Option<Value> + Send + Sync>;

/// Destructor function for a custom type: takes a Value, returns its fields.
pub type ValueDestructor = Box<dyn Fn(&Value) -> Option<Vec<Value>> + Send + Sync>;

/// Registry of user-defined types and their constructors/destructors.
pub struct TypeRegistry {
    constructors: FxHashMap<String, ValueConstructor>,
    destructors: FxHashMap<String, ValueDestructor>,
}

impl TypeRegistry {
    /// Create an empty type registry.
    pub fn new() -> Self {
        TypeRegistry {
            constructors: FxHashMap::default(),
            destructors: FxHashMap::default(),
        }
    }

    /// Look up a constructor by name.
    pub fn get(&self, name: &str) -> Option<&ValueConstructor> {
        self.constructors.get(name)
    }

    /// Look up a destructor by name.
    pub fn destructor(&self, name: &str) -> Option<&ValueDestructor> {
        self.destructors.get(name)
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TypeRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeRegistry")
            .field("types", &self.constructors.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// The evaluation engine state.
#[derive(Debug)]
pub struct Engine {
    /// Storage for each relation.
    relations: FxHashMap<String, RelationStorage>,
    /// Registry of custom type constructors.
    pub type_registry: TypeRegistry,
    /// Intern table for variable names.
    pub(crate) var_interner: VarInterner,
}

impl Engine {
    /// Create a new engine from a program.
    pub fn new(program: &Program) -> Self {
        let mut relations = FxHashMap::default();

        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            relations.insert(
                name.clone(),
                RelationStorage::with_lattice(arity, rel.is_lattice),
            );
        }

        Engine {
            relations,
            type_registry: TypeRegistry::new(),
            var_interner: VarInterner::default(),
        }
    }

    /// Register a custom type with constructor and destructor.
    ///
    /// The `name` is what appears in Ascent source (e.g., `MyPoint`).
    /// The `constructor` takes a slice of `Value` arguments and returns a
    /// `Value::Custom` wrapping the user's type.
    /// The `destructor` takes a `Value` and returns its fields if it matches
    /// the type, enabling pattern destructuring like `if let MyPoint(x, y) = expr`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// engine.register_type(
    ///     "MyPoint",
    ///     |args| {
    ///         let x = args.get(0)?.as_i32()?;
    ///         let y = args.get(1)?.as_i32()?;
    ///         Some(Value::custom(MyPoint { x, y }))
    ///     },
    ///     |val| {
    ///         let p = Engine::downcast_custom::<MyPoint>(val)?;
    ///         Some(vec![Value::I32(p.x), Value::I32(p.y)])
    ///     },
    /// );
    /// ```
    pub fn register_type(
        &mut self,
        name: &str,
        constructor: impl Fn(&[Value]) -> Option<Value> + Send + Sync + 'static,
        destructor: impl Fn(&Value) -> Option<Vec<Value>> + Send + Sync + 'static,
    ) {
        let name = name.to_string();
        self.type_registry
            .constructors
            .insert(name.clone(), Box::new(constructor));
        self.type_registry
            .destructors
            .insert(name, Box::new(destructor));
    }

    /// Downcast a custom value to a concrete type.
    pub fn downcast_custom<T: DynValue + 'static>(value: &Value) -> Option<&T> {
        if let Value::Custom(v) = value {
            v.as_any().downcast_ref::<T>()
        } else {
            None
        }
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
            let final_bindings = self.process_body(&rule.body, vec![Bindings::default()], None);
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
                    self.process_body(&rule.body, vec![Bindings::default()], Some(idx));
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
    ///
    /// Uses in-place binding modification with undo log to avoid cloning
    /// the bindings map for each tuple attempted. Only clones on match success.
    /// When multiple columns are bound, picks the most selective index and
    /// pre-filters remaining bound columns via direct comparison.
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
        let mut undo = UndoLog::new();

        for binding in &bindings {
            let mut work = binding.clone();
            let bound_cols = self.find_bound_columns(clause, binding);

            if !bound_cols.is_empty() {
                // Pick the most selective column for index lookup (smallest result set)
                let (primary_pos, _) = bound_cols
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, (col, val))| rel.lookup(*col, val).len())
                    .unwrap();
                let (primary_col, primary_val) = &bound_cols[primary_pos];
                let indices = rel.lookup(*primary_col, primary_val);

                for &idx in indices {
                    if use_recent && !rel.is_recent(idx) {
                        continue;
                    }
                    let tuple = rel.get(idx);
                    // Pre-filter: check remaining bound columns via direct comparison
                    if bound_cols.len() > 1 {
                        let pre_match = bound_cols
                            .iter()
                            .enumerate()
                            .all(|(i, (col, val))| i == primary_pos || tuple[*col] == *val);
                        if !pre_match {
                            continue;
                        }
                    }
                    let cp = undo.len();
                    if self.match_clause(clause, tuple, &mut work, &mut undo)
                        && self.check_clause_conditions(clause, &mut work, &mut undo)
                    {
                        results.push(work.clone());
                    }
                    rollback(&mut work, &mut undo, cp);
                }
            } else {
                // No bound columns: full scan
                let iter: Box<dyn Iterator<Item = &[Value]>> = if use_recent {
                    Box::new(rel.iter_recent())
                } else {
                    Box::new(rel.iter_full())
                };
                for tuple in iter {
                    let cp = undo.len();
                    if self.match_clause(clause, tuple, &mut work, &mut undo)
                        && self.check_clause_conditions(clause, &mut work, &mut undo)
                    {
                        results.push(work.clone());
                    }
                    rollback(&mut work, &mut undo, cp);
                }
            }
        }

        results
    }

    /// Find all clause columns that are already bound in the given bindings.
    ///
    /// Returns a list of (column_index, value) pairs for columns where
    /// the clause arg is a bound variable or evaluable expression.
    fn find_bound_columns(&self, clause: &Clause, binding: &Bindings) -> Vec<(usize, Value)> {
        let mut bound = Vec::new();
        for (col, arg) in clause.args.iter().enumerate() {
            match arg {
                ClauseArg::Var(var) => {
                    let var_id = self.var_interner.intern(var);
                    if let Some(val) = binding.get(&var_id) {
                        bound.push((col, val.clone()));
                    }
                }
                ClauseArg::Expr(expr) => {
                    if let Some(val) = eval_expr_with_registry(
                        expr,
                        binding,
                        &self.type_registry,
                        &self.var_interner,
                    ) {
                        bound.push((col, val));
                    }
                }
            }
        }
        bound
    }

    /// Try to match a clause against a tuple, extending bindings in place.
    ///
    /// Returns `true` if the match succeeded (bindings extended with new vars).
    /// On failure, rolls back any partial insertions via the undo log.
    fn match_clause(
        &self,
        clause: &Clause,
        tuple: &[Value],
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        if clause.args.len() != tuple.len() {
            return false;
        }

        let cp = undo.len();
        for (arg, value) in clause.args.iter().zip(tuple.iter()) {
            match arg {
                ClauseArg::Var(var) => {
                    let var_id = self.var_interner.intern(var);
                    if let Some(existing) = bindings.get(&var_id) {
                        if existing != value {
                            rollback(bindings, undo, cp);
                            return false;
                        }
                    } else {
                        bindings.insert(var_id, value.clone());
                        undo.push((var_id, None));
                    }
                }
                ClauseArg::Expr(expr) => {
                    if let Some(evaluated) = eval_expr_with_registry(
                        expr,
                        bindings,
                        &self.type_registry,
                        &self.var_interner,
                    ) && evaluated != *value
                    {
                        rollback(bindings, undo, cp);
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Check additional conditions on a clause, potentially extending bindings in place.
    fn check_clause_conditions(
        &self,
        clause: &Clause,
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        for cond in &clause.conditions {
            if !self.eval_condition(cond, bindings, undo) {
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
            if let Some(range_val) = eval_expr_with_registry(
                &generator.expr,
                &binding,
                &self.type_registry,
                &self.var_interner,
            ) && let Some(values) = expand_range(&range_val)
            {
                for value in values {
                    let mut new_binding = binding.clone();
                    if let Some(var) = generator.vars.first() {
                        let var_id = self.var_interner.intern(var);
                        new_binding.insert(var_id, value);
                    }
                    results.push(new_binding);
                }
            }
        }

        results
    }

    /// Process an aggregation clause.
    ///
    /// Uses in-place binding modification with undo log to avoid cloning
    /// the bindings map for each relation tuple during aggregation matching.
    fn process_aggregation(&self, agg: &Aggregation, bindings: Vec<Bindings>) -> Vec<Bindings> {
        let Some(rel) = self.relations.get(&agg.relation) else {
            return vec![];
        };

        // Resolve aggregator name from the expression
        let agg_name = resolve_aggregator_name(&agg.aggregator);

        let mut results = Vec::new();

        // Pre-intern bound var ids once per aggregation
        let bound_var_ids: Vec<VarId> = agg
            .bound_vars
            .iter()
            .map(|var| self.var_interner.intern(var))
            .collect();

        let mut undo = UndoLog::new();

        for binding in &bindings {
            // Collect bound-variable tuples from matching relation rows.
            // Uses in-place matching with rollback to avoid per-tuple clones.
            let mut work = binding.clone();
            let mut collected: Vec<Vec<Value>> = Vec::new();

            for tuple in rel.iter_full() {
                let cp = undo.len();
                if self.match_agg_args(agg, tuple, &mut work, &mut undo) {
                    collected.push(
                        bound_var_ids
                            .iter()
                            .filter_map(|var_id| work.get(var_id).cloned())
                            .collect(),
                    );
                }
                rollback(&mut work, &mut undo, cp);
            }

            // Apply aggregator (streaming over the collected slice)
            let agg_results = apply_aggregator(&agg_name, collected.iter().map(|v| v.as_slice()));

            // Bind results to output pattern variables
            for result_tuple in agg_results {
                let mut new_binding = binding.clone();
                for (var, val) in agg.result_vars.iter().zip(result_tuple) {
                    let var_id = self.var_interner.intern(var);
                    new_binding.insert(var_id, val);
                }
                results.push(new_binding);
            }
        }

        results
    }

    /// Match aggregation relation arguments against a tuple in place.
    ///
    /// Returns `true` if the match succeeded (bindings extended with new vars).
    /// On failure, rolls back any partial insertions via the undo log.
    fn match_agg_args(
        &self,
        agg: &Aggregation,
        tuple: &[Value],
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        if agg.args.len() != tuple.len() {
            return false;
        }

        let cp = undo.len();
        for (arg_expr, value) in agg.args.iter().zip(tuple.iter()) {
            // Check if the argument is a simple variable (bound var)
            if let syn::Expr::Path(p) = arg_expr
                && let Some(ident) = p.path.get_ident()
            {
                let var_id = self.var_interner.intern(&ident.to_string());
                if let Some(existing) = bindings.get(&var_id) {
                    if existing != value {
                        rollback(bindings, undo, cp);
                        return false;
                    }
                } else {
                    bindings.insert(var_id, value.clone());
                    undo.push((var_id, None));
                }
                continue;
            }

            // Otherwise evaluate and compare
            if let Some(evaluated) =
                eval_expr_with_registry(arg_expr, bindings, &self.type_registry, &self.var_interner)
                && evaluated != *value
            {
                rollback(bindings, undo, cp);
                return false;
            }
        }

        true
    }

    /// Process a condition, potentially binding new variables.
    fn process_condition(&self, cond: &Condition, mut bindings: Vec<Bindings>) -> Vec<Bindings> {
        let mut undo = UndoLog::new();
        bindings.retain_mut(|b| {
            undo.clear();
            self.eval_condition(cond, b, &mut undo)
            // No rollback needed: failed bindings are dropped by retain_mut
        });
        bindings
    }

    /// Evaluate a condition in place. Returns `true` on success (bindings may be extended).
    fn eval_condition(
        &self,
        cond: &Condition,
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        match cond {
            Condition::If(expr) => {
                eval_expr_with_registry(expr, bindings, &self.type_registry, &self.var_interner)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            }
            Condition::IfLet { pattern, expr } | Condition::Let { pattern, expr } => {
                if let Some(value) =
                    eval_expr_with_registry(expr, bindings, &self.type_registry, &self.var_interner)
                {
                    match_pattern(
                        pattern,
                        &value,
                        bindings,
                        Some(&self.type_registry),
                        &self.var_interner,
                        undo,
                    )
                } else {
                    false
                }
            }
        }
    }

    /// Evaluate a head clause to produce a tuple.
    fn eval_head_tuple(&self, head: &ascent_ir::HeadClause, bindings: &Bindings) -> Option<Tuple> {
        let mut tuple = Vec::with_capacity(head.args.len());

        for arg in &head.args {
            if let Some(value) =
                eval_expr_with_registry(arg, bindings, &self.type_registry, &self.var_interner)
            {
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
    let mut producers: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
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
///
/// Uses the undo log to track insertions for rollback on partial match failure.
fn match_pattern(
    pat: &syn::Pat,
    value: &Value,
    bindings: &mut Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
    undo: &mut UndoLog,
) -> bool {
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
                && !match_pattern(sub_pat, value, bindings, registry, interner, undo)
            {
                return false;
            }
            let var_id = interner.intern(&name);
            let old = bindings.insert(var_id, value.clone());
            undo.push((var_id, old));
            true
        }

        // Literal pattern: `42`, `true`, `'a'`
        syn::Pat::Lit(expr_lit) => {
            let expr = syn::Expr::Lit(expr_lit.clone());
            if let Some(lit_val) = eval_expr(&expr, bindings, interner) {
                lit_val == *value
            } else {
                false
            }
        }

        // Reference pattern: `&x` — in Datalog context, match the inner pattern
        syn::Pat::Reference(r) => match_pattern(&r.pat, value, bindings, registry, interner, undo),

        // Tuple pattern: `(a, b, c)`
        syn::Pat::Tuple(tuple) => {
            if let Value::Tuple(vals) = value {
                if vals.len() != tuple.elems.len() {
                    return false;
                }
                let cp = undo.len();
                for (pat_elem, val) in tuple.elems.iter().zip(vals.iter()) {
                    if !match_pattern(pat_elem, val, bindings, registry, interner, undo) {
                        rollback(bindings, undo, cp);
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }

        // Tuple struct pattern: `Some(x)`, `Dual(x)`, or custom types
        syn::Pat::TupleStruct(ts) => {
            let path_str = path_to_string(&ts.path);
            match path_str.as_str() {
                "Some" => {
                    if let Value::Option(Some(inner)) = value
                        && ts.elems.len() == 1
                    {
                        match_pattern(&ts.elems[0], inner, bindings, registry, interner, undo)
                    } else {
                        false
                    }
                }
                "Dual" => {
                    if let Value::Dual(inner) = value
                        && ts.elems.len() == 1
                    {
                        match_pattern(&ts.elems[0], inner, bindings, registry, interner, undo)
                    } else {
                        false
                    }
                }
                "None" => matches!(value, Value::Option(None)),
                _ => {
                    if let Some(reg) = registry
                        && let Some(destructor) = reg.destructor(&path_str)
                        && let Some(fields) = destructor(value)
                        && fields.len() == ts.elems.len()
                    {
                        let cp = undo.len();
                        for (pat_elem, val) in ts.elems.iter().zip(fields.iter()) {
                            if !match_pattern(pat_elem, val, bindings, registry, interner, undo) {
                                rollback(bindings, undo, cp);
                                return false;
                            }
                        }
                        true
                    } else {
                        false
                    }
                }
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
            let cp = undo.len();
            for case in &or_pat.cases {
                if match_pattern(case, value, bindings, registry, interner, undo) {
                    return true;
                }
                rollback(bindings, undo, cp);
            }
            false
        }

        // Parenthesized: `(pattern)`
        syn::Pat::Paren(p) => match_pattern(&p.pat, value, bindings, registry, interner, undo),

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

#[cfg(feature = "serde")]
impl Engine {
    /// Register a custom type using serde for automatic constructor/destructor.
    ///
    /// This is a convenience wrapper around [`Engine::register_type`] that uses
    /// serde's `Serialize`/`Deserialize` to automatically convert between
    /// `T` and `Vec<Value>`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
    ///          serde::Serialize, serde::Deserialize)]
    /// struct Point { x: i32, y: i32 }
    ///
    /// engine.register_serde_type::<Point>("Point");
    /// ```
    pub fn register_serde_type<T>(&mut self, name: &str)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + DynValue + 'static,
    {
        use crate::serde_bridge::{from_values, to_values};
        self.register_type(
            name,
            |args| from_values::<T>(args).ok().map(Value::custom),
            |val| Engine::downcast_custom::<T>(val).and_then(|t| to_values(t).ok()),
        );
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
        assert!(edge.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(edge.contains(&[Value::I32(2), Value::I32(3)]));
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
        assert!(path.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(path.contains(&[Value::I32(1), Value::I32(3)]));
        assert!(path.contains(&[Value::I32(1), Value::I32(4)]));
        assert!(path.contains(&[Value::I32(2), Value::I32(3)]));
        assert!(path.contains(&[Value::I32(2), Value::I32(4)]));
        assert!(path.contains(&[Value::I32(3), Value::I32(4)]));
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
        assert!(joined.contains(&[Value::I32(1), Value::I32(2), Value::I32(5)]));
        assert!(joined.contains(&[Value::I32(3), Value::I32(4), Value::I32(6)]));
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
        assert!(even.contains(&[Value::I32(2)]));
        assert!(even.contains(&[Value::I32(4)]));
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
        assert!(number.contains(&[Value::I32(0)]));
        assert!(number.contains(&[Value::I32(4)]));
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
        assert!(doubled.contains(&[Value::I32(2)]));
        assert!(doubled.contains(&[Value::I32(4)]));
        assert!(doubled.contains(&[Value::I32(6)]));
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
        assert!(small_even.contains(&[Value::I32(0)]));
        assert!(small_even.contains(&[Value::I32(8)]));
        assert!(!small_even.contains(&[Value::I32(10)]));
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
        assert!(fib.contains(&[Value::I32(0), Value::I32(1)]));
        assert!(fib.contains(&[Value::I32(1), Value::I32(1)]));
        assert!(fib.contains(&[Value::I32(2), Value::I32(2)]));
        assert!(fib.contains(&[Value::I32(3), Value::I32(3)]));
        assert!(fib.contains(&[Value::I32(4), Value::I32(5)]));
        assert!(fib.contains(&[Value::I32(5), Value::I32(8)]));
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
        assert!(lowest.contains(&[Value::I32(1)]));
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
        assert!(highest.contains(&[Value::I32(8)]));
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
        assert!(total.contains(&[Value::I32(6)]));
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
        assert!(card.contains(&[Value::I32(3)]));
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
        assert!(only_a.contains(&[Value::I32(1)]));
        assert!(only_a.contains(&[Value::I32(3)]));
        assert!(!only_a.contains(&[Value::I32(2)]));
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
        assert!(fib_max.contains(&[Value::I32(89)]));
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
        assert!(unreachable.contains(&[Value::I32(4)]));
        assert!(unreachable.contains(&[Value::I32(1)]));
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
                .contains(&[Value::I32(6)])
        );
        // max_dest: 1→4, 2→4, 3→4
        let max_dest = engine.relation("max_dest").unwrap();
        assert!(max_dest.contains(&[Value::I32(1), Value::I32(4)]));
        assert!(max_dest.contains(&[Value::I32(2), Value::I32(4)]));
        assert!(max_dest.contains(&[Value::I32(3), Value::I32(4)]));
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
                .contains(&[Value::I32(60)])
        );
        // 60 > 50, so is_high should have (60)
        assert!(
            engine
                .relation("is_high")
                .unwrap()
                .contains(&[Value::I32(60)])
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
        assert!(best.contains(&[Value::I32(1), Value::I32(20)]));
        assert!(best.contains(&[Value::I32(2), Value::I32(30)]));
        assert_eq!(best.len(), 2);

        // overall_best: max(20, 30) = 30
        assert!(
            engine
                .relation("overall_best")
                .unwrap()
                .contains(&[Value::I32(30)])
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
        assert!(isolated.contains(&[Value::I32(4)]));
        assert!(isolated.contains(&[Value::I32(5)]));
        assert_eq!(isolated.len(), 2);
    }

    // ─── Lattice tests ──────────────────────────────────────────────

    #[test]
    fn test_lattice_max_value() {
        // Lattice relation keeps the max value per key
        let engine = run_program(
            r#"
            lattice best_score(i32, i32);
            best_score(1, 10);
            best_score(1, 20);
            best_score(1, 5);
            best_score(2, 30);
            best_score(2, 15);
        "#,
        );

        let best = engine.relation("best_score").unwrap();
        // Key 1: max(10, 20, 5) = 20
        assert!(best.contains(&[Value::I32(1), Value::I32(20)]));
        // Key 2: max(30, 15) = 30
        assert!(best.contains(&[Value::I32(2), Value::I32(30)]));
        assert_eq!(best.len(), 2);
    }

    #[test]
    fn test_lattice_dual_shortest_path() {
        // Shortest path using Dual (reverses ordering so join = min)
        let engine = run_program(
            r#"
            relation edge(i32, i32, i32);
            lattice shortest(i32, i32, Dual<i32>);

            edge(1, 2, 1);
            edge(2, 3, 2);
            edge(1, 3, 10);

            shortest(x, y, Dual(*w)) <-- edge(x, y, w);
            shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
        "#,
        );

        let sp = engine.relation("shortest").unwrap();
        // 1→2: direct = 1
        assert!(sp.contains(&[
            Value::I32(1),
            Value::I32(2),
            Value::Dual(Box::new(Value::I32(1)))
        ]));
        // 2→3: direct = 2
        assert!(sp.contains(&[
            Value::I32(2),
            Value::I32(3),
            Value::Dual(Box::new(Value::I32(2)))
        ]));
        // 1→3: min(10, 1+2) = 3
        assert!(sp.contains(&[
            Value::I32(1),
            Value::I32(3),
            Value::Dual(Box::new(Value::I32(3)))
        ]));
        assert_eq!(sp.len(), 3);
    }

    #[test]
    fn test_lattice_recursive_max() {
        // Lattice with recursive rule: max propagates through edges
        let engine = run_program(
            r#"
            relation edge(i32, i32);
            relation source_val(i32, i32);
            lattice max_reach(i32, i32);

            edge(1, 2);
            edge(2, 3);
            edge(3, 4);
            source_val(1, 100);
            source_val(2, 50);
            source_val(3, 75);

            max_reach(x, v) <-- source_val(x, v);
            max_reach(y, v) <-- edge(x, y), max_reach(x, v);
        "#,
        );

        let mr = engine.relation("max_reach").unwrap();
        // Node 1: 100 (own value only)
        assert!(mr.contains(&[Value::I32(1), Value::I32(100)]));
        // Node 2: max(50, 100) = 100 (from node 1)
        assert!(mr.contains(&[Value::I32(2), Value::I32(100)]));
        // Node 3: max(75, 100) = 100 (propagated through)
        assert!(mr.contains(&[Value::I32(3), Value::I32(100)]));
        // Node 4: max(100) = 100 (propagated from node 3)
        assert!(mr.contains(&[Value::I32(4), Value::I32(100)]));
        assert_eq!(mr.len(), 4);
    }

    #[test]
    fn test_lattice_with_pattern_match() {
        // Pattern matching with ?Dual(x) to extract inner value
        let engine = run_program(
            r#"
            lattice best(i32, Dual<i32>);
            relation result(i32, i32);

            best(1, Dual(10));
            best(1, Dual(5));
            best(2, Dual(20));
            best(2, Dual(3));

            result(k, v) <-- best(k, ?Dual(v));
        "#,
        );

        let best = engine.relation("best").unwrap();
        // Dual join = min: Dual(min(10, 5)) = Dual(5)
        assert!(best.contains(&[Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));
        assert!(best.contains(&[Value::I32(2), Value::Dual(Box::new(Value::I32(3)))]));

        let result = engine.relation("result").unwrap();
        assert!(result.contains(&[Value::I32(1), Value::I32(5)]));
        assert!(result.contains(&[Value::I32(2), Value::I32(3)]));
    }

    // ─── Custom type (BYOD) tests ──────────────────────────────────

    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct Point {
        x: i32,
        y: i32,
    }

    impl std::fmt::Display for Point {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "({}, {})", self.x, self.y)
        }
    }

    #[test]
    fn test_custom_type_in_relation() {
        // Insert custom values directly and query them
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation point(i32, i32);
            relation has_point(i32);
            has_point(id) <-- point(id, _);
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        engine.insert(
            "point",
            vec![Value::I32(1), Value::custom(Point { x: 10, y: 20 })],
        );
        engine.insert(
            "point",
            vec![Value::I32(2), Value::custom(Point { x: 30, y: 40 })],
        );
        engine.run(&program);

        let hp = engine.relation("has_point").unwrap();
        assert!(hp.contains(&[Value::I32(1)]));
        assert!(hp.contains(&[Value::I32(2)]));
        assert_eq!(hp.len(), 2);
    }

    #[test]
    fn test_custom_type_equality_join() {
        // Two relations sharing custom values can join on equality
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation r(i32, i32);
            relation s(i32, i32);
            relation joined(i32, i32);
            joined(a, c) <-- r(a, b), s(b, c);
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        let p = Value::custom(Point { x: 1, y: 2 });
        engine.insert("r", vec![Value::I32(1), p.clone()]);
        engine.insert("s", vec![p, Value::I32(99)]);
        engine.run(&program);

        let joined = engine.relation("joined").unwrap();
        assert!(joined.contains(&[Value::I32(1), Value::I32(99)]));
        assert_eq!(joined.len(), 1);
    }

    #[test]
    fn test_register_type_constructor() {
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation data(i32, i32);
            data(1, Point(10, 20));
            data(2, Point(30, 40));
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        engine.register_type(
            "Point",
            |args| {
                let x = args.first()?.as_i32()?;
                let y = args.get(1)?.as_i32()?;
                Some(Value::custom(Point { x, y }))
            },
            |val| {
                let p = Engine::downcast_custom::<Point>(val)?;
                Some(vec![Value::I32(p.x), Value::I32(p.y)])
            },
        );

        engine.run(&program);

        let data = engine.relation("data").unwrap();
        assert_eq!(data.len(), 2);
        assert!(data.contains(&[Value::I32(1), Value::custom(Point { x: 10, y: 20 })]));
        assert!(data.contains(&[Value::I32(2), Value::custom(Point { x: 30, y: 40 })]));
    }

    #[test]
    fn test_downcast_custom() {
        let v = Value::custom(Point { x: 5, y: 10 });
        let point = Engine::downcast_custom::<Point>(&v).unwrap();
        assert_eq!(point.x, 5);
        assert_eq!(point.y, 10);
    }

    // ─── Custom type destructuring tests ──────────────────────────────

    fn register_point(engine: &mut Engine) {
        engine.register_type(
            "Point",
            |args| {
                let x = args.first()?.as_i32()?;
                let y = args.get(1)?.as_i32()?;
                Some(Value::custom(Point { x, y }))
            },
            |val| {
                let p = Engine::downcast_custom::<Point>(val)?;
                Some(vec![Value::I32(p.x), Value::I32(p.y)])
            },
        );
    }

    #[test]
    fn test_custom_type_if_let_destructure() {
        // Destructure a custom type via `if let Point(x, y) = expr`
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation data(i32, i32);
            relation coords(i32, i32, i32);
            coords(id, x, y) <-- data(id, p), if let Point(x, y) = p;
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 10, y: 20 })],
        );
        engine.insert(
            "data",
            vec![Value::I32(2), Value::custom(Point { x: 30, y: 40 })],
        );
        engine.run(&program);

        let coords = engine.relation("coords").unwrap();
        assert_eq!(coords.len(), 2);
        assert!(coords.contains(&[Value::I32(1), Value::I32(10), Value::I32(20)]));
        assert!(coords.contains(&[Value::I32(2), Value::I32(30), Value::I32(40)]));
    }

    #[test]
    fn test_custom_type_clause_pattern_arg() {
        // Use `?Point(x, y)` in clause body (desugared to if-let by parser)
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation data(i32, i32);
            relation coords(i32, i32, i32);
            coords(id, x, y) <-- data(id, ?Point(x, y));
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 5, y: 15 })],
        );
        engine.run(&program);

        let coords = engine.relation("coords").unwrap();
        assert_eq!(coords.len(), 1);
        assert!(coords.contains(&[Value::I32(1), Value::I32(5), Value::I32(15)]));
    }

    #[test]
    fn test_custom_type_destructure_with_wildcard() {
        // Destructure with wildcard: `if let Point(x, _) = p`
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation data(i32, i32);
            relation x_only(i32, i32);
            x_only(id, x) <-- data(id, p), if let Point(x, _) = p;
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 7, y: 99 })],
        );
        engine.run(&program);

        let x_only = engine.relation("x_only").unwrap();
        assert_eq!(x_only.len(), 1);
        assert!(x_only.contains(&[Value::I32(1), Value::I32(7)]));
    }

    #[test]
    fn test_custom_type_destructure_wrong_type() {
        // Destructuring a non-matching value returns false, doesn't crash
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation data(i32, i32);
            relation coords(i32, i32, i32);
            coords(id, x, y) <-- data(id, p), if let Point(x, y) = p;
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        register_point(&mut engine);

        // Insert a plain i32 instead of a Point — should not match
        engine.insert("data", vec![Value::I32(1), Value::I32(42)]);
        engine.run(&program);

        let coords = engine.relation("coords").unwrap();
        assert_eq!(coords.len(), 0);
    }

    #[test]
    fn test_custom_type_construct_and_destructure() {
        // Round-trip: construct with Point(x, y) in head, destructure in body
        let ast: AscentProgram = syn::parse_str(
            r#"
            relation input(i32, i32, i32);
            relation wrapped(i32, i32);
            relation unwrapped(i32, i32, i32);
            input(1, 10, 20);
            input(2, 30, 40);
            wrapped(id, Point(x, y)) <-- input(id, x, y);
            unwrapped(id, a, b) <-- wrapped(id, p), if let Point(a, b) = p;
            "#,
        )
        .unwrap();
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        register_point(&mut engine);
        engine.run(&program);

        let unwrapped = engine.relation("unwrapped").unwrap();
        assert_eq!(unwrapped.len(), 2);
        assert!(unwrapped.contains(&[Value::I32(1), Value::I32(10), Value::I32(20)]));
        assert!(unwrapped.contains(&[Value::I32(2), Value::I32(30), Value::I32(40)]));
    }

    // ─── Serde registration tests ───────────────────────────────────

    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[derive(
            Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
        )]
        struct Point {
            x: i32,
            y: i32,
        }

        impl fmt::Display for Point {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "({}, {})", self.x, self.y)
            }
        }

        #[test]
        fn test_register_serde_type_round_trip() {
            let ast: AscentProgram = syn::parse_str(
                r#"
                relation input(i32, i32, i32);
                relation wrapped(i32, i32);
                relation unwrapped(i32, i32, i32);
                input(1, 10, 20);
                input(2, 30, 40);
                wrapped(id, Point(x, y)) <-- input(id, x, y);
                unwrapped(id, a, b) <-- wrapped(id, p), if let Point(a, b) = p;
                "#,
            )
            .unwrap();
            let program = Program::from_ast(ast);
            let mut engine = Engine::new(&program);
            engine.register_serde_type::<Point>("Point");
            engine.run(&program);

            let unwrapped = engine.relation("unwrapped").unwrap();
            assert_eq!(unwrapped.len(), 2);
            assert!(unwrapped.contains(&[Value::I32(1), Value::I32(10), Value::I32(20)]));
            assert!(unwrapped.contains(&[Value::I32(2), Value::I32(30), Value::I32(40)]));
        }
    }
}
