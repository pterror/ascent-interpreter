//! Semi-naive evaluation engine.

use std::cell::RefCell;
use std::fmt;
#[cfg(feature = "jit")]
use std::sync::{Arc, Mutex};


use crate::ir::{BodyItem, Program};
use petgraph::algo::{condensation, toposort};
use petgraph::graph::DiGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::eval::aggregators::apply_aggregator;
use crate::eval::compiled::{
    CAggArg, CAggregation, CBinOp, CBodyItem, CClause, CClauseArg, CCondition, CExpr,
    CHeadClause, CRule, compile_rule, eval_cexpr,
};
use crate::eval::expr::expand_range;
use crate::eval::relation::{Relation, SourceId};
use crate::eval::value::{DynValue, Tuple, Value};

/// Interned variable identifier (u32 index instead of String).
pub type VarId = u32;

/// Variable bindings during rule evaluation.
///
/// Uses Vec-indexed slots for O(1) direct access by VarId,
/// replacing the previous FxHashMap-based approach.
#[derive(Debug, Default)]
pub struct Bindings {
    slots: Vec<Option<Value>>,
}

impl Clone for Bindings {
    fn clone(&self) -> Self {
        Bindings {
            slots: self.slots.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.slots.clone_from(&source.slots);
    }
}

impl Bindings {
    /// Create bindings with pre-allocated slots for `n` variables.
    pub fn new(capacity: usize) -> Self {
        Bindings {
            slots: vec![None; capacity],
        }
    }

    /// Look up a variable binding by VarId.
    #[inline]
    pub fn get(&self, var_id: &VarId) -> Option<&Value> {
        self.slots.get(*var_id as usize)?.as_ref()
    }

    /// Insert a variable binding. Returns the previous value if any.
    #[inline]
    pub fn insert(&mut self, var_id: VarId, value: Value) -> Option<Value> {
        let idx = var_id as usize;
        if idx >= self.slots.len() {
            self.slots.resize(idx + 1, None);
        }
        self.slots[idx].replace(value)
    }

    /// Remove a variable binding. Returns the value if it was present.
    #[inline]
    pub fn remove(&mut self, var_id: &VarId) -> Option<Value> {
        let idx = *var_id as usize;
        if idx < self.slots.len() {
            self.slots[idx].take()
        } else {
            None
        }
    }
}

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
        debug_assert!(id < u32::MAX, "VarId overflow");
        self.next_id.set(id + 1);
        self.ids.borrow_mut().insert(name.to_string(), id);
        id
    }

    /// Get the number of interned variables.
    pub fn len(&self) -> usize {
        self.next_id.get() as usize
    }

    /// Check if no variables have been interned.
    pub fn is_empty(&self) -> bool {
        self.next_id.get() == 0
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
    pub fn constructor(&self, name: &str) -> Option<&ValueConstructor> {
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

/// Pre-computed SCC stratification with compiled rules.
///
/// Caches the dependency structure and compiled rules so that
/// `run_incremental` can selectively re-run only affected strata.
#[derive(Debug, Clone)]
struct Stratification {
    /// Rule indices per SCC, in topological order.
    sccs: Vec<Vec<usize>>,
    /// Relations read by each SCC (body clauses + aggregations).
    scc_reads: Vec<FxHashSet<String>>,
    /// Relations written by each SCC (head clauses).
    scc_writes: Vec<FxHashSet<String>>,
    /// Compiled rules (indexed by original rule index in the program).
    compiled: Vec<CRule>,
    /// Whether each SCC is monotone (no negation).
    /// Monotone SCCs can use delta-only incremental evaluation on pure additions.
    scc_is_monotone: Vec<bool>,
}

impl Stratification {
    /// Build stratification from a program and variable interner.
    fn build(program: &Program, interner: &VarInterner) -> Self {
        let sccs = compute_rule_sccs(program);
        let compiled: Vec<CRule> = program
            .rules
            .iter()
            .map(|r| compile_rule(r, interner))
            .collect();

        let mut scc_reads = Vec::with_capacity(sccs.len());
        let mut scc_writes = Vec::with_capacity(sccs.len());

        let mut scc_is_monotone = Vec::with_capacity(sccs.len());

        for scc_indices in &sccs {
            let mut reads = FxHashSet::default();
            let mut writes = FxHashSet::default();
            let mut monotone = true;
            for &rule_idx in scc_indices {
                let rule = &compiled[rule_idx];
                for item in &rule.body {
                    match item {
                        CBodyItem::Clause(c) => {
                            reads.insert(c.relation.clone());
                        }
                        CBodyItem::Aggregation(a) => {
                            reads.insert(a.relation.clone());
                            if a.aggregator_name == "not" {
                                monotone = false;
                            }
                        }
                        _ => {}
                    }
                }
                for head in &rule.heads {
                    writes.insert(head.relation.clone());
                }
            }
            scc_reads.push(reads);
            scc_writes.push(writes);
            scc_is_monotone.push(monotone);
        }

        Stratification {
            sccs,
            scc_reads,
            scc_writes,
            compiled,
            scc_is_monotone,
        }
    }
}

/// Relation-name metadata for repopulating a pooled `StratumStage4Runtime`.
/// Parallel arrays match indices of the corresponding pointer arrays in the runtime.
#[cfg(all(feature = "jit", feature = "specialized"))]
pub(crate) struct StratumStage4RefreshInfo {
    /// Relation name for each slot in `_all_rels`.
    all_rel_names: Box<[String]>,
    /// Relation name for each slot in `_lookup_specs` (parallel to specs).
    lookup_spec_rel_names: Box<[String]>,
    /// Per-rule: name for each clause (+ negation + aggregation) rel slot.
    clause_rel_names: Vec<Box<[String]>>,
    /// Per-rule: name for each head rel slot.
    head_rel_names: Vec<Box<[String]>>,
}

/// Pinned runtime data for a Stage 4 stratum function (inlined rule bodies).
/// All raw pointers in `stage4_ctx` point into the boxes below.
#[cfg(all(feature = "jit", feature = "specialized"))]
#[allow(dead_code, clippy::vec_box)]
pub(crate) struct StratumStage4Runtime {
    stage4_ctx: Box<crate::eval::jit::packed_helpers::StratumStage4Ctx>,
    _per_rule_clause_rels: Vec<Box<[*const crate::eval::specialized::PackedStorage]>>,
    _per_rule_head_rels: Vec<Box<[*mut crate::eval::specialized::PackedStorage]>>,
    _per_rule_ctxs: Vec<Box<crate::eval::jit::packed_helpers::PackedJitContextV3>>,
    _rule_ctx_ptrs: Box<[*mut crate::eval::jit::packed_helpers::PackedJitContextV3]>,
    _all_rels: Box<[*mut crate::eval::specialized::PackedStorage]>,
    /// Per-rule dedup handle pointer arrays (one *mut JitDedupHandle per head relation per rule).
    _per_rule_dedup_handles: Vec<Box<[*mut crate::eval::jit_index::JitDedupHandle]>>,
    /// Inline hash-probe lookup handles; stage4_ctx.handles_buf points into this.
    _handles_buf: Box<[crate::eval::jit_index::JitLookupHandle]>,
    /// Lookup specs parallel to _handles_buf; stage4_ctx.lookup_specs points into this.
    _lookup_specs: Box<[crate::eval::jit::packed_helpers::LookupSpec]>,
    /// Optional native fast-path runtime (Step 3). `None` if any relation lacks `jit_native`.
    stage4_native_runtime: Option<crate::eval::jit::packed_helpers::StratumStage4NativeRuntime>,
    /// Present when this runtime was built with pooling support (stage4_native_runtime is None).
    /// Used to repopulate pointer slots for a new engine without re-allocating.
    refresh_info: Option<StratumStage4RefreshInfo>,
}

#[cfg(all(feature = "jit", feature = "specialized"))]
impl std::fmt::Debug for StratumStage4Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StratumStage4Runtime").finish_non_exhaustive()
    }
}

/// The evaluation engine state.
#[derive(Debug)]
pub struct Engine {
    /// The program being evaluated.
    program: Program,
    /// Storage for each relation.
    relations: FxHashMap<String, Relation>,
    /// Declared column types per relation (primitive type name, or None for complex types).
    column_types: FxHashMap<String, Vec<Option<String>>>,
    /// Registry of custom type constructors.
    pub(crate) type_registry: TypeRegistry,
    /// Intern table for variable names.
    pub(crate) var_interner: VarInterner,
    /// Number of interned variables (for pre-allocating Bindings).
    var_count: usize,
    /// Source name → SourceId interning.
    source_names: FxHashMap<String, SourceId>,
    /// Next source ID to allocate (0 is reserved for ANONYMOUS).
    next_source_id: u32,
    /// Active source context — facts from empty-body rules get this tag during `run`.
    current_source: SourceId,
    /// Cached stratification for incremental re-runs.
    stratification: Option<Stratification>,
    /// JIT compiler for rule bodies (optional, feature-gated).
    /// Arc<Mutex> allows sharing a pre-compiled JIT across Engine instances.
    #[cfg(feature = "jit")]
    jit: Option<Arc<Mutex<crate::eval::jit::JitCompiler>>>,
    /// Cache of Stage 4 stratum runtime contexts (inlined rule bodies).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    stratum_stage4_cache: FxHashMap<usize, StratumStage4Runtime>,
    /// Maximum number of fixpoint iterations before stopping evaluation.
    max_iterations: usize,
    /// Whether packed relations have been materialized since the last `run()`.
    materialized: bool,
}

/// An opaque, shareable handle to a pre-compiled JIT compiler.
///
/// Obtained from [`Engine::share_jit_compiler`] and passed to
/// [`Engine::set_jit_compiler`] to avoid recompilation across engine instances.
#[cfg(feature = "jit")]
#[derive(Clone)]
pub struct SharedJitCompiler(Arc<Mutex<crate::eval::jit::JitCompiler>>);

impl Engine {
    /// Create a new engine from a program.
    pub fn new(program: Program) -> Self {
        let mut relations = FxHashMap::default();
        let mut column_types = FxHashMap::default();

        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            let types: Vec<Option<String>> = rel
                .column_types
                .iter()
                .map(|ty| match ty {
                    crate::ir::IrType::Named(name) => Some(name.clone()),
                    crate::ir::IrType::Complex(_) => None,
                })
                .collect();
            relations.insert(
                name.clone(),
                Relation::new_auto(arity, rel.is_lattice, &types),
            );
            column_types.insert(name.clone(), types);
        }

        Engine {
            program,
            relations,
            column_types,
            type_registry: TypeRegistry::new(),
            var_interner: VarInterner::default(),
            var_count: 0,
            source_names: FxHashMap::default(),
            next_source_id: 1,
            current_source: SourceId::ANONYMOUS,
            stratification: None,
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            stratum_stage4_cache: FxHashMap::default(),
            max_iterations: 10_000,
            materialized: true,
        }
    }

    /// Return a reference to the program this engine was created with.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Enable JIT compilation for eligible rules.
    #[cfg(feature = "jit")]
    pub fn enable_jit(&mut self) -> Result<(), crate::eval::error::EvalError> {
        if self.jit.is_none() {
            match crate::eval::jit::JitCompiler::new() {
                Ok(compiler) => self.jit = Some(Arc::new(Mutex::new(compiler))),
                Err(e) => {
                    return Err(crate::eval::error::EvalError::Jit(format!(
                        "JIT init failed: {e}"
                    )))
                }
            }
        }
        Ok(())
    }

    /// Return a shared handle to the compiled JIT state.
    /// Multiple engines can share one handle to avoid recompilation.
    #[cfg(feature = "jit")]
    pub fn share_jit_compiler(&self) -> Option<SharedJitCompiler> {
        self.jit.clone().map(SharedJitCompiler)
    }

    /// Inject a pre-compiled JIT compiler from another engine.
    /// The injected compiler is shared; compilation results are visible to all sharers.
    #[cfg(feature = "jit")]
    pub fn set_jit_compiler(&mut self, jit: SharedJitCompiler) {
        self.jit = Some(jit.0);
    }

    /// Set the maximum number of fixpoint iterations before evaluation stops.
    ///
    /// Default is 10,000. Programs that exceed this limit will return
    /// `Err(EvalError::IterationLimitExceeded)`.
    pub fn set_max_iterations(&mut self, limit: usize) {
        self.max_iterations = limit;
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
    ///         let p = val.downcast_custom::<MyPoint>()?;
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

    /// Returns a reference to the type registry.
    pub fn type_registry(&self) -> &TypeRegistry {
        &self.type_registry
    }

    /// Downcast a custom value to a concrete type.
    ///
    /// Deprecated: use [`Value::downcast_custom`] instead.
    #[deprecated(since = "0.2.0", note = "use Value::downcast_custom instead")]
    pub fn downcast_custom<T: DynValue + 'static>(value: &Value) -> Option<&T> {
        value.downcast_custom()
    }

    /// Get a relation by name.
    ///
    /// If [`Engine::run`] was called since the last [`Engine::materialize`],
    /// packed-relation indices may be stale. Call `materialize()` first when
    /// you need fully-synced data (e.g., key lookups or index scans).
    pub fn relation(&self, name: &str) -> Option<&Relation> {
        if !self.materialized {
            eprintln!(
                "Warning: relation() called before materialize(). \
                 Packed relation data may be stale."
            );
        }
        self.relations.get(name)
    }

    /// Get a mutable relation by name.
    ///
    /// Automatically calls [`Engine::materialize`] if needed.
    pub fn relation_mut(&mut self, name: &str) -> Option<&mut Relation> {
        if !self.materialized {
            self.materialize();
        }
        self.relations.get_mut(name)
    }

    /// Sync interpreter-only state for all packed relations.
    ///
    /// The JIT hot path skips updating `indices`, `value_data`, and `source_tags` in
    /// [`PackedStorage`] to reduce per-tuple overhead.  Call this after [`Engine::run`]
    /// and before accessing relation contents (e.g. iterating tuples or doing key lookups).
    /// The benchmark loop does not call this method, so the benchmark measures only the
    /// core JIT computation time.
    pub fn materialize(&mut self) {
        #[cfg(feature = "specialized")]
        for rel in self.relations.values_mut() {
            if let Relation::Packed(ps) = rel {
                ps.ensure_interp_synced();
            }
        }
        self.materialized = true;
    }

    /// Insert a tuple into a relation (untagged / [`SourceId::ANONYMOUS`]).
    pub fn insert(
        &mut self,
        relation: &str,
        tuple: Tuple,
    ) -> Result<bool, crate::eval::error::EvalError> {
        if let Some(rel) = self.relations.get_mut(relation) {
            if let Some(col) = self.column_types.get(relation)
                && col.len() != tuple.len()
            {
                return Err(crate::eval::error::EvalError::ArityMismatch {
                    relation: relation.to_string(),
                    expected: col.len(),
                    got: tuple.len(),
                });
            }
            Ok(rel.insert(tuple))
        } else {
            Err(crate::eval::error::EvalError::UnknownRelation(
                relation.to_string(),
            ))
        }
    }

    /// Insert a tuple tagged with a source.
    ///
    /// Validates that the relation exists and the tuple arity matches, just
    /// like [`Engine::insert`].
    pub fn insert_with_source(
        &mut self,
        relation: &str,
        tuple: Tuple,
        source: SourceId,
    ) -> Result<bool, crate::eval::error::EvalError> {
        if let Some(rel) = self.relations.get_mut(relation) {
            if let Some(col) = self.column_types.get(relation)
                && col.len() != tuple.len()
            {
                return Err(crate::eval::error::EvalError::ArityMismatch {
                    relation: relation.to_string(),
                    expected: col.len(),
                    got: tuple.len(),
                });
            }
            Ok(rel.insert_with_source(tuple, source))
        } else {
            Err(crate::eval::error::EvalError::UnknownRelation(
                relation.to_string(),
            ))
        }
    }

    /// Intern a source name, returning a stable [`SourceId`].
    ///
    /// Repeated calls with the same name return the same ID.
    pub fn intern_source(&mut self, name: &str) -> SourceId {
        if let Some(&id) = self.source_names.get(name) {
            return id;
        }
        debug_assert!(self.next_source_id < u32::MAX, "SourceId overflow");
        let id = SourceId(self.next_source_id);
        self.next_source_id += 1;
        self.source_names.insert(name.to_string(), id);
        id
    }

    /// Set the active source context. Facts from empty-body rules during
    /// subsequent `run()` calls will be tagged with this source.
    pub fn set_source(&mut self, source: SourceId) {
        self.current_source = source;
    }

    /// Get the currently active source context.
    pub fn source(&self) -> SourceId {
        self.current_source
    }

    /// Remove all facts tagged with the given source from every relation.
    /// Returns the total number of tuples removed.
    pub fn retract_source(&mut self, source: SourceId) -> usize {
        let mut total = 0;
        for rel in self.relations.values_mut() {
            total += rel.retract_source(source);
        }
        total
    }

    /// Remove all facts tagged with any of the given sources from every relation.
    /// Returns the total number of tuples removed.
    ///
    /// More efficient than calling `retract_source` in a loop — each relation
    /// is rebuilt at most once regardless of how many sources are retracted.
    pub fn retract_sources(&mut self, sources: impl IntoIterator<Item = SourceId>) -> usize {
        let set: rustc_hash::FxHashSet<SourceId> = sources.into_iter().collect();
        if set.is_empty() {
            return 0;
        }
        let mut total = 0;
        for rel in self.relations.values_mut() {
            total += rel.retract_sources(&set);
        }
        total
    }

    /// Run the program to fixpoint using semi-naive evaluation with SCC-based stratification.
    ///
    /// Rules are grouped into strongly connected components and processed in
    /// topological order. Each SCC runs to fixpoint before dependent SCCs begin.
    pub fn run(&mut self) -> Result<(), crate::eval::error::EvalError> {
        self.materialized = false;
        self.ensure_stratification();
        let strat = self
            .stratification
            .as_ref()
            .expect("stratification must be set by ensure_stratification");

        // Clone the SCC indices to avoid borrow conflict with self.
        let sccs: Vec<Vec<usize>> = strat.sccs.clone();
        let compiled: Vec<CRule> = strat.compiled.clone();

        self.var_count = self.var_interner.len();

        for (scc_idx, scc_indices) in sccs.iter().enumerate() {
            let rules: Vec<&CRule> = scc_indices.iter().map(|&i| &compiled[i]).collect();
            self.run_stratum(&rules, scc_idx, scc_indices)?;
        }

        // Clear recent/delta so stale data doesn't confuse the next run.
        for rel in self.relations.values_mut() {
            rel.clear_recent();
        }
        Ok(())
    }

    /// Run only the strata affected by changes to the given "dirty" relations.
    ///
    /// Walks the SCC DAG in topological order. For each SCC whose reads
    /// intersect the current dirty set, clears its output relations and
    /// re-runs the stratum to fixpoint. Newly written relations propagate
    /// as dirty to downstream strata.
    ///
    /// `retracted` lists relations that had data removed (not just added).
    /// Monotone SCCs that only read addition-only dirty relations can use
    /// delta-only incremental evaluation; those that read retracted relations
    /// fall back to full clear-and-rederive.
    ///
    /// Returns the set of all relation names that were re-derived.
    pub fn run_incremental(
        &mut self,
        dirty: &[&str],
        retracted: &[&str],
    ) -> Result<FxHashSet<String>, crate::eval::error::EvalError> {
        if dirty.is_empty() {
            return Ok(FxHashSet::default());
        }

        let dirty: FxHashSet<String> = dirty.iter().map(|s| s.to_string()).collect();
        let retracted: FxHashSet<String> = retracted.iter().map(|s| s.to_string()).collect();

        self.ensure_stratification();
        let strat = self
            .stratification
            .as_ref()
            .expect("stratification must be set by ensure_stratification");

        let sccs: Vec<Vec<usize>> = strat.sccs.clone();
        let scc_reads: Vec<FxHashSet<String>> = strat.scc_reads.clone();
        let scc_writes: Vec<FxHashSet<String>> = strat.scc_writes.clone();
        let compiled: Vec<CRule> = strat.compiled.clone();
        let scc_is_monotone: Vec<bool> = strat.scc_is_monotone.clone();

        self.var_count = self.var_interner.len();

        // Phase 1: identify all dirty SCCs via forward propagation through the DAG.
        let mut dirty_sccs = vec![false; sccs.len()];
        let mut current_dirty = dirty.clone();
        for (scc_idx, _) in sccs.iter().enumerate() {
            if scc_reads[scc_idx].iter().any(|r| current_dirty.contains(r)) {
                dirty_sccs[scc_idx] = true;
                for rel_name in &scc_writes[scc_idx] {
                    current_dirty.insert(rel_name.clone());
                }
            }
        }

        // Phase 2: classify each dirty SCC as incremental (delta-only) or full (clear+rederive).
        // An SCC needs full eval if it's non-monotone, reads a retracted relation,
        // or reads a relation cleared by another full-eval SCC.
        let mut needs_full_eval = vec![false; sccs.len()];
        let mut cleared_relations: FxHashSet<String> = retracted.clone();
        for (scc_idx, _) in sccs.iter().enumerate() {
            if !dirty_sccs[scc_idx] {
                continue;
            }
            if !scc_is_monotone[scc_idx]
                || scc_reads[scc_idx]
                    .iter()
                    .any(|r| cleared_relations.contains(r))
            {
                needs_full_eval[scc_idx] = true;
                for rel_name in &scc_writes[scc_idx] {
                    cleared_relations.insert(rel_name.clone());
                }
            }
        }

        // Phase 2b: clear all output relations of full-eval SCCs (once, before any re-run).
        let mut all_rederived = FxHashSet::default();
        for (scc_idx, _) in sccs.iter().enumerate() {
            if !dirty_sccs[scc_idx] || !needs_full_eval[scc_idx] {
                continue;
            }
            for rel_name in &scc_writes[scc_idx] {
                if let Some(rel) = self.relations.get_mut(rel_name) {
                    rel.clear();
                }
                all_rederived.insert(rel_name.clone());
            }
        }

        // Phase 3: walk dirty SCCs in topological order.
        for (scc_idx, scc_indices) in sccs.iter().enumerate() {
            if !dirty_sccs[scc_idx] {
                continue;
            }
            let rules: Vec<&CRule> = scc_indices.iter().map(|&i| &compiled[i]).collect();

            if needs_full_eval[scc_idx] {
                self.run_stratum(&rules, scc_idx, scc_indices)?;
            } else {
                // Snapshot counts of owned relations before incremental eval
                let pre_counts: Vec<(String, usize)> = scc_writes[scc_idx]
                    .iter()
                    .filter_map(|name| self.relations.get(name).map(|r| (name.clone(), r.len())))
                    .collect();

                self.run_stratum_incremental(&rules, &scc_writes[scc_idx])?;

                // Propagate newly-derived tuples as deltas for downstream SCCs
                for (name, old_count) in pre_counts {
                    if let Some(rel) = self.relations.get_mut(&name)
                        && rel.len() > old_count
                    {
                        rel.set_delta_range(old_count);
                    }
                }
            }
            for rel_name in &scc_writes[scc_idx] {
                all_rederived.insert(rel_name.clone());
            }
        }

        // Clear recent/delta so stale data doesn't confuse the next run.
        for rel in self.relations.values_mut() {
            rel.clear_recent();
        }

        Ok(all_rederived)
    }

    /// Ensure the stratification cache is populated.
    fn ensure_stratification(&mut self) {
        if self.stratification.is_none() {
            self.stratification =
                Some(Stratification::build(&self.program, &self.var_interner));
        }
    }

    /// Replace the engine's program and sync the relation registry.
    ///
    /// Creates storage for new relations without disturbing existing ones.
    /// Existing relation data is preserved for incremental re-evaluation.
    /// Invalidates the cached stratification so it is rebuilt on next run.
    pub fn update_program(&mut self, program: Program) {
        self.stratification = None;
        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            let types: Vec<Option<String>> = rel
                .column_types
                .iter()
                .map(|ty| match ty {
                    crate::ir::IrType::Named(name) => Some(name.clone()),
                    crate::ir::IrType::Complex(_) => None,
                })
                .collect();
            self.relations
                .entry(name.clone())
                .or_insert_with(|| Relation::new_auto(arity, rel.is_lattice, &types));
            self.column_types.entry(name.clone()).or_insert_with(|| types);
        }
        self.program = program;
    }

    /// Run a set of rules to fixpoint.
    #[allow(unused_variables)]
    fn run_stratum(
        &mut self,
        rules: &[&CRule],
        scc_key: usize,
        rule_indices: &[usize],
    ) -> Result<(), crate::eval::error::EvalError> {
        if rules.is_empty() {
            return Ok(());
        }

        // Fast path: Stage 4 stratum (inlined rule bodies, no call_indirect)
        #[cfg(all(feature = "jit", feature = "specialized"))]
        if self.try_run_stratum_stage4(rules, scc_key, rule_indices) {
            return Ok(());
        }

        // Interpreter fallback: sync interpreter state for all packed relations.
        // JIT strata skip updating indices/value_data/source_tags; the interpreter
        // reads those structures, so they must be synced before evaluation begins.
        #[cfg(feature = "specialized")]
        {
            use crate::eval::relation::Relation;
            for rel in self.relations.values_mut() {
                if let Relation::Packed(ps) = rel {
                    ps.ensure_interp_synced();
                }
            }
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
        let mut iterations = 0usize;
        while changed {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(crate::eval::error::EvalError::IterationLimitExceeded {
                    limit: self.max_iterations,
                });
            }
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
        Ok(())
    }

    /// Check whether any rule uses ordering comparisons (Lt/Le/Gt/Ge) on variables
    /// bound to interned columns. The JIT compares raw packed u32 intern IDs, which
    /// gives wrong results for interned types where ID order != semantic order (e.g.
    /// strings: intern ID depends on insertion order, not lexicographic order).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn has_interned_ordering_cmp(rules: &[&CRule], relations: &FxHashMap<String, Relation>) -> bool {
        use crate::eval::specialized::PackedType;

        for rule in rules {
            let mut interned_vars = FxHashSet::default();
            for item in &rule.body {
                if let CBodyItem::Clause(clause) = item
                    && let Some(Relation::Packed(ps)) = relations.get(&clause.relation)
                {
                    for &(col, var_id) in &clause.fresh_cols {
                        if col < ps.col_types.len()
                            && matches!(ps.col_types[col], PackedType::Interned(_))
                        {
                            interned_vars.insert(var_id);
                        }
                    }
                }
            }
            if interned_vars.is_empty() {
                continue;
            }
            // Check all conditions (rule-level and clause-level)
            for item in &rule.body {
                match item {
                    CBodyItem::Condition(CCondition::If(expr)) => {
                        if Self::expr_has_interned_ordering(expr, &interned_vars) {
                            return true;
                        }
                    }
                    CBodyItem::Clause(clause) => {
                        for cond in &clause.conditions {
                            if let CCondition::If(expr) = cond
                                && Self::expr_has_interned_ordering(expr, &interned_vars)
                            {
                                return true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// Returns true if the expression contains an ordering comparison (Lt/Le/Gt/Ge)
    /// where at least one operand is or references an interned variable.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn expr_has_interned_ordering(expr: &CExpr, interned_vars: &FxHashSet<VarId>) -> bool {
        fn is_ordering(op: CBinOp) -> bool {
            matches!(op, CBinOp::Lt | CBinOp::Le | CBinOp::Gt | CBinOp::Ge)
        }
        fn uses_interned(expr: &CExpr, vars: &FxHashSet<VarId>) -> bool {
            match expr {
                CExpr::Var(id) | CExpr::DerefVar(id) => vars.contains(id),
                CExpr::VarBinVar(_, a, b) => vars.contains(a) || vars.contains(b),
                CExpr::VarBinLit(_, a, _) => vars.contains(a),
                CExpr::LitBinVar(_, _, b) => vars.contains(b),
                CExpr::Binary(_, a, b) => uses_interned(a, vars) || uses_interned(b, vars),
                CExpr::Unary(_, inner) => uses_interned(inner, vars),
                _ => false,
            }
        }
        match expr {
            CExpr::VarBinVar(op, a, b) if is_ordering(*op) => {
                interned_vars.contains(a) || interned_vars.contains(b)
            }
            CExpr::VarBinLit(op, a, _) if is_ordering(*op) => interned_vars.contains(a),
            CExpr::LitBinVar(op, _, b) if is_ordering(*op) => interned_vars.contains(b),
            CExpr::Binary(op, a, b) if is_ordering(*op) => {
                uses_interned(a, interned_vars) || uses_interned(b, interned_vars)
            }
            CExpr::Binary(_, a, b) => {
                Self::expr_has_interned_ordering(a, interned_vars)
                    || Self::expr_has_interned_ordering(b, interned_vars)
            }
            CExpr::Unary(_, inner) => Self::expr_has_interned_ordering(inner, interned_vars),
            _ => false,
        }
    }

    /// Try to run the stratum via the Stage 4 compiled function (inlined rule bodies).
    ///
    /// Returns `true` if successful, `false` to fall back to interpreted.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn try_run_stratum_stage4(&mut self, rules: &[&CRule], scc_key: usize, _rule_indices: &[usize]) -> bool {
        if rules.is_empty() {
            return false;
        }

        let stratum_key = scc_key;

        // Step 0: mark EDB relations for this stratum.
        //
        // A relation is EDB in this stratum if it appears as a clause body but never
        // as a rule head. EDB relations are stable — their full JIT index can be built
        // contiguously once before the stratum runs and never rebuilt.
        {
            use crate::eval::compiled::CBodyItem;
            use crate::eval::relation::Relation;

            // Collect all relation names that appear in any head.
            let head_rels: rustc_hash::FxHashSet<&str> = rules
                .iter()
                .flat_map(|r| r.heads.iter().map(|h| h.relation.as_str()))
                .collect();

            // For each body clause: set is_edb iff not in any head.
            for rule in rules {
                for item in &rule.body {
                    if let CBodyItem::Clause(c) = item {
                        let is_edb = !head_rels.contains(c.relation.as_str());
                        if let Some(Relation::Packed(ps)) = self.relations.get_mut(&c.relation) {
                            ps.jit_is_edb = is_edb;
                        }
                    }
                }
            }

            // Set jit_is_sink: true for relations that are head-only across the ENTIRE
            // program (never appear as a body clause in any rule). Using program-wide
            // analysis avoids marking a relation as a sink in one stratum when it will
            // be probed as a body clause in a later stratum.
            {
                let strat = self
                    .stratification
                    .as_ref()
                    .expect("stratification must be set by ensure_stratification");
                let program_body_rels: rustc_hash::FxHashSet<&str> = strat
                    .compiled
                    .iter()
                    .flat_map(|r| r.body.iter().filter_map(|item| {
                        if let CBodyItem::Clause(c) = item { Some(c.relation.as_str()) } else { None }
                    }))
                    .collect();
                for rel_name in &head_rels {
                    if let Some(Relation::Packed(ps)) = self.relations.get_mut(*rel_name) {
                        ps.jit_is_sink = !program_body_rels.contains(rel_name);
                    }
                }
            }

            // If an EDB relation already has a JIT index that was built with
            // `jit_is_edb = false` (tuple-index vals, from a prior fact stratum),
            // rebuild it contiguously NOW in EDB col-value mode before
            // `build_stratum_stage4_runtime` reads the index pointers.
            //
            // The index format depends on `jit_is_edb` at build time:
            //   jit_is_edb=false → vals are tuple indices (standard mode)
            //   jit_is_edb=true  → vals are column values (EDB col-value mode)
            // The asm JIT selects the mode by `!is_rec[level]`, which must
            // match the index format.  If the index was built before `jit_is_edb` was
            // set, the format is wrong and must be rebuilt.
            //
            // We detect the stale format by checking `jit_index_is_edb_fmt`:
            // false = index was last built with jit_is_edb=false (or not built yet).
            for (_, rel) in self.relations.iter_mut() {
                if let Relation::Packed(ps) = rel
                    && ps.jit_is_edb
                    && !ps.jit_index_is_edb_fmt
                {
                    ps.jit_full_indexed_count = 0;
                    ps.update_jit_indices();
                }
            }
        }

        // Step 0b: reject rules with ordering comparisons on interned columns.
        // The JIT compares raw u32 intern IDs, which gives wrong results when
        // ID order != semantic order (e.g. string lexicographic comparison).
        if Self::has_interned_ordering_cmp(rules, &self.relations) {
            if std::env::var("ASCENT_DUMP_JIT").is_ok() {
                eprintln!("JIT: stratum {stratum_key} skipped: ordering comparison on interned column");
            }
            return false;
        }

        // Step 1: compile the Stage 4 stratum function (eligibility checked inside)
        let stage4_fn = {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap_or_else(|e| e.into_inner());
            jit.var_count = self.var_count;
            match jit.compile_stratum_stage4(stratum_key, rules) {
                Some(f) => f,
                None => return false,
            }
        };

        // Step 1b: also compile the native function (reads scan data directly from JitRelData).
        // This is compiled speculatively; it will only be used if stage4_native_runtime is Some.
        let stage4_native_fn: Option<crate::eval::jit::packed_helpers::StratumStage4Fn> =
            self.jit.as_ref().and_then(|jit_cell| {
                let mut jit = jit_cell.lock().unwrap_or_else(|e| e.into_inner());
                jit.var_count = self.var_count;
                jit.compile_stratum_stage4_native(stratum_key, rules)
            });

        let native_fn_available = stage4_native_fn.is_some();

        // Step 2: build or repopulate the runtime context if not yet cached.
        if !self.stratum_stage4_cache.contains_key(&stratum_key) {
            // Pre-size jit_indices BEFORE building or repopulating the runtime so that
            // JitLookupHandle::from_index (called in both build and repopulate) captures
            // the post-reserve entries_ptr.  If we reserved after, the handle would hold
            // a dangling pointer into the freed pre-reserve allocation → SIGSEGV.
            let pool_runtime = if let Some(jit_cell) = self.jit.as_ref() {
                let mut jit = jit_cell.lock().unwrap_or_else(|e| e.into_inner());
                if !jit.tuple_count_hints.is_empty() {
                    use crate::eval::relation::Relation;
                    for rule in rules {
                        for head in &rule.heads {
                            if let Some(&count_hint) =
                                jit.tuple_count_hints.get(head.relation.as_str())
                                && let Some(Relation::Packed(ps)) =
                                    self.relations.get_mut(&head.relation)
                            {
                                for idx in &mut ps.jit_indices {
                                    idx.reserve(count_hint as usize);
                                }
                            }
                        }
                    }
                }
                // Also try the pool: if a prior engine returned its runtime, we can
                // repopulate in-place (pointer writes only, zero allocations).
                jit.stratum_runtime_pool.remove(&stratum_key)
            } else {
                None
            };

            let runtime = if let Some(mut rt) = pool_runtime
                && self.repopulate_runtime(&mut rt)
            {
                // Pool hit: pointer writes only, zero allocations.
                rt
            } else {
                // Pool miss or non-repopulatable (native path): build fresh.
                match self.build_stratum_stage4_runtime(rules, native_fn_available) {
                    Some(r) => r,
                    None => return false,
                }
            };
            self.stratum_stage4_cache.insert(stratum_key, runtime);
        }

        // Step 3: call the stratum function.
        // If the native runtime is available AND the native function compiled successfully,
        // use the native path (zero read-side Rust callbacks).
        // Otherwise fall back to the old path.
        let runtime = self.stratum_stage4_cache.get_mut(&stratum_key).unwrap();

        // Step 3a: pre-size IDB head relations using hints from a previous run.
        // Avoids 6–8 dedup reallocs and ~13 packed_data/delta reallocs on each
        // fresh-engine iteration, eliminating malloc/memmove overhead.
        if let Some(jit_cell) = self.jit.as_ref() {
            let jit = jit_cell.lock().unwrap_or_else(|e| e.into_inner());
            let has_hints = !jit.dedup_cap_hints.is_empty() || !jit.tuple_count_hints.is_empty();
            if has_hints {
                use crate::eval::relation::Relation;
                for rule in rules {
                    for head in &rule.heads {
                        if let Some(Relation::Packed(ps)) =
                            self.relations.get_mut(&head.relation)
                        {
                            if let Some(&dedup_hint) =
                                jit.dedup_cap_hints.get(head.relation.as_str())
                            {
                                ps.jit_dedup.reserve(dedup_hint);
                            }
                            if let Some(&count_hint) =
                                jit.tuple_count_hints.get(head.relation.as_str())
                            {
                                ps.reserve_tuples(count_hint as usize);
                                // Pre-size jit_native.new to avoid grow cascades in the
                                // hot inner loop. new starts at capacity 16; for triangle
                                // n=20 (1140 tuples) it grows 7× without hints, replaced
                                // here with a single upfront alloc+memset.
                                if let Some(native) = ps.jit_native.as_mut() {
                                    // Safety: native.new is a fully initialized JitRelData
                                    // built by build_native_projection (valid buffers).
                                    unsafe { native.new.pre_size(count_hint as usize) };
                                }
                                // Pre-size recent indices: recent typically holds 1–2
                                // tuples per iteration; cap=4 avoids any rehash after warmup.
                                // Safe here (after runtime build) because recent handles are
                                // only used in the recent loop, which runs after advance()
                                // refreshes the handles from the current index state.
                                for idx in &mut ps.jit_recent_indices {
                                    idx.reserve(2);
                                }
                            }
                        }
                    }
                }
            }
        }

        if let (Some(native_fn), Some(native_runtime)) =
            (stage4_native_fn, runtime.stage4_native_runtime.as_mut())
        {
            // The native fn takes *mut StratumStage4NativeCtx, but StratumStage4Fn
            // is typed as *mut StratumStage4Ctx. Both are opaque pointers at the ABI
            // level; transmute is safe here since we know which ctx the fn expects.
            type NativeFn = unsafe extern "C" fn(*mut crate::eval::jit::packed_helpers::StratumStage4NativeCtx);
            let native_fn_typed: NativeFn = unsafe { std::mem::transmute(native_fn) };
            unsafe { native_fn_typed(&raw mut *native_runtime.ctx) };
            // Update dedup capacity hints after the native path run.
            self.update_dedup_cap_hints(rules);
            return true;
        }

        unsafe { stage4_fn(&raw mut *runtime.stage4_ctx) };

        // Update dedup capacity hints for the next engine's first run.
        self.update_dedup_cap_hints(rules);

        true
    }

    /// Record the peak dedup-table capacity and tuple count for each IDB head
    /// relation into `JitCompiler`.  Called after each stratum run completes.
    /// After the final fixpoint advance, `jit_dedup.clear()` zeroes the count but
    /// preserves the capacity allocation — so `handle.cap` is the peak capacity.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn update_dedup_cap_hints(&self, rules: &[&crate::eval::compiled::CRule]) {
        let Some(jit_cell) = self.jit.as_ref() else { return };
        use crate::eval::relation::Relation;
        let mut hints: Vec<(String, u32, u32)> = Vec::new(); // (name, dedup_cap, tuple_count)
        for rule in rules {
            for head in &rule.heads {
                if let Some(Relation::Packed(ps)) = self.relations.get(&head.relation) {
                    let cap = ps.jit_dedup.handle.cap;
                    let count = ps.count as u32;
                    if cap > 0 || count > 0 {
                        hints.push((head.relation.clone(), cap, count));
                    }
                }
            }
        }
        if hints.is_empty() {
            return;
        }
        let mut jit = jit_cell.lock().unwrap_or_else(|e| e.into_inner());
        for (name, cap, count) in hints {
            if cap > 0 {
                let entry = jit.dedup_cap_hints.entry(name.clone()).or_insert(0);
                *entry = (*entry).max(cap);
            }
            if count > 0 {
                let entry = jit.tuple_count_hints.entry(name).or_insert(0);
                *entry = (*entry).max(count);
            }
        }
    }

    /// Build the runtime context for Stage 4 stratum execution.
    ///
    /// Returns `None` if any rule doesn't have all-packed clause or head relations.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn build_stratum_stage4_runtime(&mut self, rules: &[&CRule], native_fn_available: bool) -> Option<StratumStage4Runtime> {
        use crate::eval::jit::packed_helpers::{LookupSpec, PackedJitContextV3, StratumStage4Ctx};
        use crate::eval::jit_index::JitLookupHandle;
        use crate::eval::relation::Relation;
        use crate::eval::specialized::PackedStorage;

        // Rebuild stale JIT indices before building handles: asm-native strata may have
        // skipped update_jit_indices(), leaving jit_indices/jit_recent_indices stale for IDB rels.
        if !native_fn_available {
            for rule in rules {
                for item in &rule.body {
                    if let CBodyItem::Clause(c) = item
                        && let Some(Relation::Packed(ps)) = self.relations.get_mut(&c.relation)
                    {
                        ps.update_jit_indices();
                    }
                }
            }
        }

        let mut per_rule_clause_rels: Vec<Box<[*const PackedStorage]>> = Vec::new();
        let mut per_rule_head_rels: Vec<Box<[*mut PackedStorage]>> = Vec::new();
        let mut per_rule_dedup_handles: Vec<Box<[*mut crate::eval::jit_index::JitDedupHandle]>> =
            Vec::new();
        let mut per_rule_ctxs: Vec<Box<PackedJitContextV3>> = Vec::new();
        let mut rule_ctx_ptrs_vec: Vec<*mut PackedJitContextV3> = Vec::new();

        // Flat handles and specs (all rules concatenated).
        let mut handles_flat: Vec<JitLookupHandle> = Vec::new();
        let mut specs_flat: Vec<LookupSpec> = Vec::new();
        // Starting handle index for each rule (for setting lookup_handles ptr later).
        let mut rule_handle_offsets: Vec<usize> = Vec::new();

        // Refresh info: relation names parallel to pointer arrays (for pool reuse).
        let mut ri_clause_rel_names: Vec<Box<[String]>> = Vec::new();
        let mut ri_head_rel_names: Vec<Box<[String]>> = Vec::new();
        let mut ri_spec_rel_names: Vec<String> = Vec::new();

        for rule in rules {
            // Clause rel pointers (with parallel name collection for pool refresh).
            let (mut clause_rel_names_rule, mut clause_rels): (Vec<String>, Vec<*const PackedStorage>) = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => match self.relations.get(&c.relation)? {
                        Relation::Packed(p) => Some((c.relation.clone(), p as *const PackedStorage)),
                        Relation::Generic(_) => None,
                    },
                    _ => None,
                })
                .unzip();

            let clause_count = rule
                .body
                .iter()
                .filter(|i| matches!(i, CBodyItem::Clause(_)))
                .count();

            if clause_rels.len() != clause_count {
                return None;
            }

            // Append negation relation pointers after clause rels.
            // The asm backend accesses these as rels[clause_count + neg_i].
            let not_count_expected = rule.body.iter()
                .filter(|i| matches!(i, CBodyItem::Aggregation(a) if a.aggregator_name == "not"))
                .count();
            for item in &rule.body {
                if let CBodyItem::Aggregation(a) = item
                    && a.aggregator_name == "not" {
                    match self.relations.get(&a.relation) {
                        Some(Relation::Packed(p)) => {
                            clause_rels.push(p as *const PackedStorage);
                            clause_rel_names_rule.push(a.relation.clone());
                        }
                        _ => return None,
                    }
                }
            }
            if clause_rels.len() != clause_count + not_count_expected {
                return None;
            }

            // Append real aggregation relation pointers after negation rels.
            // The asm backend accesses these as rels[clause_count + not_count + agg_i].
            let agg_count_expected = rule.body.iter()
                .filter(|i| matches!(i, CBodyItem::Aggregation(a) if a.aggregator_name != "not"))
                .count();
            for item in &rule.body {
                if let CBodyItem::Aggregation(a) = item
                    && a.aggregator_name != "not" {
                    match self.relations.get(&a.relation) {
                        Some(Relation::Packed(p)) => {
                            clause_rels.push(p as *const PackedStorage);
                            clause_rel_names_rule.push(a.relation.clone());
                        }
                        _ => return None,
                    }
                }
            }
            if clause_rels.len() != clause_count + not_count_expected + agg_count_expected {
                return None;
            }

            // Head rel pointers (with parallel name collection for pool refresh).
            let (head_rel_names_rule, head_rels): (Vec<String>, Vec<*mut PackedStorage>) = rule
                .heads
                .iter()
                .filter_map(|head| match self.relations.get(&head.relation)? {
                    Relation::Packed(p) => {
                        Some((head.relation.clone(), p as *const PackedStorage as *mut PackedStorage))
                    }
                    Relation::Generic(_) => None,
                })
                .unzip();

            if head_rels.len() != rule.heads.len() {
                return None;
            }

            // Build 2 handles per clause (full + recent).
            // Index: clause_offset * 2 + use_recent
            let rule_handle_start = handles_flat.len();
            rule_handle_offsets.push(rule_handle_start);

            // Extract clause body items in order to find the primary bound column.
            let rule_clauses: Vec<&crate::eval::compiled::CClause> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => Some(c),
                    _ => None,
                })
                .collect();

            for (clause_offset, clause) in rule_clauses.iter().enumerate() {
                let rel_ptr = clause_rels[clause_offset];
                let rel_name = &clause_rel_names_rule[clause_offset];
                for use_recent_flag in [0u32, 1u32] {
                    ri_spec_rel_names.push(rel_name.clone());
                    if clause.bound_cols.is_empty() {
                        // Full-scan clause — null handle (never accessed by inline probe)
                        handles_flat.push(JitLookupHandle::null());
                        specs_flat.push(LookupSpec {
                            rel: rel_ptr,
                            col: 0,
                            use_recent: use_recent_flag,
                        });
                    } else {
                        let col = clause.bound_cols[0];
                        let ps = unsafe { &*rel_ptr };
                        let idx = if use_recent_flag != 0 {
                            ps.jit_recent_indices.get(col)
                        } else {
                            ps.jit_indices.get(col)
                        };
                        let handle = match idx {
                            Some(i) => JitLookupHandle::from_index(i),
                            None => JitLookupHandle::null(),
                        };
                        handles_flat.push(handle);
                        specs_flat.push(LookupSpec {
                            rel: rel_ptr,
                            col: col as u32,
                            use_recent: use_recent_flag,
                        });
                    }
                }
            }

            ri_clause_rel_names.push(clause_rel_names_rule.into_boxed_slice());
            ri_head_rel_names.push(head_rel_names_rule.into_boxed_slice());

            let clause_rels_box: Box<[*const PackedStorage]> = clause_rels.into_boxed_slice();
            let head_rels_box: Box<[*mut PackedStorage]> = head_rels.into_boxed_slice();
            let head_rels_ptr: *const *mut PackedStorage = head_rels_box.as_ptr();

            // Build dedup handle pointers (one per head relation).
            let dedup_handles: Box<[*mut crate::eval::jit_index::JitDedupHandle]> = head_rels_box
                .iter()
                .map(|&ps| unsafe { &raw mut (*ps).jit_dedup.handle })
                .collect();
            let head_dedup_handles_ptr = dedup_handles.as_ptr();

            // lookup_handles and jit_rels pointers will be fixed up after flat buffers are boxed.
            let ctx = Box::new(PackedJitContextV3 {
                rels: clause_rels_box.as_ptr(),
                rels_len: clause_rels_box.len() as u32,
                _pad: 0,
                head_rels: head_rels_ptr,
                lookup_handles: std::ptr::null(), // fixed up below
                head_dedup_handles: head_dedup_handles_ptr,
                jit_rels: std::ptr::null(), // fixed up below
            });

            rule_ctx_ptrs_vec
                .push(&*ctx as *const PackedJitContextV3 as *mut PackedJitContextV3);

            per_rule_clause_rels.push(clause_rels_box);
            per_rule_head_rels.push(head_rels_box);
            per_rule_dedup_handles.push(dedup_handles);
            per_rule_ctxs.push(ctx);
        }

        let total_handles = handles_flat.len() as u32;
        let mut handles_box: Box<[JitLookupHandle]> = handles_flat.into_boxed_slice();
        let specs_box: Box<[LookupSpec]> = specs_flat.into_boxed_slice();

        // Fix up per-rule ctx lookup_handles pointers now that handles_box is stable.
        for (rule_i, ctx) in per_rule_ctxs.iter_mut().enumerate() {
            let offset = rule_handle_offsets[rule_i];
            // SAFETY: handles_box is pinned in a Box which won't move.
            ctx.lookup_handles = unsafe { handles_box.as_mut_ptr().add(offset) };
        }

        // All packed rels for advance() (with parallel name collection for pool refresh).
        let (ri_all_rel_names, all_rels): (Vec<String>, Vec<*mut PackedStorage>) = self
            .relations
            .iter()
            .filter_map(|(name, rel)| match rel {
                Relation::Packed(p) => Some((name.clone(), p as *const PackedStorage as *mut PackedStorage)),
                Relation::Generic(_) => None,
            })
            .unzip();

        let all_rels_box: Box<[*mut PackedStorage]> = all_rels.into_boxed_slice();
        let rule_ctx_ptrs_box: Box<[*mut PackedJitContextV3]> =
            rule_ctx_ptrs_vec.into_boxed_slice();

        let stage4_ctx = Box::new(StratumStage4Ctx {
            rule_ctxs: rule_ctx_ptrs_box.as_ptr(),
            num_rules: per_rule_ctxs.len() as u32,
            _pad: 0,
            all_rels: all_rels_box.as_ptr(),
            n_all_rels: all_rels_box.len() as u32,
            _pad2: 0,
            handles_buf: handles_box.as_mut_ptr(),
            lookup_specs: specs_box.as_ptr(),
            total_handles,
            _pad3: 0,
            // Unused fields (reserved for future use):
            tuple_sets_buf: std::ptr::null(),
            jit_rel_specs: std::ptr::null(),
            jit_rel_ptrs: std::ptr::null_mut(),
            total_jit_rels: 0,
            _pad5: 0,
        });

        // Build the native runtime only when the asm native fn actually compiled.
        // Skipping this when native_fn_available=false avoids initializing jit_native on
        // every PackedStorage, which would then be rebuilt on every fixpoint iteration even
        // though the native asm path is never used (e.g. fibonacci unsupported exprs).
        let native_runtime = if native_fn_available {
            self.build_stratum_stage4_native_runtime(rules)
        } else {
            None
        };

        // Pool refresh info: only available for non-native path (native has additional
        // raw pointers in stage4_native_runtime that would also need updating on reuse).
        let refresh_info = if native_runtime.is_none() {
            Some(StratumStage4RefreshInfo {
                all_rel_names: ri_all_rel_names.into_boxed_slice(),
                lookup_spec_rel_names: ri_spec_rel_names.into_boxed_slice(),
                clause_rel_names: ri_clause_rel_names,
                head_rel_names: ri_head_rel_names,
            })
        } else {
            None
        };

        Some(StratumStage4Runtime {
            stage4_ctx,
            _per_rule_clause_rels: per_rule_clause_rels,
            _per_rule_head_rels: per_rule_head_rels,
            _per_rule_ctxs: per_rule_ctxs,
            _rule_ctx_ptrs: rule_ctx_ptrs_box,
            _all_rels: all_rels_box,
            _per_rule_dedup_handles: per_rule_dedup_handles,
            _handles_buf: handles_box,
            _lookup_specs: specs_box,
            stage4_native_runtime: native_runtime,
            refresh_info,
        })
    }

    /// Repopulate a pooled `StratumStage4Runtime` with new engine-specific pointers.
    ///
    /// Updates all raw `*const/*mut PackedStorage` and `*mut JitDedupHandle` slots in the
    /// runtime to point to this engine's relations.  Also rebuilds `jit_indices` and
    /// the `_handles_buf` so the first JIT body execution sees valid data.
    ///
    /// Returns `false` if any required relation is missing (shouldn't happen for pooled runtimes
    /// built from the same program).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn repopulate_runtime(&mut self, rt: &mut StratumStage4Runtime) -> bool {
        use crate::eval::jit_index::JitLookupHandle;
        use crate::eval::relation::Relation;

        let Some(info) = rt.refresh_info.as_ref() else { return false; };

        // 1. Update _all_rels
        for (slot, name) in rt._all_rels.iter_mut().zip(info.all_rel_names.iter()) {
            if let Some(Relation::Packed(ps)) = self.relations.get(name.as_str()) {
                *slot = ps as *const _ as *mut _;
            } else {
                return false;
            }
        }
        // stage4_ctx.all_rels already points into _all_rels (Box didn't move). ✓

        // 2. Update _lookup_specs.rel
        for (spec, name) in rt._lookup_specs.iter_mut().zip(info.lookup_spec_rel_names.iter()) {
            if let Some(Relation::Packed(ps)) = self.relations.get(name.as_str()) {
                spec.rel = ps as *const _;
            } else {
                return false;
            }
        }

        // 3. Update _per_rule_clause_rels and _per_rule_head_rels and _per_rule_dedup_handles
        for rule_i in 0..rt._per_rule_clause_rels.len() {
            let clause_names = &info.clause_rel_names[rule_i];
            for (slot, name) in rt._per_rule_clause_rels[rule_i].iter_mut().zip(clause_names.iter()) {
                if let Some(Relation::Packed(ps)) = self.relations.get(name.as_str()) {
                    *slot = ps as *const _;
                } else {
                    return false;
                }
            }
            // ctx.rels already points into clause_rels_box (Box didn't move). ✓

            let head_names = &info.head_rel_names[rule_i];
            for (slot, name) in rt._per_rule_head_rels[rule_i].iter_mut().zip(head_names.iter()) {
                if let Some(Relation::Packed(ps)) = self.relations.get(name.as_str()) {
                    *slot = ps as *const _ as *mut _;
                } else {
                    return false;
                }
            }
            // ctx.head_rels already points into head_rels_box (Box didn't move). ✓

            // Rebuild dedup handles from updated head_rels.
            for (slot, &head_ps) in rt._per_rule_dedup_handles[rule_i]
                .iter_mut()
                .zip(rt._per_rule_head_rels[rule_i].iter())
            {
                *slot = unsafe { &raw mut (*head_ps).jit_dedup.handle };
            }
            // ctx.head_dedup_handles already points into dedup_box (Box didn't move). ✓
        }

        // 4. Update jit_indices for clause relations (needed before rebuilding handles_buf).
        //    If jit_indices is empty (fresh engine), pre-create with capacity reserved by the
        //    step-2 pre-size (count_hint already applied to the JitHashIndex if it existed;
        //    for a fresh engine the reserve was a no-op, so pre-grow here before inserting).
        for clause_names in info.clause_rel_names.iter() {
            for name in clause_names.iter() {
                if let Some(Relation::Packed(ps)) = self.relations.get_mut(name.as_str()) {
                    // Ensure jit_indices exists with pre-reserved capacity before inserting.
                    if ps.jit_indices.is_empty() && ps.count > 0 && ps.arity > 0 {
                        ps.jit_indices = (0..ps.arity)
                            .map(|_| {
                                let mut idx = crate::eval::jit_index::JitHashIndex::empty();
                                // Pre-reserve to avoid rehash cascade during update_jit_indices.
                                // count+1 ensures enough capacity even before reserve hint is known.
                                idx.reserve(ps.count + 1);
                                idx
                            })
                            .collect();
                    }
                    ps.update_jit_indices();
                }
            }
        }

        // 5. Rebuild _handles_buf from new jit_indices.
        //    Non-native path runs full-variant body BEFORE first advance_s4 call,
        //    so stale handle entries would be dereferenced → must be refreshed here.
        let total = rt.stage4_ctx.total_handles as usize;
        for i in 0..total {
            let spec = unsafe { &*rt.stage4_ctx.lookup_specs.add(i) };
            let ps = unsafe { &*spec.rel };
            let idx_opt = if spec.use_recent != 0 {
                ps.jit_recent_indices.get(spec.col as usize)
            } else {
                ps.jit_indices.get(spec.col as usize)
            };
            let handle = unsafe { &mut *rt.stage4_ctx.handles_buf.add(i) };
            *handle = match idx_opt {
                Some(idx) => JitLookupHandle::from_index(idx),
                None => JitLookupHandle::null(),
            };
        }

        true
    }

    /// Build the native runtime context for Stage 4 stratum execution.
    ///
    /// Returns `None` if any clause or head relation lacks `jit_native` (i.e., `advance_jit()`
    /// has not yet been called), or if any relation is not packed.
    ///
    /// The returned `StratumStage4NativeRuntime` holds the `StratumStage4NativeCtx` and
    /// all backing allocations.  `jit_advance_native` refreshes the pointer buffers
    /// after each `advance_jit()` call.
    ///
    /// TODO: compile native JIT function (Step 4)
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn build_stratum_stage4_native_runtime(
        &mut self,
        rules: &[&CRule],
    ) -> Option<crate::eval::jit::packed_helpers::StratumStage4NativeRuntime> {
        use crate::eval::jit::packed_helpers::{
            NativeHeadSpec, NativeScanSpec, StratumStage4NativeCtx, StratumStage4NativeRuntime,
        };
        use crate::eval::jit::storage::JitRelData;
        use crate::eval::relation::Relation;
        use crate::eval::specialized::PackedStorage;

        // Eagerly advance any packed relation that hasn't yet had `advance_jit()` called.
        // This serves two purposes:
        //   1. Populates `jit_native` so the `?` guards below don't return None.
        //   2. Moves initial EDB facts from `delta` → `recent`, so the JIT's upfront
        //      `jit_advance_native` sees `had_delta=false` and takes the cheap EDB path
        //      instead of rebuilding the full projection a second time.
        //
        // Without pre-advancing, the eager init would call `build_native_projection()` and
        // the upfront JIT advance would immediately rebuild it (double-build), adding ~50–100µs
        // of wasted work per engine run.
        {
            // Clone the JIT Arc once before borrowing self.relations mutably,
            // so cache lookups/updates don't conflict with the relation borrow.
            let jit_arc = self.jit.as_ref().map(Arc::clone);

            let all_rels: Vec<String> = rules
                .iter()
                .flat_map(|r| {
                    r.body.iter().filter_map(|item| {
                        if let CBodyItem::Clause(c) = item { Some(c.relation.clone()) } else { None }
                    })
                    .chain(r.heads.iter().map(|h| h.relation.clone()))
                })
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            for name in &all_rels {
                if let Some(Relation::Packed(ps)) = self.relations.get_mut(name) {
                    if ps.jit_native.is_none() {
                        // Cold path: move delta→recent, then build the native projection.
                        // Skip JitHashIndex rebuild: the native path uses JitColIndex (in
                        // jit_native) and never reads jit_indices / jit_recent_indices.
                        // Hash indices are never consumed anywhere. Skipping the 4×O(n log n)
                        // JitHashIndex builds saves ~20µs per stratum run for triangle n=20.
                        ps.advance_jit_skip_hash_indices();

                        // EDB total cache: for EDB non-sink relations, the sorted `total`
                        // JitRelData (with JitColIndex) can be reused across Engine instances
                        // when the same facts are loaded (same count + same data hash).
                        // Saves O(n log n) sort + JitColIndex rebuild per hot-bench iteration.
                        let native = if ps.jit_is_edb && !ps.jit_is_sink {
                            if let Some(arc) = &jit_arc {
                                let arity = ps.arity;
                                let count = ps.count;
                                let data_hash = {
                                    use std::hash::Hasher;
                                    let mut h = rustc_hash::FxHasher::default();
                                    let words = &ps.packed_data[0..count * arity.max(1)];
                                    let bytes = unsafe {
                                        std::slice::from_raw_parts(
                                            words.as_ptr() as *const u8,
                                            words.len() * 4,
                                        )
                                    };
                                    h.write(bytes);
                                    h.finish()
                                };
                                // Try cache hit (lock, check, clone, unlock).
                                let cached = {
                                    let jit = arc.lock().unwrap_or_else(|e| e.into_inner());
                                    jit.edb_native_total_cache.get(name).and_then(
                                        |(cached_count, cached_hash, cached_rel)| {
                                            if *cached_count == count && *cached_hash == data_hash {
                                                Some(unsafe {
                                                    crate::eval::jit::storage::clone_jit_rel_data_with_indices(
                                                        cached_rel, arity, true,
                                                    )
                                                })
                                            } else {
                                                None
                                            }
                                        },
                                    )
                                };
                                if let Some(cached_total) = cached {
                                    // Use pre-sorted total from cache — O(n) clone.
                                    ps.build_native_projection_with_total(cached_total)
                                } else {
                                    // Cache miss: build normally, then populate cache.
                                    let native = ps.build_native_projection();
                                    {
                                        let total_clone = unsafe {
                                            crate::eval::jit::storage::clone_jit_rel_data_with_indices(
                                                &native.total, arity, true,
                                            )
                                        };
                                        let mut jit = arc.lock().unwrap_or_else(|e| e.into_inner());
                                        jit.edb_native_total_cache
                                            .insert(name.clone(), (count, data_hash, total_clone));
                                    }
                                    native
                                }
                            } else {
                                ps.build_native_projection()
                            }
                        } else {
                            ps.build_native_projection()
                        };
                        ps.jit_native = Some(native);
                    } else {
                        // jit_native already exists.  Two cases:
                        // 1. Delta is non-empty: deep-cloned (PackedStorage::clone) and new facts
                        //    were inserted since the clone.  Advance to move delta→recent and
                        //    refresh jit_native.
                        // 2. build_indices mismatch: jit_native was built in an earlier stratum
                        //    where jit_is_edb was false (e.g. fact stratum where the relation was
                        //    a head), but this stratum uses the relation as an EDB body clause.
                        //    Rebuild jit_native with col_indices so inner-clause probes work.
                        if !ps.delta.is_empty() {
                            ps.advance_jit_skip_hash_indices();
                        }
                        let needs_indices = ps.jit_is_edb && !ps.jit_is_sink;
                        if needs_indices && !ps.jit_native.as_ref().unwrap().build_indices {
                            ps.jit_native = Some(ps.build_native_projection());
                        }
                    }
                }
            }
        }

        let mut scan_rels_vec:        Vec<*mut JitRelData>   = Vec::new();
        let mut total_rels_vec:       Vec<*mut JitRelData>   = Vec::new();
        let mut head_rels_vec:        Vec<*mut JitRelData>   = Vec::new();
        let mut head_total_rels_vec:  Vec<*mut JitRelData>   = Vec::new();
        let mut scan_specs_vec:       Vec<NativeScanSpec>    = Vec::new();
        let mut head_specs_vec:       Vec<NativeHeadSpec>    = Vec::new();

        for rule in rules {
            // Collect clause packed relations in order.
            let rule_clauses: Vec<&crate::eval::compiled::CClause> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => Some(c),
                    _ => None,
                })
                .collect();

            for clause in &rule_clauses {
                let ps = match self.relations.get(&clause.relation)? {
                    Relation::Packed(p) => p,
                    Relation::Generic(_) => return None,
                };
                let native = ps.jit_native.as_ref()?;
                let ps_ptr = ps as *const PackedStorage as *mut PackedStorage;

                // use_recent = 0 (total scan)
                let total_ptr = native.total.as_ref() as *const JitRelData as *mut JitRelData;
                scan_rels_vec.push(total_ptr);
                total_rels_vec.push(total_ptr);
                scan_specs_vec.push(NativeScanSpec { rel: ps_ptr, use_recent: 0, _pad: 0 });

                // use_recent = 1 (recent scan)
                let recent_ptr = native.recent.as_ref() as *const JitRelData as *mut JitRelData;
                scan_rels_vec.push(recent_ptr);
                total_rels_vec.push(total_ptr);
                scan_specs_vec.push(NativeScanSpec { rel: ps_ptr, use_recent: 1, _pad: 0 });
            }

            // Collect head packed relations.
            for head in &rule.heads {
                let ps = match self.relations.get(&head.relation)? {
                    Relation::Packed(p) => p,
                    Relation::Generic(_) => return None,
                };
                let native = ps.jit_native.as_ref()?;
                let ps_ptr = ps as *const PackedStorage as *mut PackedStorage;
                let new_ptr   = native.new.as_ref()   as *const JitRelData as *mut JitRelData;
                let total_ptr = native.total.as_ref() as *const JitRelData as *mut JitRelData;
                head_rels_vec.push(new_ptr);
                head_total_rels_vec.push(total_ptr);
                head_specs_vec.push(NativeHeadSpec { rel: ps_ptr });
            }
        }

        // Unique PackedStorage relations involved in this stratum (scan + head).
        // Only these relations need advancing during the fixpoint loop.
        // Including ALL relations is incorrect: jit_advance_native builds jit_native
        // for any relation with jit_native=None, but jit_is_edb may not be set correctly
        // for relations outside the current stratum, causing col_indices to be missing.
        let mut advance_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut advance_rels_vec: Vec<*mut PackedStorage> = Vec::new();
        for spec in &scan_specs_vec {
            let addr = spec.rel as usize;
            if advance_set.insert(addr) {
                advance_rels_vec.push(spec.rel);
            }
        }
        for spec in &head_specs_vec {
            let addr = spec.rel as usize;
            if advance_set.insert(addr) {
                advance_rels_vec.push(spec.rel);
            }
        }

        let n_scan_rels = scan_rels_vec.len() as u32;
        let n_head_rels = head_rels_vec.len() as u32;
        let n_advance_rels = advance_rels_vec.len() as u32;

        let mut scan_rels_buf:         Box<[*mut JitRelData]>    = scan_rels_vec.into_boxed_slice();
        let mut total_rels_buf:        Box<[*mut JitRelData]>    = total_rels_vec.into_boxed_slice();
        let mut head_rels_buf:         Box<[*mut JitRelData]>    = head_rels_vec.into_boxed_slice();
        let mut head_total_rels_buf:   Box<[*mut JitRelData]>    = head_total_rels_vec.into_boxed_slice();
        let     advance_rels_buf:      Box<[*mut PackedStorage]> = advance_rels_vec.into_boxed_slice();
        let     scan_specs_buf:        Box<[NativeScanSpec]>     = scan_specs_vec.into_boxed_slice();
        let     head_specs_buf:        Box<[NativeHeadSpec]>     = head_specs_vec.into_boxed_slice();

        let ctx = Box::new(StratumStage4NativeCtx {
            scan_rels:        scan_rels_buf.as_mut_ptr(),
            total_rels:       total_rels_buf.as_mut_ptr(),
            n_scan_rels,
            _pad0: 0,
            head_rels:        head_rels_buf.as_mut_ptr(),
            n_head_rels,
            _pad1: 0,
            advance_rels:     advance_rels_buf.as_ptr(),
            n_advance_rels,
            _pad2: 0,
            head_total_rels:  head_total_rels_buf.as_mut_ptr(),
            scan_specs:       scan_specs_buf.as_ptr(),
            head_specs:       head_specs_buf.as_ptr(),
        });

        Some(StratumStage4NativeRuntime {
            ctx,
            _scan_rels_buf:       scan_rels_buf,
            _total_rels_buf:      total_rels_buf,
            _head_rels_buf:       head_rels_buf,
            _head_total_rels_buf: head_total_rels_buf,
            _advance_rels_buf:    advance_rels_buf,
            _scan_specs_buf:      scan_specs_buf,
            _head_specs_buf:      head_specs_buf,
        })
    }

    /// Run a set of rules incrementally (delta-only, no initial full evaluation).
    ///
    /// Assumes new facts are already in `delta` on input relations.
    /// `owned` is the set of relation names this SCC writes to.
    /// Owned relations are advanced normally; input relations use peek-advance
    /// so their deltas survive for downstream SCCs.
    fn run_stratum_incremental(
        &mut self,
        rules: &[&CRule],
        owned: &FxHashSet<String>,
    ) -> Result<(), crate::eval::error::EvalError> {
        if rules.is_empty() {
            return Ok(());
        }

        // Sync interpreter state before incremental evaluation.
        #[cfg(feature = "specialized")]
        {
            use crate::eval::relation::Relation;
            for rel in self.relations.values_mut() {
                if let Relation::Packed(ps) = rel {
                    ps.ensure_interp_synced();
                }
            }
        }

        // Initial advance: owned relations consume delta, input relations peek
        let mut changed = false;
        for (name, rel) in &mut self.relations {
            if owned.contains(name) {
                if rel.advance() {
                    changed = true;
                }
            } else if rel.advance_peek() {
                changed = true;
            }
        }

        if !changed {
            return Ok(());
        }

        // Semi-naive loop: evaluate with use_recent=true, advance, repeat
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(crate::eval::error::EvalError::IterationLimitExceeded {
                    limit: self.max_iterations,
                });
            }

            for rule in rules {
                self.evaluate_rule(rule, true);
            }

            changed = false;
            for (name, rel) in &mut self.relations {
                if owned.contains(name) {
                    if rel.advance() {
                        changed = true;
                    }
                } else {
                    // Input relations: clear recent (was peeked), don't advance again
                    rel.clear_recent();
                }
            }

            if !changed {
                break;
            }
        }

        Ok(())
    }

    /// Evaluate a single rule.
    // TODO: propagate expression evaluation errors via EvalError
    fn evaluate_rule(&mut self, rule: &CRule, use_recent: bool) {
        if use_recent && !self.rule_has_recent_data(rule) {
            return;
        }

        let derived = self.derive_tuples(rule, use_recent);

        // Facts from empty-body rules (base facts) get the current source tag.
        // Derived facts from rules with bodies stay anonymous.
        let source = if rule.body.is_empty() {
            self.current_source
        } else {
            SourceId::ANONYMOUS
        };

        for (relation, tuple) in derived {
            // Internal derivation — relation and arity are guaranteed valid.
            let _ = self.insert_with_source(relation, tuple, source);
        }
    }

    /// Check if any relation referenced in the rule body has recent tuples.
    fn rule_has_recent_data(&self, rule: &CRule) -> bool {
        for item in &rule.body {
            let rel_name = match item {
                CBodyItem::Clause(clause) => &clause.relation,
                CBodyItem::Aggregation(agg) => &agg.relation,
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

    /// Derive all tuples from a rule using streaming evaluation.
    ///
    /// In semi-naive mode, for each clause position that has recent tuples,
    /// evaluate the rule with that clause using recent and all others using full.
    /// Uses recursive body processing with undo log to avoid materializing
    /// intermediate `Vec<Bindings>` between body items.
    fn derive_tuples<'a>(&self, rule: &'a CRule, use_recent: bool) -> Vec<(&'a str, Tuple)> {

        let mut results = Vec::new();
        let mut binding = Bindings::new(self.var_count);
        let mut undo = UndoLog::new();
        let mut scratch = Vec::new();

        if !use_recent {
            // Initial iteration: all clauses use full
            self.process_body_recursive(
                &rule.body,
                0,
                &rule.heads,
                &mut binding,
                &mut undo,
                None,
                &mut results,
                &mut scratch,
            );
            return results;
        }

        // Semi-naive: try each clause position with recent data
        for (idx, item) in rule.body.iter().enumerate() {
            let rel_name = match item {
                CBodyItem::Clause(c) => &c.relation,
                _ => continue,
            };
            if let Some(rel) = self.relations.get(rel_name)
                && rel.iter_recent().next().is_some()
            {
                self.process_body_recursive(
                    &rule.body,
                    0,
                    &rule.heads,
                    &mut binding,
                    &mut undo,
                    Some(idx),
                    &mut results,
                    &mut scratch,
                );
            }
        }

        results
    }

    /// Recursively process body items in streaming fashion.
    ///
    /// Instead of materializing intermediate `Vec<Bindings>`, processes each
    /// body item against the current binding in place. On match, recurses to
    /// the next body item. On complete match of all items, emits head tuples.
    /// Uses the undo log to roll back binding modifications after each attempt.
    #[allow(clippy::too_many_arguments)]
    fn process_body_recursive<'a>(
        &self,
        body: &'a [CBodyItem],
        offset: usize,
        heads: &'a [CHeadClause],
        binding: &mut Bindings,
        undo: &mut UndoLog,
        recent_clause_idx: Option<usize>,
        results: &mut Vec<(&'a str, Tuple)>,
        scratch: &mut Vec<Value>,
    ) {
        if offset >= body.len() {
            // All body items matched — emit head tuples
            for head in heads {
                if let Some(tuple) = self.eval_head_tuple(head, binding) {
                    results.push((&head.relation, tuple));
                }
            }
            return;
        }

        match &body[offset] {
            CBodyItem::Clause(clause) => {
                let use_recent = recent_clause_idx == Some(offset);
                self.stream_clause(
                    clause,
                    use_recent,
                    body,
                    offset,
                    heads,
                    binding,
                    undo,
                    recent_clause_idx,
                    results,
                    scratch,
                );
            }
            CBodyItem::Generator(generator) => {
                if let Some(range_val) = eval_cexpr(
                    &generator.expr,
                    binding,
                    Some(&self.type_registry),
                    &self.var_interner,
                ) && let Some(values) = expand_range(&range_val)
                {
                    for value in values {
                        let cp = undo.len();
                        if let Some(&var_id) = generator.vars.first() {
                            let old = binding.insert(var_id, value);
                            undo.push((var_id, old));
                        }
                        self.process_body_recursive(
                            body,
                            offset + 1,
                            heads,
                            binding,
                            undo,
                            recent_clause_idx,
                            results,
                            scratch,
                        );
                        rollback(binding, undo, cp);
                    }
                }
            }
            CBodyItem::Condition(cond) => {
                let cp = undo.len();
                if self.eval_condition(cond, binding, undo) {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                }
                rollback(binding, undo, cp);
            }
            CBodyItem::Aggregation(agg) => {
                self.stream_aggregation(
                    agg,
                    body,
                    offset,
                    heads,
                    binding,
                    undo,
                    recent_clause_idx,
                    results,
                    scratch,
                );
            }
        }
    }

    /// Stream clause matches, recursing to the next body item for each match.
    ///
    /// Uses index lookups when bound columns are available, with pre-filtering
    /// for multi-column matches. Uses delta-specific indices for recent lookups.
    #[allow(clippy::too_many_arguments)]
    fn stream_clause<'a>(
        &self,
        clause: &CClause,
        use_recent: bool,
        body: &'a [CBodyItem],
        offset: usize,
        heads: &'a [CHeadClause],
        binding: &mut Bindings,
        undo: &mut UndoLog,
        recent_clause_idx: Option<usize>,
        results: &mut Vec<(&'a str, Tuple)>,
        scratch: &mut Vec<Value>,
    ) {
        let Some(rel) = self.relations.get(&clause.relation) else {
            return;
        };

        // Fast path: all columns known bound at compile time and not using recent
        // → build expected tuple directly and do single membership check,
        // skipping find_bound_columns allocation entirely.
        if clause.all_args_bound && !use_recent {
            scratch.clear();
            let mut ok = true;
            for arg in &clause.args {
                match arg {
                    CClauseArg::Var(var_id) => {
                        if let Some(val) = binding.get(var_id) {
                            scratch.push(val.clone());
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    CClauseArg::Expr(expr) => {
                        if let Some(val) =
                            eval_cexpr(expr, binding, Some(&self.type_registry), &self.var_interner)
                        {
                            scratch.push(val);
                        } else {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if ok && rel.contains(scratch.as_slice()) {
                if clause.conditions.is_empty() {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                } else {
                    let cp = undo.len();
                    if self.check_clause_conditions(clause, binding, undo) {
                        self.process_body_recursive(
                            body,
                            offset + 1,
                            heads,
                            binding,
                            undo,
                            recent_clause_idx,
                            results,
                            scratch,
                        );
                    }
                    rollback(binding, undo, cp);
                }
            }
            return;
        }

        // Pre-computed single bound column fast path: skip find_bound_columns
        // allocation and match_clause overhead. Directly bind fresh vars.
        if clause.bound_cols.len() == 1 {
            let col = clause.bound_cols[0];
            let val = match &clause.args[col] {
                CClauseArg::Var(id) => {
                    let Some(v) = binding.get(id) else { return };
                    std::borrow::Cow::Borrowed(v)
                }
                CClauseArg::Expr(expr) => {
                    let Some(v) =
                        eval_cexpr(expr, binding, Some(&self.type_registry), &self.var_interner)
                    else {
                        return;
                    };
                    std::borrow::Cow::Owned(v)
                }
            };
            let indices = if use_recent {
                rel.lookup_recent(col, &val)
            } else {
                rel.lookup(col, &val)
            };
            for &idx in indices {
                let tuple = rel.get(idx);
                let cp = undo.len();
                // Bind fresh vars directly (no match_clause overhead)
                for &(c, var_id) in &clause.fresh_cols {
                    binding.insert(var_id, tuple[c].clone());
                    undo.push((var_id, None));
                }
                if self.check_clause_conditions(clause, binding, undo) {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                }
                rollback(binding, undo, cp);
            }
            return;
        }

        // Multi-bound or fallback: use runtime find_bound_columns
        let bound_cols = self.find_bound_columns(clause, binding);

        // Runtime fast path fallback: all columns happen to be bound
        // (catches cases the compile-time analysis missed, e.g. vars from if-let patterns).
        if !use_recent && bound_cols.len() == clause.args.len() {
            scratch.clear();
            scratch.extend(bound_cols.iter().map(|(_, val)| val.clone()));
            if rel.contains(scratch.as_slice()) {
                if clause.conditions.is_empty() {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                } else {
                    let cp = undo.len();
                    if self.check_clause_conditions(clause, binding, undo) {
                        self.process_body_recursive(
                            body,
                            offset + 1,
                            heads,
                            binding,
                            undo,
                            recent_clause_idx,
                            results,
                            scratch,
                        );
                    }
                    rollback(binding, undo, cp);
                }
            }
            return;
        }

        if !bound_cols.is_empty() {
            // Pick the most selective column for index lookup
            let (primary_pos, _) = bound_cols
                .iter()
                .enumerate()
                .min_by_key(|&(_, (col, val))| rel.lookup(*col, val).len())
                .expect("bound_cols is non-empty");
            let (primary_col, primary_val) = &bound_cols[primary_pos];
            let indices = if use_recent {
                rel.lookup_recent(*primary_col, primary_val)
            } else {
                rel.lookup(*primary_col, primary_val)
            };

            for &idx in indices {
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
                if self.match_clause(clause, tuple, binding, undo)
                    && self.check_clause_conditions(clause, binding, undo)
                {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                }
                rollback(binding, undo, cp);
            }
        } else if clause.bound_cols.is_empty() && !clause.fresh_cols.is_empty() {
            // All args are fresh vars (first clause in body, or all new vars).
            // Skip match_clause: directly bind fresh vars from tuples.
            //
            // Dedup optimization: when meaningful_fresh_cols is a proper subset of fresh_cols
            // (i.e., some fresh vars are wildcards never used downstream) and there are no
            // conditions referencing those wildcards, tuples differing only in wildcard
            // columns produce identical downstream results — we can skip the duplicates.
            let can_dedup = clause.conditions.is_empty()
                && !clause.meaningful_fresh_cols.is_empty()
                && clause.meaningful_fresh_cols.len() < clause.fresh_cols.len();
            if can_dedup && !use_recent {
                if clause.meaningful_fresh_cols.len() == 1 {
                    let (col, var_id) = clause.meaningful_fresh_cols[0];
                    let mut seen: FxHashSet<Value> = FxHashSet::default();
                    for tuple in rel.iter_full() {
                        let val = tuple[col].clone();
                        if seen.insert(val.clone()) {
                            let cp = undo.len();
                            binding.insert(var_id, val);
                            undo.push((var_id, None));
                            self.process_body_recursive(
                                body,
                                offset + 1,
                                heads,
                                binding,
                                undo,
                                recent_clause_idx,
                                results,
                                scratch,
                            );
                            rollback(binding, undo, cp);
                        }
                    }
                } else {
                    let mut seen: FxHashSet<Vec<Value>> = FxHashSet::default();
                    for tuple in rel.iter_full() {
                        let key: Vec<Value> = clause
                            .meaningful_fresh_cols
                            .iter()
                            .map(|&(c, _)| tuple[c].clone())
                            .collect();
                        if seen.insert(key) {
                            let cp = undo.len();
                            for &(c, var_id) in &clause.meaningful_fresh_cols {
                                binding.insert(var_id, tuple[c].clone());
                                undo.push((var_id, None));
                            }
                            self.process_body_recursive(
                                body,
                                offset + 1,
                                heads,
                                binding,
                                undo,
                                recent_clause_idx,
                                results,
                                scratch,
                            );
                            rollback(binding, undo, cp);
                        }
                    }
                }
                return;
            }
            macro_rules! bind_fresh_and_recurse {
                ($tuple:expr) => {{
                    let cp = undo.len();
                    for &(c, var_id) in &clause.fresh_cols {
                        binding.insert(var_id, $tuple[c].clone());
                        undo.push((var_id, None));
                    }
                    if self.check_clause_conditions(clause, binding, undo) {
                        self.process_body_recursive(
                            body,
                            offset + 1,
                            heads,
                            binding,
                            undo,
                            recent_clause_idx,
                            results,
                            scratch,
                        );
                    }
                    rollback(binding, undo, cp);
                }};
            }
            if use_recent {
                for tuple in rel.iter_recent() {
                    bind_fresh_and_recurse!(tuple);
                }
            } else {
                for tuple in rel.iter_full() {
                    bind_fresh_and_recurse!(tuple);
                }
            }
        } else if use_recent {
            // Fallback for complex patterns (expressions, repeated vars, etc.)
            for tuple in rel.iter_recent() {
                let cp = undo.len();
                if self.match_clause(clause, tuple, binding, undo)
                    && self.check_clause_conditions(clause, binding, undo)
                {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                }
                rollback(binding, undo, cp);
            }
        } else {
            for tuple in rel.iter_full() {
                let cp = undo.len();
                if self.match_clause(clause, tuple, binding, undo)
                    && self.check_clause_conditions(clause, binding, undo)
                {
                    self.process_body_recursive(
                        body,
                        offset + 1,
                        heads,
                        binding,
                        undo,
                        recent_clause_idx,
                        results,
                        scratch,
                    );
                }
                rollback(binding, undo, cp);
            }
        }
    }

    /// Find all clause columns that are already bound in the given bindings.
    ///
    /// Returns a list of (column_index, value) pairs for columns where
    /// the clause arg is a bound variable or evaluable expression.
    fn find_bound_columns(&self, clause: &CClause, binding: &Bindings) -> Vec<(usize, Value)> {
        let mut bound = Vec::with_capacity(clause.args.len());
        for (col, arg) in clause.args.iter().enumerate() {
            match arg {
                CClauseArg::Var(var_id) => {
                    if let Some(val) = binding.get(var_id) {
                        bound.push((col, val.clone()));
                    }
                }
                CClauseArg::Expr(expr) => {
                    if let Some(val) =
                        eval_cexpr(expr, binding, Some(&self.type_registry), &self.var_interner)
                    {
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
        clause: &CClause,
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
                CClauseArg::Var(var_id) => {
                    if let Some(existing) = bindings.get(var_id) {
                        if existing != value {
                            rollback(bindings, undo, cp);
                            return false;
                        }
                    } else {
                        bindings.insert(*var_id, value.clone());
                        undo.push((*var_id, None));
                    }
                }
                CClauseArg::Expr(expr) => {
                    if let Some(evaluated) = eval_cexpr(
                        expr,
                        bindings,
                        Some(&self.type_registry),
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
        clause: &CClause,
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

    /// Stream aggregation: collect matches, apply aggregator, recurse for each result.
    ///
    /// Uses index lookup when aggregation args have bound columns to avoid
    /// full relation scans. Falls back to full scan when no args are bound.
    #[allow(clippy::too_many_arguments)]
    fn stream_aggregation<'a>(
        &self,
        agg: &CAggregation,
        body: &'a [CBodyItem],
        offset: usize,
        heads: &'a [CHeadClause],
        binding: &mut Bindings,
        undo: &mut UndoLog,
        recent_clause_idx: Option<usize>,
        results: &mut Vec<(&'a str, Tuple)>,
        scratch: &mut Vec<Value>,
    ) {
        let Some(rel) = self.relations.get(&agg.relation) else {
            return;
        };

        // Fast path: all-Var args — skip binding/undo machinery, avoid per-tuple Vec alloc.
        let (agg_results, _flat_storage);
        if let Some(flat) = self.collect_agg_flat_vars(agg, binding) {
            let stride = agg.bound_vars.len().max(1);
            _flat_storage = flat;
            agg_results = apply_aggregator(
                &agg.aggregator_name,
                _flat_storage.chunks(stride),
            );
        } else {
            _flat_storage = Vec::new();
            let mut collected: Vec<Vec<Value>> = Vec::new();
            let bound_cols = self.find_bound_agg_columns(agg, binding);

            if !bound_cols.is_empty() {
                let (primary_pos, _) = bound_cols
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, (col, val))| rel.lookup(*col, val).len())
                    .expect("bound_cols is non-empty");
                let (primary_col, primary_val) = &bound_cols[primary_pos];
                let indices = rel.lookup(*primary_col, primary_val);

                for &idx in indices {
                    let tuple = rel.get(idx);
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
                    if self.match_agg_args(agg, tuple, binding, undo) {
                        collected.push(
                            agg.bound_vars
                                .iter()
                                .filter_map(|var_id| binding.get(var_id).cloned())
                                .collect(),
                        );
                    }
                    rollback(binding, undo, cp);
                }
            } else {
                for tuple in rel.iter_full() {
                    let cp = undo.len();
                    if self.match_agg_args(agg, tuple, binding, undo) {
                        collected.push(
                            agg.bound_vars
                                .iter()
                                .filter_map(|var_id| binding.get(var_id).cloned())
                                .collect(),
                        );
                    }
                    rollback(binding, undo, cp);
                }
            }

            agg_results = apply_aggregator(
                &agg.aggregator_name,
                collected.iter().map(|v| v.as_slice()),
            );
        }

        // Apply aggregator and recurse for each result

        for result_tuple in agg_results {
            let cp = undo.len();
            for (&var_id, val) in agg.result_vars.iter().zip(result_tuple) {
                let old = binding.insert(var_id, val);
                undo.push((var_id, old));
            }
            self.process_body_recursive(
                body,
                offset + 1,
                heads,
                binding,
                undo,
                recent_clause_idx,
                results,
                scratch,
            );
            rollback(binding, undo, cp);
        }
    }

    /// Fast-path: collect aggregation values when all agg.args are plain Vars.
    ///
    /// Returns `Some(flat)` where `flat` is a flat buffer of values, stride = `agg.bound_vars.len()`,
    /// each stride encoding one matching tuple's bound-var values — ready to pass to
    /// `apply_aggregator` via `.chunks(stride)`.  Returns `None` if any arg is an Expr.
    ///
    /// This avoids per-match binding mutations and inner `Vec<Value>` allocations.
    fn collect_agg_flat_vars(&self, agg: &CAggregation, binding: &Bindings) -> Option<Vec<Value>> {
        use crate::eval::compiled::CAggArg;

        // Require all args to be plain Vars.
        if !agg.args.iter().all(|a| matches!(a, CAggArg::Var(_))) {
            return None;
        }

        let rel = self.relations.get(&agg.relation)?;
        let n_bv = agg.bound_vars.len();

        // Per-column plan: either filter by a pre-bound value, or extract to a bound_var slot.
        enum ColPlan {
            Filter(Value),
            Extract(usize), // index into bound_vars
            Skip,           // free var not in bound_vars (wildcard-like, ignored)
        }
        let col_plans: Vec<ColPlan> = agg
            .args
            .iter()
            .map(|arg| {
                let CAggArg::Var(var_id) = arg else { unreachable!() };
                if let Some(val) = binding.get(var_id) {
                    ColPlan::Filter(val.clone())
                } else if let Some(pos) = agg.bound_vars.iter().position(|v| v == var_id) {
                    ColPlan::Extract(pos)
                } else {
                    ColPlan::Skip
                }
            })
            .collect();

        // Primary filter column for index lookup (first Filter column).
        let primary = col_plans.iter().enumerate().find_map(|(col, plan)| {
            if let ColPlan::Filter(val) = plan {
                Some((col, val))
            } else {
                None
            }
        });

        let mut flat: Vec<Value> = Vec::new();

        // Shared inline scan body: append one stride to flat if tuple passes all filters.
        macro_rules! scan_tuple {
            ($tuple:expr) => {{
                let tuple = $tuple;
                let start = flat.len();
                flat.resize(start + n_bv.max(1), Value::Unit);
                let mut ok = true;
                for (col, plan) in col_plans.iter().enumerate() {
                    match plan {
                        ColPlan::Filter(expected) => {
                            if tuple.get(col) != Some(expected) {
                                ok = false;
                                break;
                            }
                        }
                        ColPlan::Extract(pos) => {
                            if let Some(v) = tuple.get(col) {
                                flat[start + pos] = v.clone();
                            }
                        }
                        ColPlan::Skip => {}
                    }
                }
                if !ok {
                    flat.truncate(start);
                }
            }};
        }

        if let Some((primary_col, primary_val)) = primary {
            let indices = rel.lookup(primary_col, primary_val);
            flat.reserve(indices.len() * n_bv.max(1));
            for &idx in indices {
                scan_tuple!(rel.get(idx));
            }
        } else {
            flat.reserve(rel.len() * n_bv.max(1));
            for tuple in rel.iter_full() {
                scan_tuple!(tuple);
            }
        }

        Some(flat)
    }

    /// Find aggregation columns that are already bound in the given bindings.
    fn find_bound_agg_columns(
        &self,
        agg: &CAggregation,
        binding: &Bindings,
    ) -> Vec<(usize, Value)> {
        let mut bound = Vec::with_capacity(agg.args.len());
        for (col, arg) in agg.args.iter().enumerate() {
            match arg {
                CAggArg::Var(var_id) => {
                    if let Some(val) = binding.get(var_id) {
                        bound.push((col, val.clone()));
                    }
                }
                CAggArg::Expr(expr) => {
                    if let Some(val) =
                        eval_cexpr(expr, binding, Some(&self.type_registry), &self.var_interner)
                    {
                        bound.push((col, val));
                    }
                }
            }
        }
        bound
    }

    /// Match aggregation relation arguments against a tuple in place.
    ///
    /// Returns `true` if the match succeeded (bindings extended with new vars).
    /// On failure, rolls back any partial insertions via the undo log.
    fn match_agg_args(
        &self,
        agg: &CAggregation,
        tuple: &[Value],
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        if agg.args.len() != tuple.len() {
            return false;
        }

        let cp = undo.len();
        for (arg, value) in agg.args.iter().zip(tuple.iter()) {
            match arg {
                CAggArg::Var(var_id) => {
                    if let Some(existing) = bindings.get(var_id) {
                        if existing != value {
                            rollback(bindings, undo, cp);
                            return false;
                        }
                    } else {
                        bindings.insert(*var_id, value.clone());
                        undo.push((*var_id, None));
                    }
                }
                CAggArg::Expr(expr) => {
                    if let Some(evaluated) = eval_cexpr(
                        expr,
                        bindings,
                        Some(&self.type_registry),
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

    /// Evaluate a condition in place. Returns `true` on success (bindings may be extended).
    fn eval_condition(
        &self,
        cond: &CCondition,
        bindings: &mut Bindings,
        undo: &mut UndoLog,
    ) -> bool {
        match cond {
            CCondition::If(expr) => eval_cexpr(
                expr,
                bindings,
                Some(&self.type_registry),
                &self.var_interner,
            )
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
            CCondition::IfLet { pattern, expr } | CCondition::Let { pattern, expr } => {
                if let Some(value) = eval_cexpr(
                    expr,
                    bindings,
                    Some(&self.type_registry),
                    &self.var_interner,
                ) {
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
    ///
    /// Coerces numeric values to match the declared column types, so that
    /// unsuffixed integer literals (which default to i32) work correctly
    /// in relations declared with other integer types (e.g. u32).
    fn eval_head_tuple(&self, head: &CHeadClause, bindings: &Bindings) -> Option<Tuple> {
        let mut tuple = Vec::with_capacity(head.args.len());
        let declared = self.column_types.get(head.relation.as_str());

        for (col_idx, arg) in head.args.iter().enumerate() {
            let value = eval_cexpr(arg, bindings, Some(&self.type_registry), &self.var_interner)?;
            let value = if let Some(ty_name) = declared
                .and_then(|cols| cols.get(col_idx))
                .and_then(|t| t.as_deref())
            {
                coerce_to_col_type(value, ty_name)
            } else {
                value
            };
            tuple.push(value);
        }

        Some(tuple)
    }
}

/// Coerce a value to a declared column type if they are compatible integer types.
///
/// This handles the case where an unsuffixed literal (e.g. `0`) is evaluated as
/// `Value::I32` but the column is declared as `u32` or another integer type.
/// Only coerces between numeric types; leaves all other values unchanged.
///
/// If the cast fails (e.g., negative i32 to u32, or incompatible types), the
/// original value is kept as-is. This may cause join mismatches when a column
/// contains mixed types, which is acceptable — it matches Ascent's behavior
/// where type mismatches simply fail to unify.
fn coerce_to_col_type(value: Value, ty_name: &str) -> Value {
    value.cast_to(ty_name).unwrap_or(value)
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
    pat: &crate::ir::IrPattern,
    value: &Value,
    bindings: &mut Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
    undo: &mut UndoLog,
) -> bool {
    match pat {
        // Wildcard: always matches, binds nothing
        crate::ir::IrPattern::Wild => true,

        // Variable binding: `x` or `x @ subpat`
        crate::ir::IrPattern::Var(name, sub_pat) => {
            // Handle `_` identifiers as wildcards
            if name == "_" {
                return true;
            }
            // If there's a subpattern (`name @ pattern`), match it too
            if let Some(sub) = sub_pat
                && !match_pattern(sub, value, bindings, registry, interner, undo)
            {
                return false;
            }
            let var_id = interner.intern(name);
            let old = bindings.insert(var_id, value.clone());
            undo.push((var_id, old));
            true
        }

        // Literal pattern: `42`, `true`, `'a'`
        crate::ir::IrPattern::Lit(ir_lit) => {
            let lit_val = crate::eval::compiled::ir_lit_to_value(ir_lit);
            lit_val == *value
        }

        // Tuple pattern: `(a, b, c)`
        crate::ir::IrPattern::Tuple(pats) => {
            if let Value::Tuple(vals) = value {
                if vals.len() != pats.len() {
                    return false;
                }
                let cp = undo.len();
                for (pat_elem, val) in pats.iter().zip(vals.iter()) {
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
        crate::ir::IrPattern::TupleStruct(path_str, fields) => match path_str.as_str() {
            "Some" => {
                if let Value::Option(Some(inner)) = value
                    && fields.len() == 1
                {
                    match_pattern(&fields[0], inner, bindings, registry, interner, undo)
                } else {
                    false
                }
            }
            "Dual" => {
                if let Value::Dual(inner) = value
                    && fields.len() == 1
                {
                    match_pattern(&fields[0], inner, bindings, registry, interner, undo)
                } else {
                    false
                }
            }
            "None" => matches!(value, Value::Option(None)),
            _ => {
                if let Some(reg) = registry
                    && let Some(destructor) = reg.destructor(path_str)
                    && let Some(field_vals) = destructor(value)
                    && field_vals.len() == fields.len()
                {
                    let cp = undo.len();
                    for (pat_elem, val) in fields.iter().zip(field_vals.iter()) {
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
        },

        // Path pattern: `None`, `true`, `false`
        crate::ir::IrPattern::Path(path_str) => match path_str.as_str() {
            "None" => matches!(value, Value::Option(None)),
            "true" => matches!(value, Value::Bool(true)),
            "false" => matches!(value, Value::Bool(false)),
            _ => false,
        },

        // Or pattern: `A | B`
        crate::ir::IrPattern::Or(cases) => {
            let cp = undo.len();
            for case in cases {
                if match_pattern(case, value, bindings, registry, interner, undo) {
                    return true;
                }
                rollback(bindings, undo, cp);
            }
            false
        }

        // Reference pattern: `&x` — in Datalog context, match the inner pattern
        crate::ir::IrPattern::Ref(inner) => {
            match_pattern(inner, value, bindings, registry, interner, undo)
        }
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
        use crate::eval::serde_bridge::{from_values, to_values};
        self.register_type(
            name,
            |args| from_values::<T>(args).ok().map(Value::custom),
            |val| val.downcast_custom::<T>().and_then(|t| to_values(t).ok()),
        );
    }
}

#[cfg(test)]
#[allow(unused_mut)]
mod tests {
    use super::*;
    use crate::ir::Program;
    use crate::syntax::AscentProgram;

    fn run_program(input: &str) -> Engine {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.run().unwrap();
        engine
    }

    fn run_with_facts(input: &str, facts: Vec<(&str, Vec<Tuple>)>) -> Engine {
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);

        for (rel, tuples) in facts {
            for tuple in tuples {
                engine.insert(rel, tuple).unwrap();
            }
        }

        engine.run().unwrap();
        engine
    }

    #[test]
    fn test_unsuffixed_literal_coercion() {
        // Unsuffixed literals default to i32 in the evaluator, but should be
        // coerced to the declared column type (e.g. u32) at head insertion time.
        let mut engine = run_program(
            r#"
            relation node(u32);
            relation edge(u32, u32);
            relation path(u32, u32);
            node(0);
            node(1);
            edge(0, 1);
            path(x, y) <-- edge(x, y);
        "#,
        );

        let node = engine.relation("node").unwrap();
        assert!(node.contains(&[Value::U32(0)]));
        assert!(node.contains(&[Value::U32(1)]));

        let path = engine.relation("path").unwrap();
        assert!(path.contains(&[Value::U32(0), Value::U32(1)]));
    }

    #[test]
    fn test_fact_with_literals() {
        let mut engine = run_program(
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
        let mut engine = run_with_facts(
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
        let mut engine = run_program(
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
        let mut engine = run_with_facts(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let mut engine = run_program(
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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);

        engine.insert(
            "point",
            vec![Value::I32(1), Value::custom(Point { x: 10, y: 20 })],
        ).unwrap();
        engine.insert(
            "point",
            vec![Value::I32(2), Value::custom(Point { x: 30, y: 40 })],
        ).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);

        let p = Value::custom(Point { x: 1, y: 2 });
        engine.insert("r", vec![Value::I32(1), p.clone()]).unwrap();
        engine.insert("s", vec![p, Value::I32(99)]).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);

        engine.register_type(
            "Point",
            |args| {
                let x = args.first()?.as_i32()?;
                let y = args.get(1)?.as_i32()?;
                Some(Value::custom(Point { x, y }))
            },
            |val| {
                let p = val.downcast_custom::<Point>()?;
                Some(vec![Value::I32(p.x), Value::I32(p.y)])
            },
        );

        engine.run().unwrap();

        let data = engine.relation("data").unwrap();
        assert_eq!(data.len(), 2);
        assert!(data.contains(&[Value::I32(1), Value::custom(Point { x: 10, y: 20 })]));
        assert!(data.contains(&[Value::I32(2), Value::custom(Point { x: 30, y: 40 })]));
    }

    #[test]
    fn test_downcast_custom() {
        let v = Value::custom(Point { x: 5, y: 10 });
        let point = v.downcast_custom::<Point>().unwrap();
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
                let p = val.downcast_custom::<Point>()?;
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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 10, y: 20 })],
        ).unwrap();
        engine.insert(
            "data",
            vec![Value::I32(2), Value::custom(Point { x: 30, y: 40 })],
        ).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 5, y: 15 })],
        ).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        register_point(&mut engine);

        engine.insert(
            "data",
            vec![Value::I32(1), Value::custom(Point { x: 7, y: 99 })],
        ).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        register_point(&mut engine);

        // Insert a plain i32 instead of a Point — should not match
        engine.insert("data", vec![Value::I32(1), Value::I32(42)]).unwrap();
        engine.run().unwrap();

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
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        register_point(&mut engine);
        engine.run().unwrap();

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
            let program = Program::from_ast(ast).expect("lowering should succeed");
            let mut engine = Engine::new(program);
            engine.register_serde_type::<Point>("Point");
            engine.run().unwrap();

            let unwrapped = engine.relation("unwrapped").unwrap();
            assert_eq!(unwrapped.len(), 2);
            assert!(unwrapped.contains(&[Value::I32(1), Value::I32(10), Value::I32(20)]));
            assert!(unwrapped.contains(&[Value::I32(2), Value::I32(30), Value::I32(40)]));
        }
    }

    // ─── Incremental re-evaluation tests ────────────────────────────

    #[test]
    fn test_run_incremental_empty_dirty_set_is_noop() {
        let input = r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
        engine.run().unwrap();

        let rederived =
            engine.run_incremental(&[], &[]).unwrap();
        assert!(rederived.is_empty());
        // path should still have its data
        assert_eq!(engine.relation("path").unwrap().len(), 1);
    }

    #[test]
    fn test_run_incremental_retract_and_rederive() {
        let input = r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);

        let src = engine.intern_source("initial");
        engine
            .insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], src)
            .unwrap();
        engine
            .insert_with_source("edge", vec![Value::I32(2), Value::I32(3)], src)
            .unwrap();
        engine.run().unwrap();
        assert_eq!(engine.relation("path").unwrap().len(), 3); // (1,2), (2,3), (1,3)

        // Retract and add different edges
        engine.retract_source(src);
        let src2 = engine.intern_source("updated");
        engine
            .insert_with_source("edge", vec![Value::I32(10), Value::I32(20)], src2)
            .unwrap();

        let rederived = engine.run_incremental(&["edge"], &["edge"]).unwrap();

        assert!(rederived.contains("path"));
        let path = engine.relation("path").unwrap();
        assert_eq!(path.len(), 1);
        assert!(path.contains(&[Value::I32(10), Value::I32(20)]));
    }

    #[test]
    fn test_run_incremental_multi_level_propagation() {
        // A → B → C chain: dirtying A should re-derive B and C
        let input = r#"
            relation a(i32);
            relation b(i32);
            relation c(i32);
            relation unrelated(i32);
            b(x) <-- a(x);
            c(x) <-- b(x);
            unrelated(42);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("a", vec![Value::I32(1)]).unwrap();
        engine.run().unwrap();

        assert_eq!(engine.relation("b").unwrap().len(), 1);
        assert_eq!(engine.relation("c").unwrap().len(), 1);
        assert_eq!(engine.relation("unrelated").unwrap().len(), 1);

        // Change a's contents
        engine.relation_mut("a").unwrap().clear();
        engine.insert("a", vec![Value::I32(2)]).unwrap();
        engine.insert("a", vec![Value::I32(3)]).unwrap();

        let rederived = engine.run_incremental(&["a"], &["a"]).unwrap();

        assert!(rederived.contains("b"));
        assert!(rederived.contains("c"));
        // unrelated should NOT be in rederived
        assert!(!rederived.contains("unrelated"));

        let b = engine.relation("b").unwrap();
        assert_eq!(b.len(), 2);
        assert!(b.contains(&[Value::I32(2)]));
        assert!(b.contains(&[Value::I32(3)]));

        let c = engine.relation("c").unwrap();
        assert_eq!(c.len(), 2);

        // unrelated should still have its original data
        assert_eq!(engine.relation("unrelated").unwrap().len(), 1);
    }

    #[test]
    fn test_run_incremental_returns_affected_relations() {
        let input = r#"
            relation src(i32);
            relation derived(i32);
            derived(x) <-- src(x);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("src", vec![Value::I32(1)]).unwrap();
        engine.run().unwrap();

        let rederived = engine.run_incremental(&["src"], &[]).unwrap();
        assert_eq!(rederived, FxHashSet::from_iter(["derived".to_string()]));
    }

    // ─── Incremental addition (delta-only) tests ─────────────────────

    #[test]
    fn test_incremental_addition_monotone_preserves_old_and_adds_new() {
        // Monotone transitive closure: adding edges should preserve old paths
        // and derive only new ones via delta-only evaluation.
        let input = r#"
            relation edge(i32, i32);
            relation path(i32, i32);
            path(x, y) <-- edge(x, y);
            path(x, z) <-- edge(x, y), path(y, z);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
        engine.insert("edge", vec![Value::I32(2), Value::I32(3)]).unwrap();
        engine.run().unwrap();
        assert_eq!(engine.relation("path").unwrap().len(), 3); // (1,2), (2,3), (1,3)

        // Add a new edge — old paths should be preserved, new ones derived
        engine.insert("edge", vec![Value::I32(3), Value::I32(4)]).unwrap();
        engine.run_incremental(&["edge"], &[]).unwrap();

        let path = engine.relation("path").unwrap();
        // Old: (1,2), (2,3), (1,3)
        // New: (3,4), (2,4), (1,4)
        assert_eq!(path.len(), 6);
        assert!(path.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(path.contains(&[Value::I32(2), Value::I32(3)]));
        assert!(path.contains(&[Value::I32(1), Value::I32(3)]));
        assert!(path.contains(&[Value::I32(3), Value::I32(4)]));
        assert!(path.contains(&[Value::I32(2), Value::I32(4)]));
        assert!(path.contains(&[Value::I32(1), Value::I32(4)]));
    }

    #[test]
    fn test_incremental_addition_negation_clears_and_rederives() {
        // Non-monotone SCC (contains negation) should clear and re-derive.
        let input = r#"
            relation node(i32);
            relation excluded(i32);
            relation included(i32);
            included(x) <-- node(x), !excluded(x);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("node", vec![Value::I32(1)]).unwrap();
        engine.insert("node", vec![Value::I32(2)]).unwrap();
        engine.insert("node", vec![Value::I32(3)]).unwrap();
        engine.run().unwrap();
        assert_eq!(engine.relation("included").unwrap().len(), 3);

        // Add an exclusion — negation SCC must re-derive from scratch
        engine.insert("excluded", vec![Value::I32(2)]).unwrap();
        engine.run_incremental(&["excluded"], &[]).unwrap();

        let included = engine.relation("included").unwrap();
        assert_eq!(included.len(), 2);
        assert!(included.contains(&[Value::I32(1)]));
        assert!(included.contains(&[Value::I32(3)]));
        assert!(!included.contains(&[Value::I32(2)]));
    }

    #[test]
    fn test_incremental_monotone_feeds_nonmonotone() {
        // Monotone SCC A feeds non-monotone SCC B.
        // A should use delta-only; B should clear and re-derive.
        let input = r#"
            relation raw(i32);
            relation derived(i32);
            relation excluded(i32);
            relation filtered(i32);
            derived(x) <-- raw(x);
            filtered(x) <-- derived(x), !excluded(x);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("raw", vec![Value::I32(1)]).unwrap();
        engine.insert("raw", vec![Value::I32(2)]).unwrap();
        engine.insert("excluded", vec![Value::I32(2)]).unwrap();
        engine.run().unwrap();

        let filtered = engine.relation("filtered").unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains(&[Value::I32(1)]));

        // Add more raw data — derived uses delta, filtered re-derives
        engine.insert("raw", vec![Value::I32(3)]).unwrap();
        engine.run_incremental(&["raw"], &[]).unwrap();

        let derived = engine.relation("derived").unwrap();
        assert_eq!(derived.len(), 3);

        let filtered = engine.relation("filtered").unwrap();
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&[Value::I32(1)]));
        assert!(filtered.contains(&[Value::I32(3)]));
    }

    #[test]
    fn test_incremental_monotone_downstream_of_nonmonotone_falls_back() {
        // Non-monotone SCC clears relation R, monotone SCC downstream reads R.
        // The downstream monotone SCC should fall back to full eval.
        let input = r#"
            relation node(i32);
            relation excluded(i32);
            relation kept(i32);
            relation doubled(i32);
            kept(x) <-- node(x), !excluded(x);
            doubled(x) <-- kept(x);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("node", vec![Value::I32(1)]).unwrap();
        engine.insert("node", vec![Value::I32(2)]).unwrap();
        engine.insert("node", vec![Value::I32(3)]).unwrap();
        engine.run().unwrap();
        assert_eq!(engine.relation("doubled").unwrap().len(), 3);

        // Exclude node 2 — kept clears (non-monotone), doubled must also re-derive
        engine.insert("excluded", vec![Value::I32(2)]).unwrap();
        engine.run_incremental(&["excluded"], &[]).unwrap();

        let kept = engine.relation("kept").unwrap();
        assert_eq!(kept.len(), 2);
        let doubled = engine.relation("doubled").unwrap();
        assert_eq!(doubled.len(), 2);
        assert!(doubled.contains(&[Value::I32(1)]));
        assert!(doubled.contains(&[Value::I32(3)]));
        assert!(!doubled.contains(&[Value::I32(2)]));
    }

    #[test]
    fn test_incremental_addition_no_change_is_noop() {
        // If no new deltas exist in a monotone SCC, run_stratum_incremental
        // should return immediately.
        let input = r#"
            relation src(i32);
            relation dst(i32);
            dst(x) <-- src(x);
        "#;
        let ast: AscentProgram = syn::parse_str(input).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let mut engine = Engine::new(program);
        engine.insert("src", vec![Value::I32(1)]).unwrap();
        engine.run().unwrap();
        assert_eq!(engine.relation("dst").unwrap().len(), 1);

        // Insert same fact again (deduplication means no actual delta)
        engine.insert("src", vec![Value::I32(1)]).unwrap();
        engine.run_incremental(&["src"], &[]).unwrap();

        // Should still have exactly 1 tuple
        assert_eq!(engine.relation("dst").unwrap().len(), 1);
    }
}

#[cfg(test)]
#[cfg(all(feature = "jit", feature = "specialized"))]
#[allow(unused_must_use)]
mod jit_hot_tests {
    use super::*;
    use crate::syntax::AscentProgram;
    use crate::ir::Program;

    fn run_shared_jit(source: &str) {
        let ast: AscentProgram = syn::parse_str(source).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        // warmup
        let mut warmup = Engine::new(program.clone());
        warmup.enable_jit().expect("JIT init should succeed");
        warmup.run();
        let jit = warmup.share_jit_compiler().unwrap();
        // hot run
        let mut engine = Engine::new(program);
        engine.set_jit_compiler(jit);
        engine.run().unwrap();
    }

    #[test]
    fn tc_shared_jit() {
        let mut source = String::from("relation edge(i32,i32);\nrelation path(i32,i32);\n");
        for i in 1..10 { source.push_str(&format!("edge({},{});\n", i, i+1)); }
        source.push_str("path(x,y) <-- edge(x,y);\npath(x,z) <-- edge(x,y),path(y,z);\n");
        run_shared_jit(&source);
    }

    #[test]
    fn tc_shared_jit_external_facts() {
        // Verifies shared-JIT correctness across multiple hot runs with external facts.
        // Use small n and few iterations to keep debug-mode runtime acceptable.
        let source = "relation edge(i32,i32);\nrelation path(i32,i32);\npath(x,y) <-- edge(x,y);\npath(x,z) <-- edge(x,y),path(y,z);\n";
        let ast: AscentProgram = syn::parse_str(source).unwrap();
        let program = Program::from_ast(ast).expect("lowering should succeed");
        let n = 10i32;
        // warmup
        let mut warmup = Engine::new(program.clone());
        warmup.enable_jit().expect("JIT init should succeed");
        for i in 1..n {
            warmup.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
        }
        warmup.run();
        let jit = warmup.share_jit_compiler().unwrap();
        // hot runs: verify correctness across several iterations
        for _ in 0..10 {
            let mut engine = Engine::new(program.clone());
            engine.set_jit_compiler(jit.clone());
            for i in 1..n {
                engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]).unwrap();
            }
            engine.run().unwrap();
            let rel = engine.relation("path").unwrap();
            let expected = (n * (n - 1) / 2) as usize;
            assert_eq!(
                rel.len(),
                expected,
                "expected {expected} path tuples from shared jit run"
            );
        }
    }

}

/// Return Stage 4 runtimes to the JIT pool when an engine is dropped.
/// On the next hot-bench iteration, `repopulate_runtime` rewrites engine-specific pointers
/// into the already-allocated boxes (zero allocations vs the ~9 Box allocs in a fresh build).
#[cfg(all(feature = "jit", feature = "specialized"))]
impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(jit_cell) = self.jit.as_ref()
            && !self.stratum_stage4_cache.is_empty()
            && let Ok(mut jit) = jit_cell.lock()
        {
            for (key, runtime) in self.stratum_stage4_cache.drain() {
                // Keep at most one runtime per stratum in the pool.
                jit.stratum_runtime_pool.entry(key).or_insert(runtime);
            }
        }
    }
}
