//! Semi-naive evaluation engine.

use std::cell::RefCell;
use std::fmt;
use std::sync::{Arc, Mutex};

use ascent_ir::{BodyItem, Program};
use petgraph::algo::{condensation, toposort};
use petgraph::graph::DiGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::aggregators::apply_aggregator;
use crate::compiled::{
    CAggArg, CAggregation, CBodyItem, CClause, CClauseArg, CCondition, CHeadClause, CRule,
    compile_rule, eval_cexpr,
};
use crate::expr::{eval_expr, expand_range};
use crate::relation::{Relation, SourceId};
use crate::value::{DynValue, Tuple, Value};

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

/// Buffer type for per-rule JIT results: (head_rel_idx, packed_tuple).
#[cfg(all(feature = "jit", feature = "specialized"))]
type RuleResultsBuf = Vec<(usize, Vec<u32>)>;

/// Pinned heap allocations backing a stratum meta-function's context.
///
/// Raw pointers inside `StratumMetaCtx` point into these allocations, so they
/// must all outlive the `StratumMetaCtx` they support.  The struct is intentionally
/// not `Send`/`Sync`; Engine is single-threaded.
#[cfg(all(feature = "jit", feature = "specialized"))]
// Box<Vec<T>> is intentional: stable heap address for raw-pointer aliasing.
#[allow(dead_code, clippy::vec_box)]
struct StratumMetaRuntime {
    meta_ctx: Box<crate::jit::packed_helpers::StratumMetaCtx>,
    // Fields prefixed with `_` are kept alive for their raw pointers; not read directly.
    _per_rule_bindings: Vec<Box<[u32]>>,
    _per_rule_results: Vec<Box<RuleResultsBuf>>,
    _per_rule_clause_rels: Vec<Box<[*const crate::specialized::PackedStorage]>>,
    _per_rule_ctxs: Vec<Box<crate::jit::packed_helpers::PackedJitContext>>,
    _full_fns: Box<[crate::jit::packed_helpers::PackedJitFn]>,
    _full_ctx_ptrs: Box<[*mut crate::jit::packed_helpers::PackedJitContext]>,
    _recent_fns: Box<[crate::jit::packed_helpers::PackedJitFn]>,
    _recent_ctx_ptrs: Box<[*mut crate::jit::packed_helpers::PackedJitContext]>,
    _flusher: Box<crate::jit::packed_helpers::StratumFlusher>,
}

#[cfg(all(feature = "jit", feature = "specialized"))]
impl std::fmt::Debug for StratumMetaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StratumMetaRuntime").finish_non_exhaustive()
    }
}

/// Pinned runtime data for a Stage 3 stratum function (direct-insert, no results buffer).
/// All raw pointers in `stage3_ctx` point into the boxes below.
#[cfg(all(feature = "jit", feature = "specialized"))]
#[allow(dead_code, clippy::vec_box)]
struct StratumStage3Runtime {
    stage3_ctx: Box<crate::jit::packed_helpers::StratumStage3Ctx>,
    _per_rule_clause_rels: Vec<Box<[*const crate::specialized::PackedStorage]>>,
    _per_rule_head_rels: Vec<Box<[*mut crate::specialized::PackedStorage]>>,
    _per_rule_ctxs: Vec<Box<crate::jit::packed_helpers::PackedJitContextV3>>,
    _full_fns: Box<[crate::jit::packed_helpers::PackedJitFnV3]>,
    _full_ctx_ptrs: Box<[*mut crate::jit::packed_helpers::PackedJitContextV3]>,
    _recent_fns: Box<[crate::jit::packed_helpers::PackedJitFnV3]>,
    _recent_ctx_ptrs: Box<[*mut crate::jit::packed_helpers::PackedJitContextV3]>,
    _all_rels: Box<[*mut crate::specialized::PackedStorage]>,
    /// Per-rule dedup handle pointer arrays (one *mut JitDedupHandle per head relation per rule).
    _per_rule_dedup_handles: Vec<Box<[*mut crate::jit_index::JitDedupHandle]>>,
}

#[cfg(all(feature = "jit", feature = "specialized"))]
impl std::fmt::Debug for StratumStage3Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StratumStage3Runtime").finish_non_exhaustive()
    }
}

/// Pinned runtime data for a Stage 4 stratum function (inlined rule bodies).
/// All raw pointers in `stage4_ctx` point into the boxes below.
#[cfg(all(feature = "jit", feature = "specialized"))]
#[allow(dead_code, clippy::vec_box)]
struct StratumStage4Runtime {
    stage4_ctx: Box<crate::jit::packed_helpers::StratumStage4Ctx>,
    _per_rule_clause_rels: Vec<Box<[*const crate::specialized::PackedStorage]>>,
    _per_rule_head_rels: Vec<Box<[*mut crate::specialized::PackedStorage]>>,
    _per_rule_ctxs: Vec<Box<crate::jit::packed_helpers::PackedJitContextV3>>,
    _rule_ctx_ptrs: Box<[*mut crate::jit::packed_helpers::PackedJitContextV3]>,
    _all_rels: Box<[*mut crate::specialized::PackedStorage]>,
    /// Flat buffer of all JitLookupHandle for all rules (handles_buf in ctx).
    _handles_buf: Box<[crate::jit_index::JitLookupHandle]>,
    /// Parallel spec array for handle refresh (lookup_specs in ctx).
    _lookup_specs: Box<[crate::jit::packed_helpers::LookupSpec]>,
    /// Per-rule dedup handle pointer arrays (one *mut JitDedupHandle per head relation per rule).
    _per_rule_dedup_handles: Vec<Box<[*mut crate::jit_index::JitDedupHandle]>>,
    /// Owned JitTupleSet boxes (one per fully-bound inner clause).
    _tuple_sets: Vec<Box<crate::jit::storage::JitTupleSet>>,
    /// Flat pointer array parallel to handles_buf (tuple_sets_buf in ctx).
    _tuple_sets_buf: Box<[*const crate::jit::storage::JitTupleSet]>,
    /// Owned JitHeadBuf boxes (one per (rule, head) pair).
    _head_buf_boxes: Vec<Box<crate::jit::packed_helpers::JitHeadBuf>>,
    /// Backing data storage for each JitHeadBuf (kept alive alongside the buf).
    _head_buf_data: Vec<Vec<u32>>,
    /// Flat pointer array: head_write_bufs in ctx.
    _head_write_bufs: Box<[*mut crate::jit::packed_helpers::JitHeadBuf]>,
    /// Flat pointer array: head_rel_ptrs in ctx.
    _head_rel_ptrs: Box<[*mut crate::specialized::PackedStorage]>,
    /// Optional native fast-path runtime (Step 3). `None` if any relation lacks `jit_native`.
    /// TODO: compile native JIT function (Step 4)
    stage4_native_runtime: Option<crate::jit::packed_helpers::StratumStage4NativeRuntime>,
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
    /// Storage for each relation.
    relations: FxHashMap<String, Relation>,
    /// Declared column types per relation (primitive type name, or None for complex types).
    col_types: FxHashMap<String, Vec<Option<String>>>,
    /// Registry of custom type constructors.
    pub type_registry: TypeRegistry,
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
    jit: Option<Arc<Mutex<crate::jit::JitCompiler>>>,
    /// Cache of stratum meta-function runtime contexts.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    stratum_meta_cache: FxHashMap<usize, StratumMetaRuntime>,
    /// Cache of Stage 3 stratum runtime contexts.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    stratum_stage3_cache: FxHashMap<usize, StratumStage3Runtime>,
    /// Cache of Stage 4 stratum runtime contexts (inlined rule bodies).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    stratum_stage4_cache: FxHashMap<usize, StratumStage4Runtime>,
}

/// An opaque, shareable handle to a pre-compiled JIT compiler.
///
/// Obtained from [`Engine::share_jit_compiler`] and passed to
/// [`Engine::with_jit_compiler`] to avoid recompilation across engine instances.
#[cfg(feature = "jit")]
#[derive(Clone)]
pub struct SharedJitCompiler(Arc<Mutex<crate::jit::JitCompiler>>);

impl Engine {
    /// Create a new engine from a program.
    pub fn new(program: &Program) -> Self {
        let mut relations = FxHashMap::default();
        let mut col_types = FxHashMap::default();

        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            let types: Vec<Option<String>> = rel
                .column_types
                .iter()
                .map(|ty| {
                    if let syn::Type::Path(tp) = ty {
                        tp.path.get_ident().map(|id| id.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            relations.insert(
                name.clone(),
                Relation::new_auto(arity, rel.is_lattice, &types),
            );
            col_types.insert(name.clone(), types);
        }

        Engine {
            relations,
            col_types,
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
            stratum_meta_cache: FxHashMap::default(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            stratum_stage3_cache: FxHashMap::default(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            stratum_stage4_cache: FxHashMap::default(),
        }
    }

    /// Enable JIT compilation for eligible rules.
    #[cfg(feature = "jit")]
    pub fn enable_jit(&mut self) {
        if self.jit.is_none() {
            match crate::jit::JitCompiler::new() {
                Ok(compiler) => self.jit = Some(Arc::new(Mutex::new(compiler))),
                Err(e) => eprintln!("JIT init failed: {e}"),
            }
        }
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
    pub fn with_jit_compiler(&mut self, jit: SharedJitCompiler) {
        self.jit = Some(jit.0);
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
    pub fn relation(&self, name: &str) -> Option<&Relation> {
        self.relations.get(name)
    }

    /// Get a mutable relation by name.
    pub fn relation_mut(&mut self, name: &str) -> Option<&mut Relation> {
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
    }

    /// Insert a tuple into a relation (untagged / [`SourceId::ANONYMOUS`]).
    pub fn insert(&mut self, relation: &str, tuple: Tuple) -> bool {
        if let Some(rel) = self.relations.get_mut(relation) {
            rel.insert(tuple)
        } else {
            false
        }
    }

    /// Insert a tuple tagged with a source.
    pub fn insert_with_source(&mut self, relation: &str, tuple: Tuple, source: SourceId) -> bool {
        if let Some(rel) = self.relations.get_mut(relation) {
            rel.insert_with_source(tuple, source)
        } else {
            false
        }
    }

    /// Intern a source name, returning a stable [`SourceId`].
    ///
    /// Repeated calls with the same name return the same ID.
    pub fn intern_source(&mut self, name: &str) -> SourceId {
        if let Some(&id) = self.source_names.get(name) {
            return id;
        }
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
    pub fn run(&mut self, program: &Program) {
        self.ensure_stratification(program);
        let strat = self.stratification.as_ref().unwrap();

        // Clone the SCC indices to avoid borrow conflict with self.
        let sccs: Vec<Vec<usize>> = strat.sccs.clone();
        let compiled: Vec<CRule> = strat.compiled.clone();

        self.var_count = self.var_interner.len();
        for (scc_idx, scc_indices) in sccs.iter().enumerate() {
            let rules: Vec<&CRule> = scc_indices.iter().map(|&i| &compiled[i]).collect();
            self.run_stratum(&rules, scc_idx, scc_indices);
        }

        // Clear recent/delta so stale data doesn't confuse the next run.
        for rel in self.relations.values_mut() {
            rel.clear_recent();
        }
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
        program: &Program,
        dirty: &FxHashSet<String>,
        retracted: &FxHashSet<String>,
    ) -> FxHashSet<String> {
        if dirty.is_empty() {
            return FxHashSet::default();
        }

        self.ensure_stratification(program);
        let strat = self.stratification.as_ref().unwrap();

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
                self.run_stratum(&rules, scc_idx, scc_indices);
            } else {
                // Snapshot counts of owned relations before incremental eval
                let pre_counts: Vec<(String, usize)> = scc_writes[scc_idx]
                    .iter()
                    .filter_map(|name| self.relations.get(name).map(|r| (name.clone(), r.len())))
                    .collect();

                self.run_stratum_incremental(&rules, &scc_writes[scc_idx]);

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

        all_rederived
    }

    /// Ensure the stratification cache is populated.
    fn ensure_stratification(&mut self, program: &Program) {
        if self.stratification.is_none() {
            self.stratification = Some(Stratification::build(program, &self.var_interner));
        }
    }

    /// Sync the engine's relation registry with a (potentially changed) program.
    ///
    /// Creates storage for new relations without disturbing existing ones.
    /// Existing relation data is preserved for incremental re-evaluation.
    /// Invalidates the cached stratification so it is rebuilt on next run.
    pub fn update_program(&mut self, program: &Program) {
        self.stratification = None;
        for (name, rel) in &program.relations {
            let arity = rel.column_types.len();
            let types: Vec<Option<String>> = rel
                .column_types
                .iter()
                .map(|ty| {
                    if let syn::Type::Path(tp) = ty {
                        tp.path.get_ident().map(|id| id.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            self.relations
                .entry(name.clone())
                .or_insert_with(|| Relation::new_auto(arity, rel.is_lattice, &types));
            self.col_types.entry(name.clone()).or_insert_with(|| types);
        }
    }

    /// Run a set of rules to fixpoint.
    fn run_stratum(&mut self, rules: &[&CRule], scc_key: usize, rule_indices: &[usize]) {
        if rules.is_empty() {
            return;
        }

        // Fast path: Stage 4 stratum (inlined rule bodies, no call_indirect)
        #[cfg(all(feature = "jit", feature = "specialized"))]
        if self.try_run_stratum_stage4(rules, scc_key, rule_indices) {
            return;
        }

        // Fast path: Stage 3 stratum (direct-insert, no results buffer)
        #[cfg(all(feature = "jit", feature = "specialized"))]
        if self.try_run_stratum_stage3(rules, scc_key, rule_indices) {
            return;
        }

        // Fallback: Stage 2 stratum meta-function (buffered JIT loop)
        #[cfg(all(feature = "jit", feature = "specialized"))]
        if self.try_run_stratum_meta(rules, scc_key, rule_indices) {
            return;
        }

        // Interpreter fallback: sync interpreter state for all packed relations.
        // JIT strata skip updating indices/value_data/source_tags; the interpreter
        // reads those structures, so they must be synced before evaluation begins.
        #[cfg(feature = "specialized")]
        {
            use crate::relation::Relation;
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

    /// Try to run the stratum via the compiled meta-function.
    ///
    /// Returns `true` if successful (all rules are packed-JIT eligible and the
    /// meta-function was compiled), `false` to fall back to the interpreted loop.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn try_run_stratum_meta(&mut self, rules: &[&CRule], scc_key: usize, rule_indices: &[usize]) -> bool {
        if rules.is_empty() {
            return false;
        }

        let stratum_key = scc_key;

        // Step 1: ensure all packed rules are compiled
        {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap();
            for (rule, &rule_idx) in rules.iter().zip(rule_indices.iter()) {
                if jit.get_or_compile_packed(rule_idx, rule).is_none() {
                    return false;
                }
            }
        }

        // Step 2: build the runtime context if not yet cached
        if !self.stratum_meta_cache.contains_key(&stratum_key) {
            let runtime = match self.build_stratum_meta_runtime(rules, rule_indices) {
                Some(r) => r,
                None => return false,
            };
            self.stratum_meta_cache.insert(stratum_key, runtime);
        }

        // Step 3: compile or retrieve the stratum meta-function
        let meta_fn = {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap();
            match jit.compile_stratum_meta(stratum_key, rules) {
                Some(f) => f,
                None => return false,
            }
        };

        // Step 4: call the meta-function
        let runtime = self.stratum_meta_cache.get_mut(&stratum_key).unwrap();
        unsafe { meta_fn(&raw mut *runtime.meta_ctx) };
        true
    }

    /// Try to run the stratum via the Stage 4 compiled function (inlined rule bodies).
    ///
    /// Returns `true` if successful, `false` to fall back to Stage 3 or interpreted.
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
            use crate::compiled::CBodyItem;
            use crate::relation::Relation;

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
                let strat = self.stratification.as_ref().unwrap();
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
            // The ASM/Cranelift JIT selects the mode by `!is_rec[level]`, which must
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

        // Step 1: compile the Stage 4 stratum function (eligibility checked inside)
        let stage4_fn = {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap();
            jit.var_count = self.var_count;
            match jit.compile_stratum_stage4(stratum_key, rules) {
                Some(f) => f,
                None => return false,
            }
        };

        // Step 1b: also compile the native function if the jit-asm feature is available.
        // The native fn reads scan data directly from JitRelData (no packed_count/pdptr callbacks).
        // This is compiled speculatively; it will only be used if stage4_native_runtime is Some.
        #[cfg(feature = "jit-asm")]
        let stage4_native_fn: Option<crate::jit::packed_helpers::StratumStage4Fn> =
            self.jit.as_ref().and_then(|jit_cell| {
                let mut jit = jit_cell.lock().unwrap();
                jit.var_count = self.var_count;
                jit.compile_stratum_stage4_native(stratum_key, rules)
            });

        // Step 2: build the runtime context if not yet cached.
        // Pass whether the native fn compiled so `build_stratum_stage4_runtime` can skip
        // the expensive `build_stratum_stage4_native_runtime` call (and jit_native init)
        // when the asm native path will never activate for this stratum.
        #[cfg(feature = "jit-asm")]
        let native_fn_available = stage4_native_fn.is_some();
        #[cfg(not(feature = "jit-asm"))]
        let native_fn_available = false;

        if !self.stratum_stage4_cache.contains_key(&stratum_key) {
            let runtime = match self.build_stratum_stage4_runtime(rules, native_fn_available) {
                Some(r) => r,
                None => return false,
            };
            self.stratum_stage4_cache.insert(stratum_key, runtime);
        }

        // Step 3: call the stratum function.
        // If the native runtime is available AND the native function compiled successfully,
        // use the native path (zero read-side Rust callbacks).
        // Otherwise fall back to the old path.
        let runtime = self.stratum_stage4_cache.get_mut(&stratum_key).unwrap();

        #[cfg(feature = "jit-asm")]
        if let (Some(native_fn), Some(native_runtime)) =
            (stage4_native_fn, runtime.stage4_native_runtime.as_mut())
        {
            // The native fn takes *mut StratumStage4NativeCtx, but StratumStage4Fn
            // is typed as *mut StratumStage4Ctx. Both are opaque pointers at the ABI
            // level; transmute is safe here since we know which ctx the fn expects.
            type NativeFn = unsafe extern "C" fn(*mut crate::jit::packed_helpers::StratumStage4NativeCtx);
            let native_fn_typed: NativeFn = unsafe { std::mem::transmute(native_fn) };
            unsafe { native_fn_typed(&raw mut *native_runtime.ctx) };
            return true;
        }

        unsafe { stage4_fn(&raw mut *runtime.stage4_ctx) };

        true
    }

    /// Build the runtime context for Stage 4 stratum execution.
    ///
    /// Returns `None` if any rule doesn't have all-packed clause or head relations.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn build_stratum_stage4_runtime(&mut self, rules: &[&CRule], native_fn_available: bool) -> Option<StratumStage4Runtime> {
        use crate::jit::packed_helpers::{JitHeadBuf, LookupSpec, PackedJitContextV3, StratumStage4Ctx};
        use crate::jit_index::JitLookupHandle;
        use crate::relation::Relation;
        use crate::specialized::PackedStorage;

        let mut per_rule_clause_rels: Vec<Box<[*const PackedStorage]>> = Vec::new();
        let mut per_rule_head_rels: Vec<Box<[*mut PackedStorage]>> = Vec::new();
        let mut per_rule_dedup_handles: Vec<Box<[*mut crate::jit_index::JitDedupHandle]>> =
            Vec::new();
        let mut per_rule_ctxs: Vec<Box<PackedJitContextV3>> = Vec::new();
        let mut rule_ctx_ptrs_vec: Vec<*mut PackedJitContextV3> = Vec::new();

        // Flat handles and specs (all rules concatenated).
        let mut handles_flat: Vec<JitLookupHandle> = Vec::new();
        let mut specs_flat: Vec<LookupSpec> = Vec::new();
        // Starting handle index for each rule (for setting lookup_handles ptr later).
        let mut rule_handle_offsets: Vec<usize> = Vec::new();

        for rule in rules {
            // Clause rel pointers
            let clause_rels: Vec<*const PackedStorage> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => match self.relations.get(&c.relation)? {
                        Relation::Packed(p) => Some(p as *const PackedStorage),
                        Relation::Generic(_) => None,
                    },
                    _ => None,
                })
                .collect();

            let clause_count = rule
                .body
                .iter()
                .filter(|i| matches!(i, CBodyItem::Clause(_)))
                .count();

            if clause_rels.len() != clause_count {
                return None;
            }

            // Head rel pointers
            let head_rels: Vec<*mut PackedStorage> = rule
                .heads
                .iter()
                .filter_map(|head| match self.relations.get(&head.relation)? {
                    Relation::Packed(p) => {
                        Some(p as *const PackedStorage as *mut PackedStorage)
                    }
                    Relation::Generic(_) => None,
                })
                .collect();

            if head_rels.len() != rule.heads.len() {
                return None;
            }

            // Build 2 handles per clause (full + recent).
            // Index: clause_offset * 2 + use_recent
            let rule_handle_start = handles_flat.len();
            rule_handle_offsets.push(rule_handle_start);

            // Extract clause body items in order to find the primary bound column.
            let rule_clauses: Vec<&crate::compiled::CClause> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => Some(c),
                    _ => None,
                })
                .collect();

            for (clause_offset, clause) in rule_clauses.iter().enumerate() {
                let rel_ptr = clause_rels[clause_offset];
                for use_recent_flag in [0u32, 1u32] {
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

            let clause_rels_box: Box<[*const PackedStorage]> = clause_rels.into_boxed_slice();
            let head_rels_box: Box<[*mut PackedStorage]> = head_rels.into_boxed_slice();
            let head_rels_ptr: *const *mut PackedStorage = head_rels_box.as_ptr();

            // Build dedup handle pointers (one per head relation).
            let dedup_handles: Box<[*mut crate::jit_index::JitDedupHandle]> = head_rels_box
                .iter()
                .map(|&ps| unsafe { &raw mut (*ps).jit_dedup.handle })
                .collect();
            let head_dedup_handles_ptr = dedup_handles.as_ptr();

            // lookup_handles pointer will be fixed up after handles_flat is boxed.
            let ctx = Box::new(PackedJitContextV3 {
                rels: clause_rels_box.as_ptr(),
                rels_len: clause_rels_box.len() as u32,
                _pad: 0,
                head_rels: head_rels_ptr,
                lookup_handles: std::ptr::null(), // fixed up below
                head_dedup_handles: head_dedup_handles_ptr,
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

        // Build tuple_sets_buf: parallel to handles_buf.
        // For fully-bound clauses (clause_offset > 0 and fresh_cols.is_empty()),
        // store a pointer to a JitTupleSet built from the total relation's packed data.
        // All other slots are null.
        let mut tuple_sets_storage: Vec<*const crate::jit::storage::JitTupleSet> =
            vec![std::ptr::null(); total_handles as usize];
        let mut tuple_set_boxes: Vec<Box<crate::jit::storage::JitTupleSet>> = Vec::new();
        for (rule_i, rule) in rules.iter().enumerate() {
            let rule_handle_start = rule_handle_offsets[rule_i];
            let rule_clauses: Vec<&crate::compiled::CClause> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => Some(c),
                    _ => None,
                })
                .collect();
            for (clause_offset, clause) in rule_clauses.iter().enumerate() {
                if clause_offset == 0 {
                    // Clause 0 is always a full scan; skip.
                    continue;
                }
                if !clause.fresh_cols.is_empty() {
                    // Not fully-bound; no tuple set needed.
                    continue;
                }
                // Fully-bound inner clause: build JitTupleSet from total relation.
                let ps = match self.relations.get(&clause.relation) {
                    Some(Relation::Packed(p)) => p,
                    _ => continue,
                };
                let arity = ps.arity;
                let ts = crate::jit::storage::JitTupleSet::build(&ps.packed_data, arity);
                tuple_set_boxes.push(ts);
                let ts_ptr = tuple_set_boxes.last().unwrap().as_ref() as *const crate::jit::storage::JitTupleSet;
                // Store for both use_recent variants (the tuple set is for total, correct
                // for existence checks regardless of which delta is being iterated).
                let base = rule_handle_start + clause_offset * 2;
                tuple_sets_storage[base] = ts_ptr;
                tuple_sets_storage[base + 1] = ts_ptr;
            }
        }
        let tuple_sets_box: Box<[*const crate::jit::storage::JitTupleSet]> =
            tuple_sets_storage.into_boxed_slice();

        // Fix up per-rule ctx lookup_handles pointers now that handles_box is stable.
        for (rule_i, ctx) in per_rule_ctxs.iter_mut().enumerate() {
            let offset = rule_handle_offsets[rule_i];
            // SAFETY: handles_box is pinned in a Box which won't move.
            ctx.lookup_handles = unsafe { handles_box.as_mut_ptr().add(offset) };
        }

        // Build head write buffers: one JitHeadBuf per (rule_i, head_i) pair.
        // Also build a parallel head_rel_ptrs array mapping each buf → *mut PackedStorage.
        const HEAD_BUF_INIT_CAP: u32 = 256;
        let mut head_buf_boxes: Vec<Box<JitHeadBuf>> = Vec::new();
        let mut head_buf_data: Vec<Vec<u32>> = Vec::new();
        let mut head_write_bufs_vec: Vec<*mut JitHeadBuf> = Vec::new();
        let mut head_rel_ptrs_vec: Vec<*mut PackedStorage> = Vec::new();
        for (rule_i, rule) in rules.iter().enumerate() {
            for (hi, head) in rule.heads.iter().enumerate() {
                let arity = head.args.len();
                let cap = HEAD_BUF_INIT_CAP;
                let mut data: Vec<u32> = vec![0u32; cap as usize * arity.max(1)];
                let data_ptr = data.as_mut_ptr();
                let buf = Box::new(JitHeadBuf {
                    data: data_ptr,
                    len: 0,
                    cap,
                    arity: arity as u32,
                    _pad: 0,
                });
                // head_rel_ptrs[rule_head_base + hi] = per_rule_head_rels[rule_i][hi]
                let rel_ptr = per_rule_head_rels[rule_i][hi];
                head_buf_boxes.push(buf);
                head_buf_data.push(data);
                head_write_bufs_vec.push(head_buf_boxes.last_mut().unwrap().as_mut() as *mut JitHeadBuf);
                head_rel_ptrs_vec.push(rel_ptr);
                let _ = (rule_i, hi); // suppress unused warnings
            }
        }
        let total_heads = head_write_bufs_vec.len() as u32;
        let head_write_bufs_box: Box<[*mut JitHeadBuf]> = head_write_bufs_vec.into_boxed_slice();
        let head_rel_ptrs_box: Box<[*mut PackedStorage]> = head_rel_ptrs_vec.into_boxed_slice();

        // All packed rels for advance()
        let all_rels: Vec<*mut PackedStorage> = self
            .relations
            .values()
            .filter_map(|rel| match rel {
                Relation::Packed(p) => Some(p as *const PackedStorage as *mut PackedStorage),
                Relation::Generic(_) => None,
            })
            .collect();

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
            tuple_sets_buf: tuple_sets_box.as_ptr(),
            head_write_bufs: head_write_bufs_box.as_ptr(),
            head_rel_ptrs: head_rel_ptrs_box.as_ptr(),
            total_heads,
            _pad4: 0,
        });

        // Build the native runtime only when the asm native fn actually compiled.
        // Skipping this when native_fn_available=false avoids initializing jit_native on
        // every PackedStorage, which would then be rebuilt on every fixpoint iteration even
        // though the native asm path is never used (e.g. fibonacci unsupported exprs).
        #[cfg(feature = "jit-asm")]
        let native_runtime = if native_fn_available {
            self.build_stratum_stage4_native_runtime(rules)
        } else {
            None
        };
        #[cfg(not(feature = "jit-asm"))]
        let native_runtime: Option<crate::jit::packed_helpers::StratumStage4NativeRuntime> = None;

        Some(StratumStage4Runtime {
            stage4_ctx,
            _per_rule_clause_rels: per_rule_clause_rels,
            _per_rule_head_rels: per_rule_head_rels,
            _per_rule_ctxs: per_rule_ctxs,
            _rule_ctx_ptrs: rule_ctx_ptrs_box,
            _all_rels: all_rels_box,
            _handles_buf: handles_box,
            _lookup_specs: specs_box,
            _per_rule_dedup_handles: per_rule_dedup_handles,
            _tuple_sets: tuple_set_boxes,
            _tuple_sets_buf: tuple_sets_box,
            _head_buf_boxes: head_buf_boxes,
            _head_buf_data: head_buf_data,
            _head_write_bufs: head_write_bufs_box,
            _head_rel_ptrs: head_rel_ptrs_box,
            stage4_native_runtime: native_runtime,
        })
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
    ) -> Option<crate::jit::packed_helpers::StratumStage4NativeRuntime> {
        use crate::jit::packed_helpers::{
            NativeHeadSpec, NativeScanSpec, StratumStage4NativeCtx, StratumStage4NativeRuntime,
        };
        use crate::jit::storage::JitRelData;
        use crate::relation::Relation;
        use crate::specialized::PackedStorage;

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
                if let Some(Relation::Packed(ps)) = self.relations.get_mut(name)
                    && ps.jit_native.is_none()
                {
                    // advance_jit() moves delta→recent and updates jit indices.
                    // It does NOT build jit_native when it's None (to avoid the cost on the
                    // Cranelift path, which never calls build_stratum_stage4_native_runtime).
                    // Explicitly initialize jit_native here so the `?` guards below succeed.
                    ps.advance_jit();
                    ps.jit_native = Some(ps.build_native_projection());
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
            let rule_clauses: Vec<&crate::compiled::CClause> = rule
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

        // All unique PackedStorage relations for advance.
        let mut advance_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mut advance_rels_vec: Vec<*mut PackedStorage> = Vec::new();
        for (_, rel) in self.relations.iter() {
            if let Relation::Packed(ps) = rel {
                let ptr = ps as *const PackedStorage as *mut PackedStorage;
                let addr = ptr as usize;
                if advance_set.insert(addr) {
                    advance_rels_vec.push(ptr);
                }
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

    /// Try to run the stratum via the Stage 3 compiled function (direct-insert, no buffer).
    ///
    /// Returns `true` if successful, `false` to fall back to Stage 2 or interpreted.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn try_run_stratum_stage3(&mut self, rules: &[&CRule], scc_key: usize, rule_indices: &[usize]) -> bool {
        if rules.is_empty() {
            return false;
        }

        let stratum_key = scc_key;

        // Step 1: ensure all V3 packed rules are compiled
        {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap();
            jit.var_count = self.var_count;
            for (rule, &rule_idx) in rules.iter().zip(rule_indices.iter()) {
                if jit.get_or_compile_packed_v3(rule_idx, rule).is_none() {
                    return false;
                }
            }
        }

        // Step 2: build the runtime context if not yet cached
        if !self.stratum_stage3_cache.contains_key(&stratum_key) {
            let runtime = match self.build_stratum_stage3_runtime(rules, rule_indices) {
                Some(r) => r,
                None => return false,
            };
            self.stratum_stage3_cache.insert(stratum_key, runtime);
        }

        // Step 3: compile or retrieve the Stage 3 stratum function
        let stage3_fn = {
            let Some(jit_cell) = self.jit.as_ref() else {
                return false;
            };
            let mut jit = jit_cell.lock().unwrap();
            match jit.compile_stratum_stage3(stratum_key, rules) {
                Some(f) => f,
                None => return false,
            }
        };

        // Step 4: call the Stage 3 stratum function
        let runtime = self.stratum_stage3_cache.get_mut(&stratum_key).unwrap();
        unsafe { stage3_fn(&raw mut *runtime.stage3_ctx) };
        true
    }

    /// Build the runtime context for Stage 3 stratum execution.
    ///
    /// Returns `None` if any rule doesn't have all-packed clause or head relations.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn build_stratum_stage3_runtime(&self, rules: &[&CRule], rule_indices: &[usize]) -> Option<StratumStage3Runtime> {
        use crate::jit::packed_helpers::{PackedJitContextV3, StratumStage3Ctx};
        use crate::relation::Relation;
        use crate::specialized::PackedStorage;

        let jit_ref = self.jit.as_ref()?.lock().unwrap();

        let mut per_rule_clause_rels: Vec<Box<[*const PackedStorage]>> = Vec::new();
        let mut per_rule_head_rels: Vec<Box<[*mut PackedStorage]>> = Vec::new();
        let mut per_rule_dedup_handles: Vec<Box<[*mut crate::jit_index::JitDedupHandle]>> =
            Vec::new();
        let mut per_rule_ctxs: Vec<Box<PackedJitContextV3>> = Vec::new();

        let mut full_fns_vec: Vec<crate::jit::packed_helpers::PackedJitFnV3> = Vec::new();
        let mut full_ctx_ptrs_vec: Vec<*mut PackedJitContextV3> = Vec::new();
        let mut recent_fns_vec: Vec<crate::jit::packed_helpers::PackedJitFnV3> = Vec::new();
        let mut recent_ctx_ptrs_vec: Vec<*mut PackedJitContextV3> = Vec::new();

        for (rule, &rule_idx) in rules.iter().zip(rule_indices.iter()) {
            let compiled = jit_ref.packed_cache_v3.get(&rule_idx)?.as_ref()?;

            let full_fn = compiled.full_variant()?;

            // Clause rel pointers
            let clause_rels: Vec<*const PackedStorage> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => match self.relations.get(&c.relation)? {
                        Relation::Packed(p) => Some(p as *const PackedStorage),
                        Relation::Generic(_) => None,
                    },
                    _ => None,
                })
                .collect();

            let clause_count = rule
                .body
                .iter()
                .filter(|i| matches!(i, CBodyItem::Clause(_)))
                .count();

            if clause_rels.len() != clause_count {
                return None;
            }

            // Head rel pointers
            let head_rels: Vec<*mut PackedStorage> = rule
                .heads
                .iter()
                .filter_map(|head| match self.relations.get(&head.relation)? {
                    Relation::Packed(p) => {
                        Some(p as *const PackedStorage as *mut PackedStorage)
                    }
                    Relation::Generic(_) => None,
                })
                .collect();

            if head_rels.len() != rule.heads.len() {
                return None;
            }

            let clause_rels_box: Box<[*const PackedStorage]> = clause_rels.into_boxed_slice();
            let head_rels_box: Box<[*mut PackedStorage]> = head_rels.into_boxed_slice();
            let head_rels_ptr: *const *mut PackedStorage = head_rels_box.as_ptr();

            // Build dedup handle pointers (one per head relation).
            let dedup_handles: Box<[*mut crate::jit_index::JitDedupHandle]> = head_rels_box
                .iter()
                .map(|&ps| unsafe { &raw mut (*ps).jit_dedup.handle })
                .collect();
            let head_dedup_handles_ptr = dedup_handles.as_ptr();

            let ctx = Box::new(PackedJitContextV3 {
                rels: clause_rels_box.as_ptr(),
                rels_len: clause_rels_box.len() as u32,
                _pad: 0,
                head_rels: head_rels_ptr,
                lookup_handles: std::ptr::null(),
                head_dedup_handles: head_dedup_handles_ptr,
            });

            // Build recent variants for this rule
            let mut clause_seq = 0usize;
            for item in rule.body.iter() {
                if !matches!(item, CBodyItem::Clause(_)) {
                    continue;
                }
                if let Some(recent_fn) = compiled.recent_variant(clause_seq) {
                    recent_fns_vec.push(recent_fn);
                    recent_ctx_ptrs_vec
                        .push(&*ctx as *const PackedJitContextV3 as *mut PackedJitContextV3);
                }
                clause_seq += 1;
            }

            full_fns_vec.push(full_fn);
            full_ctx_ptrs_vec
                .push(&*ctx as *const PackedJitContextV3 as *mut PackedJitContextV3);

            per_rule_clause_rels.push(clause_rels_box);
            per_rule_head_rels.push(head_rels_box);
            per_rule_dedup_handles.push(dedup_handles);
            per_rule_ctxs.push(ctx);
        }

        // All packed rels for advance()
        let all_rels: Vec<*mut PackedStorage> = self
            .relations
            .values()
            .filter_map(|rel| match rel {
                Relation::Packed(p) => Some(p as *const PackedStorage as *mut PackedStorage),
                Relation::Generic(_) => None,
            })
            .collect();

        let all_rels_box: Box<[*mut PackedStorage]> = all_rels.into_boxed_slice();
        let full_fns_box = full_fns_vec.into_boxed_slice();
        let full_ctx_ptrs_box = full_ctx_ptrs_vec.into_boxed_slice();
        let recent_fns_box = recent_fns_vec.into_boxed_slice();
        let recent_ctx_ptrs_box = recent_ctx_ptrs_vec.into_boxed_slice();

        let stage3_ctx = Box::new(StratumStage3Ctx {
            full_fns: full_fns_box.as_ptr(),
            full_ctxs: full_ctx_ptrs_box.as_ptr(),
            num_full: full_fns_box.len() as u32,
            num_recent: recent_fns_box.len() as u32,
            recent_fns: recent_fns_box.as_ptr(),
            recent_ctxs: recent_ctx_ptrs_box.as_ptr(),
            all_rels: all_rels_box.as_ptr(),
            n_all_rels: all_rels_box.len() as u32,
            _pad: 0,
        });

        Some(StratumStage3Runtime {
            stage3_ctx,
            _per_rule_clause_rels: per_rule_clause_rels,
            _per_rule_head_rels: per_rule_head_rels,
            _per_rule_ctxs: per_rule_ctxs,
            _full_fns: full_fns_box,
            _full_ctx_ptrs: full_ctx_ptrs_box,
            _recent_fns: recent_fns_box,
            _recent_ctx_ptrs: recent_ctx_ptrs_box,
            _all_rels: all_rels_box,
            _per_rule_dedup_handles: per_rule_dedup_handles,
        })
    }

    /// Build the runtime context for the stratum meta-function.
    ///
    /// Returns `None` if any rule doesn't have all-packed clause or head relations.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn build_stratum_meta_runtime(&self, rules: &[&CRule], rule_indices: &[usize]) -> Option<StratumMetaRuntime> {
        use crate::jit::packed_helpers::{
            PackedJitContext, RuleFlushInfo, StratumFlusher, StratumMetaCtx,
        };
        use crate::relation::Relation;
        use crate::specialized::PackedStorage;

        let jit_ref = self.jit.as_ref()?.lock().unwrap();

        // Box<RuleResultsBuf>: stable heap address so the raw *mut pointer remains valid.
        #[allow(clippy::vec_box)]
        let mut per_rule_results: Vec<Box<RuleResultsBuf>> = Vec::new();
        let mut per_rule_bindings: Vec<Box<[u32]>> = Vec::new();
        let mut per_rule_clause_rels: Vec<Box<[*const PackedStorage]>> = Vec::new();
        let mut per_rule_ctxs: Vec<Box<PackedJitContext>> = Vec::new();

        let mut flush_rules: Vec<RuleFlushInfo> = Vec::new();

        // full variant: one per rule
        let mut full_fns_vec: Vec<crate::jit::packed_helpers::PackedJitFn> = Vec::new();
        let mut full_ctx_ptrs_vec: Vec<*mut PackedJitContext> = Vec::new();

        // recent variants: (rule_idx, clause_seq) pairs — flattened
        let mut recent_fns_vec: Vec<crate::jit::packed_helpers::PackedJitFn> = Vec::new();
        let mut recent_ctx_ptrs_vec: Vec<*mut PackedJitContext> = Vec::new();

        for (rule, &rule_idx) in rules.iter().zip(rule_indices.iter()) {
            let compiled = jit_ref.packed_cache.get(&rule_idx)?.as_ref()?;

            // Require full variant
            let full_fn = compiled.full_variant()?;

            // Collect clause storage pointers
            let clause_rels: Vec<*const PackedStorage> = rule
                .body
                .iter()
                .filter_map(|item| match item {
                    CBodyItem::Clause(c) => match self.relations.get(&c.relation)? {
                        Relation::Packed(p) => Some(p as *const PackedStorage),
                        Relation::Generic(_) => None,
                    },
                    _ => None,
                })
                .collect();

            let clause_count = rule
                .body
                .iter()
                .filter(|i| matches!(i, CBodyItem::Clause(_)))
                .count();

            // Bail if any clause relation is not packed
            if clause_rels.len() != clause_count {
                return None;
            }

            // Collect head storage pointers
            let head_rels: Vec<*mut PackedStorage> = rule
                .heads
                .iter()
                .filter_map(|head| match self.relations.get(&head.relation)? {
                    Relation::Packed(p) => Some(p as *const PackedStorage as *mut PackedStorage),
                    Relation::Generic(_) => None,
                })
                .collect();

            if head_rels.len() != rule.heads.len() {
                return None;
            }

            let clause_rels_box: Box<[*const PackedStorage]> = clause_rels.into_boxed_slice();
            let bindings_box: Box<[u32]> = vec![0u32; self.var_count].into_boxed_slice();
            #[allow(clippy::vec_box)]
            let results_box: Box<RuleResultsBuf> = Box::default();

            let bindings_ptr: *mut u32 = bindings_box.as_ptr() as *mut u32;
            let results_ptr: *mut Vec<(usize, Vec<u32>)> =
                std::ptr::addr_of!(*results_box) as *mut _;

            let ctx = Box::new(PackedJitContext {
                rels: clause_rels_box.as_ptr(),
                rels_len: clause_rels_box.len() as u32,
                _pad: 0,
                bindings: bindings_ptr,
                results: results_ptr,
            });

            // Build recent variants for this rule: one per clause
            let mut clause_seq = 0usize;
            for item in rule.body.iter() {
                if !matches!(item, CBodyItem::Clause(_)) {
                    continue;
                }
                if let Some(recent_fn) = compiled.recent_variant(clause_seq) {
                    recent_fns_vec.push(recent_fn);
                    // ctx ptr will be filled after we push to per_rule_ctxs; use placeholder
                    // We need the ptr to ctx before we move it into the vec — get it first
                    recent_ctx_ptrs_vec.push(&*ctx as *const PackedJitContext as *mut PackedJitContext);
                }
                clause_seq += 1;
            }

            full_fns_vec.push(full_fn);
            full_ctx_ptrs_vec.push(&*ctx as *const PackedJitContext as *mut PackedJitContext);

            flush_rules.push(RuleFlushInfo {
                results: results_ptr,
                head_rels,
            });

            per_rule_bindings.push(bindings_box);
            per_rule_results.push(results_box);
            per_rule_clause_rels.push(clause_rels_box);
            per_rule_ctxs.push(ctx);
        }

        // Collect all packed relations for advance()
        let all_packed_rels: Vec<*mut PackedStorage> = self
            .relations
            .values()
            .filter_map(|rel| match rel {
                Relation::Packed(p) => Some(p as *const PackedStorage as *mut PackedStorage),
                Relation::Generic(_) => None,
            })
            .collect();

        // Box all arrays for stable addresses
        let full_fns_box: Box<[crate::jit::packed_helpers::PackedJitFn]> =
            full_fns_vec.into_boxed_slice();
        let full_ctx_ptrs_box: Box<[*mut PackedJitContext]> =
            full_ctx_ptrs_vec.into_boxed_slice();
        let recent_fns_box: Box<[crate::jit::packed_helpers::PackedJitFn]> =
            recent_fns_vec.into_boxed_slice();
        let recent_ctx_ptrs_box: Box<[*mut PackedJitContext]> =
            recent_ctx_ptrs_vec.into_boxed_slice();

        let flusher = Box::new(StratumFlusher {
            rules: flush_rules,
            all_packed_rels,
        });

        let meta_ctx = Box::new(StratumMetaCtx {
            full_fns: full_fns_box.as_ptr(),
            full_ctxs: full_ctx_ptrs_box.as_ptr(),
            num_full: full_fns_box.len() as u32,
            num_recent: recent_fns_box.len() as u32,
            recent_fns: recent_fns_box.as_ptr(),
            recent_ctxs: recent_ctx_ptrs_box.as_ptr(),
            flusher: &*flusher as *const StratumFlusher as *mut StratumFlusher,
        });

        Some(StratumMetaRuntime {
            meta_ctx,
            _per_rule_bindings: per_rule_bindings,
            _per_rule_results: per_rule_results,
            _per_rule_clause_rels: per_rule_clause_rels,
            _per_rule_ctxs: per_rule_ctxs,
            _full_fns: full_fns_box,
            _full_ctx_ptrs: full_ctx_ptrs_box,
            _recent_fns: recent_fns_box,
            _recent_ctx_ptrs: recent_ctx_ptrs_box,
            _flusher: flusher,
        })
    }

    /// Run a set of rules incrementally (delta-only, no initial full evaluation).
    ///
    /// Assumes new facts are already in `delta` on input relations.
    /// `owned` is the set of relation names this SCC writes to.
    /// Owned relations are advanced normally; input relations use peek-advance
    /// so their deltas survive for downstream SCCs.
    fn run_stratum_incremental(&mut self, rules: &[&CRule], owned: &FxHashSet<String>) {
        if rules.is_empty() {
            return;
        }

        // Sync interpreter state before incremental evaluation.
        #[cfg(feature = "specialized")]
        {
            use crate::relation::Relation;
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
            return;
        }

        // Semi-naive loop: evaluate with use_recent=true, advance, repeat
        loop {
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
    }

    /// Evaluate a single rule.
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
            self.insert_with_source(relation, tuple, source);
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
        // Try JIT path first
        #[cfg(feature = "jit")]
        if let Some(ref jit_cell) = self.jit {
            let rule_idx = rule as *const CRule as usize;

            // Try typed packed JIT when all clause relations are PackedStorage
            #[cfg(feature = "specialized")]
            {
                let packed_compiled = {
                    let mut jit = jit_cell.lock().unwrap();
                    jit.get_or_compile_packed(rule_idx, rule).is_some()
                };
                if packed_compiled
                    && let Some(results) =
                        self.derive_tuples_packed_jit(rule, use_recent, rule_idx)
                {
                    return results;
                }
            }

            // Fall back to trampoline JIT
            let compiled = {
                let mut jit = jit_cell.lock().unwrap();
                jit.get_or_compile(rule_idx, rule).is_some()
            };
            if compiled && let Some(results) = self.derive_tuples_jit(rule, use_recent, rule_idx) {
                return results;
            }
        }

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

    /// JIT-compiled derive_tuples path.
    #[cfg(feature = "jit")]
    fn derive_tuples_jit<'a>(
        &self,
        rule: &'a CRule,
        use_recent: bool,
        rule_idx: usize,
    ) -> Option<Vec<(&'a str, Tuple)>> {
        use crate::jit::helpers::JitContext;

        let jit = self.jit.as_ref()?.lock().unwrap();
        let compiled = jit.cache.get(&rule_idx)?.as_ref()?;

        // Resolve relation pointers for clauses in order
        let clause_rels: Vec<*const Relation> = rule
            .body
            .iter()
            .filter_map(|item| match item {
                CBodyItem::Clause(c) => self
                    .relations
                    .get(&c.relation)
                    .map(|r| r as *const Relation),
                _ => None,
            })
            .collect();

        // Resolve head clause pointers
        let head_ptrs: Vec<*const CHeadClause> =
            rule.heads.iter().map(|h| h as *const CHeadClause).collect();

        let mut bindings = Bindings::new(self.var_count);
        let mut indexed_results: Vec<(usize, Tuple)> = Vec::new();

        let mut ctx = JitContext {
            rels: clause_rels.as_ptr(),
            rels_len: clause_rels.len() as u32,
            bindings: &mut bindings,
            results: &mut indexed_results,
            heads: head_ptrs.as_ptr(),
            heads_len: head_ptrs.len() as u32,
            registry: &self.type_registry,
            interner: &self.var_interner,
        };

        if !use_recent {
            if let Some(fn_ptr) = compiled.full_variant() {
                unsafe { fn_ptr(&mut ctx) };
            }
        } else {
            // Semi-naive: try each clause with recent data
            let mut clause_seq = 0;
            for item in rule.body.iter() {
                let rel_name = match item {
                    CBodyItem::Clause(c) => &c.relation,
                    _ => continue,
                };
                if let Some(rel) = self.relations.get(rel_name)
                    && rel.iter_recent().next().is_some()
                    && let Some(fn_ptr) = compiled.recent_variant(clause_seq)
                {
                    unsafe { fn_ptr(&mut ctx) };
                }
                clause_seq += 1;
            }
        }

        // Convert indexed results to named results
        let results: Vec<(&'a str, Tuple)> = indexed_results
            .into_iter()
            .map(|(head_idx, tuple)| (rule.heads[head_idx].relation.as_str(), tuple))
            .collect();

        Some(results)
    }

    /// Typed packed JIT path: reads u32 directly from PackedStorage.
    ///
    /// Returns None if any clause relation is not packed (fall through to trampoline JIT).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    fn derive_tuples_packed_jit<'a>(
        &self,
        rule: &'a CRule,
        use_recent: bool,
        rule_idx: usize,
    ) -> Option<Vec<(&'a str, Tuple)>> {
        use crate::jit::packed_helpers::PackedJitContext;
        use crate::relation::Relation;
        use crate::specialized::PackedStorage;

        // Collect PackedStorage pointers for each clause relation
        let clause_packed: Vec<*const PackedStorage> = rule
            .body
            .iter()
            .filter_map(|item| match item {
                CBodyItem::Clause(c) => match self.relations.get(&c.relation)? {
                    Relation::Packed(p) => Some(p as *const PackedStorage),
                    Relation::Generic(_) => None,
                },
                _ => None,
            })
            .collect();

        let clause_count = rule
            .body
            .iter()
            .filter(|i| matches!(i, CBodyItem::Clause(_)))
            .count();

        // Bail if any relation is not packed
        if clause_packed.len() != clause_count {
            return None;
        }

        let jit = self.jit.as_ref()?.lock().unwrap();
        let compiled = jit.packed_cache.get(&rule_idx)?.as_ref()?;

        // Flat u32 binding scratch (one slot per var_id)
        let mut bindings: Vec<u32> = vec![0u32; self.var_count];
        let mut packed_results: Vec<(usize, Vec<u32>)> = Vec::new();

        let mut ctx = PackedJitContext {
            rels: clause_packed.as_ptr(),
            rels_len: clause_packed.len() as u32,
            _pad: 0,
            bindings: bindings.as_mut_ptr(),
            results: &mut packed_results,
        };

        if !use_recent {
            if let Some(fn_ptr) = compiled.full_variant() {
                unsafe { fn_ptr(&mut ctx) };
            }
        } else {
            let mut clause_seq = 0;
            for item in rule.body.iter() {
                let rel_name = match item {
                    CBodyItem::Clause(c) => &c.relation,
                    _ => continue,
                };
                if let Some(rel) = self.relations.get(rel_name)
                    && rel.iter_recent().next().is_some()
                    && let Some(fn_ptr) = compiled.recent_variant(clause_seq)
                {
                    unsafe { fn_ptr(&mut ctx) };
                }
                clause_seq += 1;
            }
        }

        // Convert packed u32 results back to (relation_name, Tuple)
        let results: Vec<(&'a str, Tuple)> = packed_results
            .into_iter()
            .filter_map(|(head_idx, packed_tuple)| {
                let head = &rule.heads[head_idx];
                let rel_name = head.relation.as_str();
                // Get the head relation's col_types for unpacking
                let col_types = match self.relations.get(rel_name)? {
                    Relation::Packed(p) => &p.col_types,
                    Relation::Generic(_) => return None,
                };
                let tuple: Tuple = packed_tuple
                    .iter()
                    .zip(col_types.iter())
                    .map(|(&raw, ty)| ty.unpack(raw))
                    .collect();
                Some((rel_name, tuple))
            })
            .collect();

        Some(results)
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
                .unwrap();
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

        let mut collected: Vec<Vec<Value>> = Vec::new();
        let bound_cols = self.find_bound_agg_columns(agg, binding);

        if !bound_cols.is_empty() {
            // Use index lookup: pick most selective column
            let (primary_pos, _) = bound_cols
                .iter()
                .enumerate()
                .min_by_key(|&(_, (col, val))| rel.lookup(*col, val).len())
                .unwrap();
            let (primary_col, primary_val) = &bound_cols[primary_pos];
            let indices = rel.lookup(*primary_col, primary_val);

            for &idx in indices {
                let tuple = rel.get(idx);
                // Pre-filter secondary bound columns
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
            // No bound columns: full scan
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

        // Apply aggregator and recurse for each result
        let agg_results =
            apply_aggregator(&agg.aggregator_name, collected.iter().map(|v| v.as_slice()));

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
        let declared = self.col_types.get(head.relation.as_str());

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
    fn test_unsuffixed_literal_coercion() {
        // Unsuffixed literals default to i32 in the evaluator, but should be
        // coerced to the declared column type (e.g. u32) at head insertion time.
        let engine = run_program(
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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
        engine.run(&program);

        let rederived =
            engine.run_incremental(&program, &FxHashSet::default(), &FxHashSet::default());
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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);

        let src = engine.intern_source("initial");
        engine.insert_with_source("edge", vec![Value::I32(1), Value::I32(2)], src);
        engine.insert_with_source("edge", vec![Value::I32(2), Value::I32(3)], src);
        engine.run(&program);
        assert_eq!(engine.relation("path").unwrap().len(), 3); // (1,2), (2,3), (1,3)

        // Retract and add different edges
        engine.retract_source(src);
        let src2 = engine.intern_source("updated");
        engine.insert_with_source("edge", vec![Value::I32(10), Value::I32(20)], src2);

        let dirty = FxHashSet::from_iter(["edge".to_string()]);
        let retracted = FxHashSet::from_iter(["edge".to_string()]);
        let rederived = engine.run_incremental(&program, &dirty, &retracted);

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("a", vec![Value::I32(1)]);
        engine.run(&program);

        assert_eq!(engine.relation("b").unwrap().len(), 1);
        assert_eq!(engine.relation("c").unwrap().len(), 1);
        assert_eq!(engine.relation("unrelated").unwrap().len(), 1);

        // Change a's contents
        engine.relation_mut("a").unwrap().clear();
        engine.insert("a", vec![Value::I32(2)]);
        engine.insert("a", vec![Value::I32(3)]);

        let dirty = FxHashSet::from_iter(["a".to_string()]);
        let retracted = FxHashSet::from_iter(["a".to_string()]);
        let rederived = engine.run_incremental(&program, &dirty, &retracted);

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("src", vec![Value::I32(1)]);
        engine.run(&program);

        let dirty = FxHashSet::from_iter(["src".to_string()]);
        let rederived = engine.run_incremental(&program, &dirty, &FxHashSet::default());
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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
        engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
        engine.run(&program);
        assert_eq!(engine.relation("path").unwrap().len(), 3); // (1,2), (2,3), (1,3)

        // Add a new edge — old paths should be preserved, new ones derived
        engine.insert("edge", vec![Value::I32(3), Value::I32(4)]);
        let dirty = FxHashSet::from_iter(["edge".to_string()]);
        engine.run_incremental(&program, &dirty, &FxHashSet::default());

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("node", vec![Value::I32(1)]);
        engine.insert("node", vec![Value::I32(2)]);
        engine.insert("node", vec![Value::I32(3)]);
        engine.run(&program);
        assert_eq!(engine.relation("included").unwrap().len(), 3);

        // Add an exclusion — negation SCC must re-derive from scratch
        engine.insert("excluded", vec![Value::I32(2)]);
        let dirty = FxHashSet::from_iter(["excluded".to_string()]);
        engine.run_incremental(&program, &dirty, &FxHashSet::default());

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("raw", vec![Value::I32(1)]);
        engine.insert("raw", vec![Value::I32(2)]);
        engine.insert("excluded", vec![Value::I32(2)]);
        engine.run(&program);

        let filtered = engine.relation("filtered").unwrap();
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains(&[Value::I32(1)]));

        // Add more raw data — derived uses delta, filtered re-derives
        engine.insert("raw", vec![Value::I32(3)]);
        let dirty = FxHashSet::from_iter(["raw".to_string()]);
        engine.run_incremental(&program, &dirty, &FxHashSet::default());

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("node", vec![Value::I32(1)]);
        engine.insert("node", vec![Value::I32(2)]);
        engine.insert("node", vec![Value::I32(3)]);
        engine.run(&program);
        assert_eq!(engine.relation("doubled").unwrap().len(), 3);

        // Exclude node 2 — kept clears (non-monotone), doubled must also re-derive
        engine.insert("excluded", vec![Value::I32(2)]);
        let dirty = FxHashSet::from_iter(["excluded".to_string()]);
        engine.run_incremental(&program, &dirty, &FxHashSet::default());

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
        let program = Program::from_ast(ast);
        let mut engine = Engine::new(&program);
        engine.insert("src", vec![Value::I32(1)]);
        engine.run(&program);
        assert_eq!(engine.relation("dst").unwrap().len(), 1);

        // Insert same fact again (deduplication means no actual delta)
        engine.insert("src", vec![Value::I32(1)]);
        let dirty = FxHashSet::from_iter(["src".to_string()]);
        engine.run_incremental(&program, &dirty, &FxHashSet::default());

        // Should still have exactly 1 tuple
        assert_eq!(engine.relation("dst").unwrap().len(), 1);
    }
}

#[cfg(test)]
#[cfg(all(feature = "jit", feature = "specialized"))]
mod jit_hot_tests {
    use super::*;
    use ascent_syntax::AscentProgram;
    use ascent_ir::Program;

    fn run_shared_jit(source: &str) {
        let ast: AscentProgram = syn::parse_str(source).unwrap();
        let program = Program::from_ast(ast);
        // warmup
        let mut warmup = Engine::new(&program);
        warmup.enable_jit();
        warmup.run(&program);
        let jit = warmup.share_jit_compiler().unwrap();
        // hot run
        let mut engine = Engine::new(&program);
        engine.with_jit_compiler(jit);
        engine.run(&program);
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
        let program = Program::from_ast(ast);
        let n = 10i32;
        // warmup
        let mut warmup = Engine::new(&program);
        warmup.enable_jit();
        for i in 1..n {
            warmup.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
        }
        warmup.run(&program);
        let jit = warmup.share_jit_compiler().unwrap();
        // hot runs: verify correctness across several iterations
        for _ in 0..10 {
            let mut engine = Engine::new(&program);
            engine.with_jit_compiler(jit.clone());
            for i in 1..n {
                engine.insert("edge", vec![Value::I32(i), Value::I32(i + 1)]);
            }
            engine.run(&program);
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
