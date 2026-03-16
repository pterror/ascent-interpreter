//! Arity-specialized relation storage for u32-representable values.
//!
//! When all columns of a relation can be represented as `u32` (i32, u32,
//! interned strings, booleans), this module provides an accelerated storage
//! backend. Deduplication and index operations use raw `u32` values for
//! faster hashing and comparison, while a parallel `Vec<Value>` buffer
//! maintains compatibility with the generic evaluation loop.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

use hashbrown::HashTable;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::intern;
use crate::relation::SourceId;
use crate::value::{InternTable, Tuple, Value};

/// General-purpose intern table for arbitrary `Value`s.
///
/// Assigns sequential u32 IDs to distinct values on first pack. Use this for
/// any column type that is `Hash + Eq` but lacks a built-in compact id (e.g.
/// custom types). The `filter` predicate gates which `Value` variants are
/// accepted; non-matching variants return `None` so packed storage can
/// downgrade gracefully.
#[allow(dead_code)]
pub struct HashInternTable {
    filter: fn(&Value) -> bool,
    to_id: RefCell<FxHashMap<Value, u32>>,
    to_val: RefCell<Vec<Value>>,
}

#[allow(dead_code)]
impl HashInternTable {
    /// Create a new intern table that accepts values matching `filter`.
    pub fn new(filter: fn(&Value) -> bool) -> Self {
        Self {
            filter,
            to_id: RefCell::new(FxHashMap::default()),
            to_val: RefCell::new(Vec::new()),
        }
    }
}

impl InternTable for HashInternTable {
    fn pack(&self, val: &Value) -> Option<u32> {
        if !(self.filter)(val) {
            return None;
        }
        if let Some(&id) = self.to_id.borrow().get(val) {
            return Some(id);
        }
        let id = self.to_val.borrow().len() as u32;
        self.to_val.borrow_mut().push(val.clone());
        self.to_id.borrow_mut().insert(val.clone(), id);
        Some(id)
    }

    fn fmt_display(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.to_val.borrow()[id as usize])
    }

    fn fmt_debug(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.to_val.borrow()[id as usize])
    }

    fn cmp_ids(&self, a: u32, b: u32) -> Ordering {
        let vals = self.to_val.borrow();
        vals[a as usize]
            .partial_cmp_val(&vals[b as usize])
            .unwrap_or(Ordering::Equal)
    }
}

/// Column type tag for packing/unpacking Value <-> u32.
#[derive(Clone)]
pub enum PackedType {
    I32,
    U32,
    Bool,
    /// Any type backed by an intern table.
    /// Strings use `intern::StringTable`; other types use `HashInternTable`.
    Interned(Rc<dyn InternTable>),
}

impl std::fmt::Debug for PackedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackedType::I32 => write!(f, "I32"),
            PackedType::U32 => write!(f, "U32"),
            PackedType::Bool => write!(f, "Bool"),
            PackedType::Interned(_) => write!(f, "Interned(..)"),
        }
    }
}

impl PartialEq for PackedType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PackedType::I32, PackedType::I32)
                | (PackedType::U32, PackedType::U32)
                | (PackedType::Bool, PackedType::Bool)
                | (PackedType::Interned(_), PackedType::Interned(_))
        )
    }
}

impl Eq for PackedType {}

impl PackedType {
    /// Try to classify a declared column type name as packable to u32.
    pub fn from_type_name(name: &str) -> Option<Self> {
        match name {
            "i32" => Some(PackedType::I32),
            "u32" => Some(PackedType::U32),
            "String" | "&str" | "str" => Some(PackedType::Interned(intern::string_table())),
            "bool" => Some(PackedType::Bool),
            _ => None,
        }
    }

    /// Pack a Value to its u32 representation.
    #[inline]
    pub fn pack(&self, val: &Value) -> Option<u32> {
        match (self, val) {
            (PackedType::I32, Value::I32(n)) => Some(*n as u32),
            (PackedType::U32, Value::U32(n)) => Some(*n),
            (PackedType::Bool, Value::Bool(b)) => Some(*b as u32),
            (PackedType::Interned(table), _) => table.pack(val),
            _ => None,
        }
    }

    /// Unpack u32 back to Value.
    #[inline]
    pub fn unpack(&self, raw: u32) -> Value {
        match self {
            PackedType::I32 => Value::I32(raw as i32),
            PackedType::U32 => Value::U32(raw),
            PackedType::Bool => Value::Bool(raw != 0),
            PackedType::Interned(table) => Value::Interned(table.clone(), raw),
        }
    }
}


/// Try to classify all columns as packable. Returns None if any column
/// is unknown or not u32-representable.
pub fn try_packed_col_types(col_types: &[Option<String>]) -> Option<Vec<PackedType>> {
    if col_types.is_empty() {
        return None;
    }
    col_types
        .iter()
        .map(|t| t.as_deref().and_then(PackedType::from_type_name))
        .collect()
}

/// Accelerated relation storage using packed u32 representation.
///
/// Maintains a dual-buffer layout:
/// - `packed_data: Vec<u32>` — flat u32 buffer for fast dedup/index operations
/// - `value_data: Vec<Value>` — flat Value buffer for eval loop compatibility
///
/// Dedup hashing and per-column indices use the u32 representation,
/// avoiding Value enum dispatch overhead. The Value buffer is maintained
/// in parallel so the evaluation loop can read tuples without conversion.
#[derive(Debug)]
pub struct PackedStorage {
    /// Flat Value buffer for eval loop reads. Tuple `i` at `[i*arity..(i+1)*arity]`.
    pub(crate) value_data: Vec<Value>,
    /// Flat packed u32 buffer. Tuple `i` at `[i*arity..(i+1)*arity]`.
    pub(crate) packed_data: Vec<u32>,
    /// Per-column type for Value <-> u32 conversion.
    pub(crate) col_types: Vec<PackedType>,
    /// Number of tuples.
    pub(crate) count: usize,
    /// Number of columns.
    pub(crate) arity: usize,
    /// Tuple indices added this iteration.
    pub(crate) delta: Vec<usize>,
    /// Tuple indices from previous iteration.
    pub(crate) recent: Vec<usize>,
    /// Per-column u32-keyed index.
    indices: Vec<FxHashMap<u32, Vec<usize>>>,
    /// Per-column u32-keyed index for recent tuples.
    recent_col_indices: Vec<FxHashMap<u32, Vec<usize>>>,
    /// Source tag per tuple.
    pub(crate) source_tags: Vec<SourceId>,
    /// Number of tuples whose interpreter state (indices, value_data, source_tags) is synced.
    ///
    /// Tuples inserted via the JIT fast path skip updating these interpreter-only structures
    /// to reduce per-tuple overhead.  `ensure_interp_synced()` catches them up on demand —
    /// called before interpreter stratum evaluation and before result access.
    pub(crate) interp_synced_count: usize,
    /// Per-column JIT hash index (full data). Updated incrementally.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_indices: Vec<crate::jit_index::JitHashIndex>,
    /// Per-column JIT hash index (recent data). Rebuilt from scratch each iteration.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_recent_indices: Vec<crate::jit_index::JitHashIndex>,
    /// Number of tuples already indexed into jit_indices (for incremental update).
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_full_indexed_count: usize,
    /// True if this relation is EDB in the current stratum (never written by any rule head).
    ///
    /// When true, `update_jit_indices` builds the full index contiguously ONCE and then
    /// skips all subsequent rebuilds (the relation is stable). The recent index is also
    /// built contiguously each iteration. Set by `eval.rs` before running a stratum.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_is_edb: bool,
    /// True if this relation is a sink in the current stratum (appears only in rule heads,
    /// never in any body clause). When true, `update_jit_indices` skips all index building
    /// since the indices are never probed. Set by `eval.rs` before running a stratum.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_is_sink: bool,
    /// True if `jit_indices` was last built with `jit_is_edb = true` (col-value format).
    /// False if built with `jit_is_edb = false` (tuple-index format) or not yet built.
    /// Used by `eval.rs` to detect stale non-EDB indices that must be rebuilt when
    /// `jit_is_edb` transitions from false to true.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_index_is_edb_fmt: bool,
    /// Native JIT storage projection: total, recent, and empty-new views.
    /// Rebuilt on every `advance_jit()` call.  `None` before the first advance.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) jit_native: Option<crate::jit::storage::JitNativeRelData>,
    /// Authoritative dedup table (inline u32 hash table, also probed by JIT code directly).
    pub(crate) jit_dedup: crate::jit_index::JitDedupTable,
}

impl PackedStorage {
    /// Create a new packed storage with the given column types.
    pub fn new(col_types: Vec<PackedType>) -> Self {
        let arity = col_types.len();
        Self {
            value_data: Vec::new(),
            packed_data: Vec::new(),
            col_types,
            count: 0,
            arity,
            delta: Vec::new(),
            recent: Vec::new(),
            indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            recent_col_indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            source_tags: Vec::new(),
            interp_synced_count: 0,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_indices: Vec::new(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_recent_indices: Vec::new(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_full_indexed_count: 0,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_is_edb: false,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_is_sink: false,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_index_is_edb_fmt: false,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_native: None,
            jit_dedup: crate::jit_index::JitDedupTable::new(arity),
        }
    }

    /// Build a `JitNativeRelData` projection from the current state of this storage.
    ///
    /// - `total`:  covers all tuples in `packed_data[0..count*arity]`, column indices built
    ///   (skipped for sink relations since they are never probed as body clauses).
    /// - `recent`: covers only the recent tuples (gathered from `self.recent` index list).
    /// - `new`:    empty write buffer with initial capacity 64 tuples.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) fn build_native_projection(
        &self,
    ) -> crate::jit::storage::JitNativeRelData {
        use crate::jit::storage::{JitNativeRelData, JitRelData};

        let arity = self.arity;
        // Sink relations never appear in body clauses, so JitColIndex is never probed.
        // Skip expensive index building (sort + group) for them.
        let build_indices = !self.jit_is_sink;
        let total_slice = &self.packed_data[0..self.count * arity.max(1)];

        // Gather recent tuples into a contiguous slice.
        let mut recent_buf: Vec<u32> = Vec::with_capacity(self.recent.len() * arity.max(1));
        for &idx in &self.recent {
            let start = idx * arity;
            recent_buf.extend_from_slice(&self.packed_data[start..start + arity]);
        }

        JitNativeRelData {
            total: JitRelData::build_from_packed(total_slice, arity, build_indices),
            recent: JitRelData::build_from_packed(&recent_buf, arity, build_indices),
            // `new` must have an allocated tuple_set so the JIT can probe it inline.
            // `build_from_packed` on an empty slice allocates a minimum-capacity (16 slots)
            // tuple_set and a data buffer, with len=0 and cap=1.
            new: JitRelData::build_from_packed(&[], arity, false),
        }
    }

    #[inline]
    fn value_slice(&self, idx: usize) -> &[Value] {
        &self.value_data[idx * self.arity..(idx + 1) * self.arity]
    }

    #[allow(dead_code)]
    #[inline]
    fn packed_slice(&self, idx: usize) -> &[u32] {
        &self.packed_data[idx * self.arity..(idx + 1) * self.arity]
    }

    /// Pack a Value tuple to u32. Returns None if any value can't be packed.
    fn pack_tuple(&self, tuple: &[Value]) -> Option<Vec<u32>> {
        let mut packed = Vec::with_capacity(self.arity);
        for (val, ty) in tuple.iter().zip(self.col_types.iter()) {
            packed.push(ty.pack(val)?);
        }
        Some(packed)
    }

    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Sync interpreter-only state (indices, value_data, source_tags) for any tuples inserted
    /// via the JIT fast path since the last sync.
    ///
    /// The JIT insert path (`insert_packed_raw`) skips these structures to avoid per-tuple
    /// overhead when only the JIT is reading data.  Call this before any code that reads
    /// `indices`, `value_data`, or `source_tags` — specifically, before interpreter stratum
    /// evaluation and before exposing results to callers.
    pub(crate) fn ensure_interp_synced(&mut self) {
        if self.interp_synced_count >= self.count {
            return;
        }
        for i in self.interp_synced_count..self.count {
            let start = i * self.arity;
            let packed = &self.packed_data[start..start + self.arity];
            for (col, &p) in packed.iter().enumerate() {
                self.indices[col].entry(p).or_default().push(i);
            }
            for (col, &p) in packed.iter().enumerate() {
                self.value_data.push(self.col_types[col].unpack(p));
            }
            self.source_tags.push(SourceId::ANONYMOUS);
        }
        self.interp_synced_count = self.count;
    }

    /// Look up the index for a packed u32 key in column `col`.
    /// Used by the typed packed JIT helpers.
    pub(crate) fn lookup_packed(&self, col: usize, key: u32, use_recent: bool) -> &[usize] {
        if use_recent {
            self.recent_col_indices[col]
                .get(&key)
                .map_or(&[], |v: &Vec<usize>| v.as_slice())
        } else {
            self.indices[col]
                .get(&key)
                .map_or(&[], |v: &Vec<usize>| v.as_slice())
        }
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn insert(&mut self, tuple: Tuple) -> Result<bool, Tuple> {
        self.try_insert_with_source(tuple, SourceId::ANONYMOUS)
    }

    /// Try to insert a tuple. Returns `Err(tuple)` if the tuple can't be
    /// packed (type mismatch), signaling the caller should downgrade to generic.
    pub fn try_insert_with_source(
        &mut self,
        tuple: Tuple,
        source: SourceId,
    ) -> Result<bool, Tuple> {
        debug_assert_eq!(tuple.len(), self.arity, "tuple arity mismatch");

        if self.arity == 0 {
            if self.count > 0 {
                return Ok(false);
            }
            self.count = 1;
            self.interp_synced_count = 1;
            self.source_tags.push(source);
            self.delta.push(0);
            return Ok(true);
        }

        // Sync any lazily-inserted (JIT-path) tuples so that idx stays consistent.
        self.ensure_interp_synced();

        let Some(packed) = self.pack_tuple(&tuple) else {
            return Err(tuple);
        };

        // Dedup via jit_dedup (also probed inline by JIT code)
        let hash = crate::jit_index::jit_dedup_hash(&packed);
        if !self.jit_dedup.insert_if_new(hash, &packed) {
            return Ok(false);
        }

        // New tuple: append to both buffers
        let idx = self.count;
        for (col, &p) in packed.iter().enumerate() {
            self.indices[col].entry(p).or_default().push(idx);
        }
        self.value_data.extend(tuple);
        self.packed_data.extend(packed);
        self.source_tags.push(source);
        self.count += 1;
        self.interp_synced_count = self.count;
        self.delta.push(idx);
        Ok(true)
    }

    /// Insert a pre-packed u32 tuple directly (no Value→u32 packing step).
    ///
    /// Called from the JIT hot path via `packed_try_insert`.  Skips updating the
    /// interpreter-only structures (`indices`, `value_data`, `source_tags`) to avoid
    /// per-tuple overhead when the JIT is the only reader.  Call `ensure_interp_synced()`
    /// before any code that reads those structures.
    pub fn insert_packed_raw(&mut self, packed: &[u32]) -> bool {
        debug_assert_eq!(packed.len(), self.arity, "packed tuple arity mismatch");

        if self.arity == 0 {
            if self.count > 0 {
                return false;
            }
            self.count = 1;
            self.interp_synced_count = 1;
            self.source_tags.push(SourceId::ANONYMOUS);
            self.delta.push(0);
            return true;
        }

        let hash = crate::jit_index::jit_dedup_hash(packed);
        if !self.jit_dedup.insert_if_new(hash, packed) {
            return false;
        }

        let idx = self.count;
        // Skip indices, value_data, source_tags — deferred to ensure_interp_synced().
        self.packed_data.extend_from_slice(packed);
        self.count += 1;
        self.delta.push(idx);
        true
    }

    /// Insert a pre-packed tuple whose dedup entry has already been written by the JIT.
    ///
    /// Skips the `jit_dedup.insert_if_new` check (the JIT wrote to the dedup table inline).
    /// Updates `indices`, `value_data`, `packed_data`, `source_tags`, `count`, and `delta`.
    ///
    /// # Safety (invariant)
    /// The caller guarantees this tuple is NEW (not already in `packed_data`).  The dedup
    /// table entry has already been filled by the JIT, so `insert_if_new` would return false.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) fn insert_packed_raw_no_dedup(&mut self, packed: &[u32]) {
        debug_assert_eq!(packed.len(), self.arity, "packed tuple arity mismatch");
        let idx = self.count;
        for (col, &p) in packed.iter().enumerate() {
            self.indices[col].entry(p).or_default().push(idx);
        }
        for (col, &p) in packed.iter().enumerate() {
            self.value_data.push(self.col_types[col].unpack(p));
        }
        self.packed_data.extend_from_slice(packed);
        self.source_tags.push(SourceId::ANONYMOUS);
        self.count += 1;
        self.delta.push(idx);
    }

    pub fn contains(&self, tuple: &[Value]) -> bool {
        if self.arity == 0 {
            return self.count > 0;
        }

        let Some(packed) = self.pack_tuple(tuple) else {
            return false;
        };

        let hash = crate::jit_index::jit_dedup_hash(&packed);
        self.jit_dedup.probe(hash, &packed)
    }

    pub fn iter(&self) -> impl Iterator<Item = &[Value]> {
        (0..self.count).map(move |i| self.value_slice(i))
    }

    pub fn iter_recent(&self) -> impl Iterator<Item = &[Value]> {
        self.recent.iter().map(|&i| self.value_slice(i))
    }

    pub fn iter_full(&self) -> impl Iterator<Item = &[Value]> {
        (0..self.count).map(move |i| self.value_slice(i))
    }

    pub fn lookup(&self, col: usize, value: &Value) -> &[usize] {
        let Some(packed_val) = self.col_types[col].pack(value) else {
            return &[];
        };
        self.indices
            .get(col)
            .and_then(|idx| idx.get(&packed_val))
            .map_or(&[], Vec::as_slice)
    }

    pub fn lookup_recent(&self, col: usize, value: &Value) -> &[usize] {
        let Some(packed_val) = self.col_types[col].pack(value) else {
            return &[];
        };
        self.recent_col_indices
            .get(col)
            .and_then(|idx| idx.get(&packed_val))
            .map_or(&[], Vec::as_slice)
    }

    pub fn get(&self, idx: usize) -> &[Value] {
        self.value_slice(idx)
    }

    pub fn has_delta(&self) -> bool {
        !self.delta.is_empty()
    }

    pub fn advance(&mut self) -> bool {
        let had_delta = !self.delta.is_empty();
        self.recent = std::mem::take(&mut self.delta);
        for col_idx in &mut self.recent_col_indices {
            col_idx.clear();
        }
        for &idx in &self.recent {
            let start = idx * self.arity;
            let end = start + self.arity;
            for (col, &p) in self.packed_data[start..end].iter().enumerate() {
                self.recent_col_indices[col].entry(p).or_default().push(idx);
            }
        }
        #[cfg(all(feature = "jit", feature = "specialized"))]
        self.update_jit_indices();
        had_delta
    }

    /// Advance variant for Stage 4 JIT: skips `recent_col_indices` rebuild.
    ///
    /// `recent_col_indices` is only used by the interpreter's `lookup_recent` path.
    /// Stage 4 JIT uses `jit_recent_indices` (rebuilt by `update_jit_indices`) instead.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) fn advance_jit(&mut self) -> bool {
        // Flush any tuples the JIT wrote to jit_native.new into this relation's delta.
        // Takes only the `new` buffer from jit_native so we can update jit_native in-place
        // when nothing changed (avoids a full rebuild of total + recent).
        let new_written = if let Some(ref mut native) = self.jit_native {
            let arity = self.arity;
            // Swap out the `new` buffer, replacing with a fresh empty one.
            let new_buf = std::mem::replace(
                &mut native.new,
                crate::jit::storage::JitRelData::build_from_packed(&[], arity, false),
            );
            let new_len = new_buf.len as usize;
            if new_len > 0 {
                if arity == 0 {
                    // Zero-arity: the JIT wrote len=1 to signal "insert unit fact".
                    self.insert_packed_raw(&[]);
                } else {
                    let data_ptr = new_buf.data;
                    // Safety: data_ptr points to new_len * arity u32 words allocated by
                    // jit_rel_data_grow or build_from_packed.
                    let data = unsafe {
                        std::slice::from_raw_parts(data_ptr, new_len * arity)
                    };
                    for i in 0..new_len {
                        let tuple = &data[i * arity..(i + 1) * arity];
                        self.insert_packed_raw(tuple);
                    }
                }
                true
            } else {
                false
            }
            // new_buf drops here, freeing the old `new` JitRelData allocation.
        } else {
            false
        };

        let had_delta = !self.delta.is_empty();
        self.recent = std::mem::take(&mut self.delta);
        // Skip recent_col_indices rebuild — JIT uses jit_recent_indices.
        self.update_jit_indices();

        // Rebuild jit_native only if it has already been initialized.
        // jit_native is None on the Cranelift path (build_stratum_stage4_native_runtime is
        // never called without jit-asm), so this guard prevents paying the build cost there.
        // The asm native runtime builder explicitly initializes jit_native before first run;
        // after that, this block keeps it fresh on every fixpoint iteration.
        if self.jit_native.is_some() {
            if new_written || had_delta {
                // Data changed: full rebuild.
                self.jit_native = Some(self.build_native_projection());
            } else if self.jit_is_edb {
                // EDB, nothing changed: total is still valid; just reset recent (empty).
                let native = self.jit_native.as_mut().unwrap();
                use crate::jit::storage::JitRelData;
                let build_indices = !self.jit_is_sink;
                native.recent = JitRelData::build_from_packed(&[], self.arity, build_indices);
                // `new` was already reset above by the swap.
            } else {
                // IDB: full rebuild to include any new tuples in total.
                self.jit_native = Some(self.build_native_projection());
            }
        }

        had_delta
    }

    pub fn advance_peek(&mut self) -> bool {
        let had_delta = !self.delta.is_empty();
        self.recent = self.delta.clone();
        for col_idx in &mut self.recent_col_indices {
            col_idx.clear();
        }
        for &idx in &self.recent {
            let start = idx * self.arity;
            let end = start + self.arity;
            for (col, &p) in self.packed_data[start..end].iter().enumerate() {
                self.recent_col_indices[col].entry(p).or_default().push(idx);
            }
        }
        #[cfg(all(feature = "jit", feature = "specialized"))]
        self.update_jit_indices();
        had_delta
    }

    /// Update `jit_indices` and rebuild `jit_recent_indices` using contiguous mode for all relations.
    ///
    /// All relations (EDB and derived) now use `build_contiguous`, eliminating linked-list
    /// pointer chasing in the value loop. The full index is rebuilt whenever new tuples
    /// have been added since the last build.
    ///
    /// Must be called after `advance()` / `advance_peek()` to keep JIT handles fresh.
    #[cfg(all(feature = "jit", feature = "specialized"))]
    pub(crate) fn update_jit_indices(&mut self) {
        if self.jit_is_sink {
            return;
        }
        // ── Full index ───────────────────────────────────────────────────────
        // EDB: one-time build using contiguous mode (col-value for arity-2).
        // Derived: incremental linked-list insertion (O(delta) per advance).
        if self.jit_full_indexed_count < self.count {
            if self.jit_is_edb {
                self.jit_indices = (0..self.arity)
                    .map(|col| {
                        let pairs: Vec<(u32, u32)> = (0..self.count)
                            .map(|idx| {
                                let key = self.packed_data[idx * self.arity + col];
                                let val = if self.arity == 2 {
                                    // EDB arity-2: col-value mode (free col data directly).
                                    self.packed_data[idx * 2 + (1 - col)]
                                } else {
                                    idx as u32
                                };
                                (key, val)
                            })
                            .collect();
                        crate::jit_index::JitHashIndex::build_contiguous(&pairs)
                    })
                    .collect();
                self.jit_index_is_edb_fmt = true;
            } else {
                if self.jit_indices.is_empty() {
                    self.jit_indices = (0..self.arity)
                        .map(|_| crate::jit_index::JitHashIndex::empty())
                        .collect();
                }
                for idx in self.jit_full_indexed_count..self.count {
                    let base = idx * self.arity;
                    if self.arity == 2 {
                        // Col-value linked-list for arity-2 derived: store the other
                        // column's value directly, eliminating a packed_data_ptr call
                        // per inner-loop match in gen_index_scan_v3.
                        self.jit_indices[0].insert(self.packed_data[base], self.packed_data[base + 1]);
                        self.jit_indices[1].insert(self.packed_data[base + 1], self.packed_data[base]);
                    } else {
                        for col in 0..self.arity {
                            self.jit_indices[col].insert(self.packed_data[base + col], idx as u32);
                        }
                    }
                }
                self.jit_index_is_edb_fmt = false;
            }
            self.jit_full_indexed_count = self.count;
        }

        // ── Recent index: always rebuild from current recent set ─────────────
        let arity = self.arity;
        let is_edb = self.jit_is_edb;
        if self.jit_recent_indices.len() != arity {
            self.jit_recent_indices.resize_with(arity, crate::jit_index::JitHashIndex::empty);
        }
        if is_edb {
            for col in 0..arity {
                let pairs: Vec<(u32, u32)> = self.recent
                    .iter()
                    .map(|&idx| {
                        let key = self.packed_data[idx * arity + col];
                        let val = if arity == 2 {
                            self.packed_data[idx * 2 + (1 - col)]
                        } else {
                            idx as u32
                        };
                        (key, val)
                    })
                    .collect();
                self.jit_recent_indices[col] =
                    crate::jit_index::JitHashIndex::build_contiguous(&pairs);
            }
        } else {
            for col in 0..arity {
                self.jit_recent_indices[col].clear_for_rebuild();
            }
            for &idx in &self.recent {
                let base = idx * arity;
                if arity == 2 {
                    // Col-value linked-list for arity-2 derived (mirrors full index).
                    self.jit_recent_indices[0].insert(self.packed_data[base], self.packed_data[base + 1]);
                    self.jit_recent_indices[1].insert(self.packed_data[base + 1], self.packed_data[base]);
                } else {
                    for col in 0..arity {
                        self.jit_recent_indices[col].insert(self.packed_data[base + col], idx as u32);
                    }
                }
            }
        }
    }

    pub fn set_delta_range(&mut self, start: usize) {
        self.delta.clear();
        for i in start..self.count {
            self.delta.push(i);
        }
    }

    pub fn clear_recent(&mut self) {
        self.recent.clear();
    }

    pub fn clear(&mut self) {
        self.value_data.clear();
        self.packed_data.clear();
        self.count = 0;
        self.jit_dedup.clear();
        self.delta.clear();
        self.recent.clear();
        for idx in &mut self.indices {
            idx.clear();
        }
        for idx in &mut self.recent_col_indices {
            idx.clear();
        }
        self.source_tags.clear();
    }

    pub fn recent_indices(&self) -> &[usize] {
        &self.recent
    }

    pub fn tuple_count(&self) -> usize {
        self.count
    }

    pub fn source_of(&self, idx: usize) -> SourceId {
        self.source_tags
            .get(idx)
            .copied()
            .unwrap_or(SourceId::ANONYMOUS)
    }

    pub fn retract_source(&mut self, source: SourceId) -> usize {
        let keep: Vec<bool> = self.source_tags.iter().map(|&s| s != source).collect();
        let removed = keep.iter().filter(|&&k| !k).count();
        if removed == 0 {
            return 0;
        }
        self.rebuild_keeping(&keep);
        removed
    }

    pub fn retract_sources(&mut self, sources: &FxHashSet<SourceId>) -> usize {
        let keep: Vec<bool> = self
            .source_tags
            .iter()
            .map(|s| !sources.contains(s))
            .collect();
        let removed = keep.iter().filter(|&&k| !k).count();
        if removed == 0 {
            return 0;
        }
        self.rebuild_keeping(&keep);
        removed
    }

    fn rebuild_keeping(&mut self, keep: &[bool]) {
        let arity = self.arity;

        if arity == 0 {
            if self.count > 0 && !keep.first().copied().unwrap_or(true) {
                self.count = 0;
                self.source_tags.clear();
            }
            self.delta.clear();
            self.recent.clear();
            return;
        }

        let mut new_value_data = Vec::with_capacity(self.value_data.len());
        let mut new_packed_data = Vec::with_capacity(self.packed_data.len());
        let mut new_sources = Vec::with_capacity(self.source_tags.len());
        let mut new_jit_dedup = crate::jit_index::JitDedupTable::new(arity);
        let mut new_indices: Vec<FxHashMap<u32, Vec<usize>>> =
            (0..arity).map(|_| FxHashMap::default()).collect();
        let mut new_count = 0;

        for (i, &kept) in keep.iter().enumerate().take(self.count) {
            if !kept {
                continue;
            }
            let value_tuple = &self.value_data[i * arity..(i + 1) * arity];
            let packed_tuple = &self.packed_data[i * arity..(i + 1) * arity];
            let idx = new_count;

            new_value_data.extend_from_slice(value_tuple);
            new_packed_data.extend_from_slice(packed_tuple);
            new_sources.push(self.source_tags[i]);

            for (col, &p) in packed_tuple.iter().enumerate() {
                new_indices[col].entry(p).or_default().push(idx);
            }

            let hash = crate::jit_index::jit_dedup_hash(packed_tuple);
            new_jit_dedup.insert(hash, packed_tuple);

            new_count += 1;
        }

        self.value_data = new_value_data;
        self.packed_data = new_packed_data;
        self.source_tags = new_sources;
        self.count = new_count;
        self.jit_dedup = new_jit_dedup;
        self.indices = new_indices;
        self.delta.clear();
        self.recent.clear();
        for col in &mut self.recent_col_indices {
            col.clear();
        }
    }

    /// Convert to a generic RelationStorage, preserving all data and state.
    ///
    /// Called when a packed relation receives a value that can't be packed
    /// (type mismatch), requiring fallback to generic storage.
    pub fn into_generic(self) -> super::RelationStorage {
        use std::hash::{Hash, Hasher};

        let arity = self.arity;
        let mut dedup = HashTable::new();
        let mut indices: Vec<FxHashMap<super::Value, Vec<usize>>> =
            (0..arity).map(|_| FxHashMap::default()).collect();

        // Rebuild dedup and indices from the value buffer
        for i in 0..self.count {
            let tuple = &self.value_data[i * arity..(i + 1) * arity];
            let hash = {
                let mut h = FxHasher::default();
                tuple.hash(&mut h);
                h.finish()
            };
            let data_ref = &self.value_data;
            dedup.insert_unique(hash, i, |&j| {
                let mut h = FxHasher::default();
                data_ref[j * arity..(j + 1) * arity].hash(&mut h);
                h.finish()
            });
            for (col, val) in tuple.iter().enumerate() {
                indices[col].entry(val.clone()).or_default().push(i);
            }
        }

        // Rebuild recent_col_indices
        let mut recent_col_indices: Vec<FxHashMap<super::Value, Vec<usize>>> =
            (0..arity).map(|_| FxHashMap::default()).collect();
        for &idx in &self.recent {
            let tuple = &self.value_data[idx * arity..(idx + 1) * arity];
            for (col, val) in tuple.iter().enumerate() {
                recent_col_indices[col]
                    .entry(val.clone())
                    .or_default()
                    .push(idx);
            }
        }

        // Build recent_set from recent for generic storage (PackedStorage no longer tracks it).
        let recent_set: FxHashSet<usize> = self.recent.iter().copied().collect();

        super::RelationStorage {
            data: self.value_data,
            count: self.count,
            dedup,
            delta: self.delta,
            recent: self.recent,
            recent_set,
            indices,
            recent_col_indices,
            arity,
            is_lattice: false,
            key_index: HashTable::new(),
            source_tags: self.source_tags,
        }
    }
}

impl Clone for PackedStorage {
    /// Clone all interpreter-side state.
    ///
    /// `jit_native` is a rebuild cache; cloned instances start with `None` and
    /// the cache is repopulated on the first `advance_jit()` call.
    fn clone(&self) -> Self {
        Self {
            value_data: self.value_data.clone(),
            packed_data: self.packed_data.clone(),
            col_types: self.col_types.clone(),
            count: self.count,
            arity: self.arity,
            delta: self.delta.clone(),
            recent: self.recent.clone(),
            indices: self.indices.clone(),
            recent_col_indices: self.recent_col_indices.clone(),
            source_tags: self.source_tags.clone(),
            interp_synced_count: self.interp_synced_count,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_indices: self.jit_indices.clone(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_recent_indices: self.jit_recent_indices.clone(),
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_full_indexed_count: self.jit_full_indexed_count,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_is_edb: self.jit_is_edb,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_is_sink: self.jit_is_sink,
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_index_is_edb_fmt: self.jit_index_is_edb_fmt,
            // jit_native is a rebuild cache; start fresh on clone.
            #[cfg(all(feature = "jit", feature = "specialized"))]
            jit_native: None,
            jit_dedup: self.jit_dedup.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_type_roundtrip() {
        let types = [
            (PackedType::I32, Value::I32(42), 42u32),
            (PackedType::I32, Value::I32(-1), u32::MAX),
            (PackedType::U32, Value::U32(100), 100),
            (PackedType::Bool, Value::Bool(true), 1),
            (PackedType::Bool, Value::Bool(false), 0),
        ];
        for (ty, val, expected_raw) in &types {
            let packed = ty.pack(val).unwrap();
            assert_eq!(packed, *expected_raw);
            let unpacked = ty.unpack(packed);
            assert_eq!(&unpacked, val);
        }
    }

    #[test]
    fn test_packed_string_roundtrip() {
        let val = Value::string("hello");
        let pt = PackedType::from_type_name("String").unwrap();
        let packed = pt.pack(&val).unwrap();
        let unpacked = pt.unpack(packed);
        assert_eq!(unpacked, val);
        assert_eq!(format!("{unpacked}"), "hello");
    }

    /// Helper: unwrap insert result (panics on type mismatch, fine for tests).
    fn insert(rel: &mut PackedStorage, tuple: Tuple) -> bool {
        rel.insert(tuple).unwrap()
    }

    /// Helper: unwrap insert_with_source result.
    fn insert_src(rel: &mut PackedStorage, tuple: Tuple, source: SourceId) -> bool {
        rel.try_insert_with_source(tuple, source).unwrap()
    }

    #[test]
    fn test_packed_storage_insert_dedup() {
        let mut rel = PackedStorage::new(vec![PackedType::I32, PackedType::I32]);
        let t1 = vec![Value::I32(1), Value::I32(2)];
        let t2 = vec![Value::I32(3), Value::I32(4)];

        assert!(insert(&mut rel, t1.clone()));
        assert!(!insert(&mut rel, t1.clone())); // duplicate
        assert!(insert(&mut rel, t2.clone()));
        assert_eq!(rel.len(), 2);
        assert!(rel.contains(&t1));
        assert!(rel.contains(&t2));
        assert!(!rel.contains(&[Value::I32(5), Value::I32(6)]));
    }

    #[test]
    fn test_packed_storage_index_lookup() {
        let mut rel = PackedStorage::new(vec![PackedType::I32, PackedType::I32]);
        insert(&mut rel, vec![Value::I32(1), Value::I32(10)]);
        insert(&mut rel, vec![Value::I32(1), Value::I32(20)]);
        insert(&mut rel, vec![Value::I32(2), Value::I32(30)]);

        assert_eq!(rel.lookup(0, &Value::I32(1)).len(), 2);
        assert_eq!(rel.lookup(0, &Value::I32(2)).len(), 1);
        assert_eq!(rel.lookup(1, &Value::I32(10)).len(), 1);
        assert_eq!(rel.lookup(0, &Value::I32(99)).len(), 0);
    }

    #[test]
    fn test_packed_storage_delta_advance() {
        let mut rel = PackedStorage::new(vec![PackedType::I32]);
        insert(&mut rel, vec![Value::I32(1)]);
        insert(&mut rel, vec![Value::I32(2)]);

        assert!(rel.has_delta());
        assert!(rel.advance());
        assert!(!rel.has_delta());
        assert_eq!(rel.recent.len(), 2);

        // Recent lookup works
        assert_eq!(rel.lookup_recent(0, &Value::I32(1)).len(), 1);
        assert_eq!(rel.lookup_recent(0, &Value::I32(2)).len(), 1);

        insert(&mut rel, vec![Value::I32(3)]);
        assert!(rel.advance());
        assert_eq!(rel.recent.len(), 1);
        assert_eq!(rel.len(), 3);
    }

    #[test]
    fn test_packed_storage_retract() {
        let src_a = SourceId(1);
        let src_b = SourceId(2);
        let mut rel = PackedStorage::new(vec![PackedType::I32, PackedType::I32]);
        insert_src(&mut rel, vec![Value::I32(1), Value::I32(2)], src_a);
        insert_src(&mut rel, vec![Value::I32(3), Value::I32(4)], src_b);
        insert_src(&mut rel, vec![Value::I32(5), Value::I32(6)], src_a);
        assert_eq!(rel.len(), 3);

        assert_eq!(rel.retract_source(src_a), 2);
        assert_eq!(rel.len(), 1);
        assert!(rel.contains(&[Value::I32(3), Value::I32(4)]));
        assert!(!rel.contains(&[Value::I32(1), Value::I32(2)]));
    }

    #[test]
    fn test_packed_storage_iter() {
        let mut rel = PackedStorage::new(vec![PackedType::I32, PackedType::I32]);
        insert(&mut rel, vec![Value::I32(1), Value::I32(2)]);
        insert(&mut rel, vec![Value::I32(3), Value::I32(4)]);

        let tuples: Vec<&[Value]> = rel.iter().collect();
        assert_eq!(tuples.len(), 2);
        assert_eq!(tuples[0], &[Value::I32(1), Value::I32(2)]);
        assert_eq!(tuples[1], &[Value::I32(3), Value::I32(4)]);
    }

    #[test]
    fn test_packed_negative_i32() {
        let mut rel = PackedStorage::new(vec![PackedType::I32]);
        insert(&mut rel, vec![Value::I32(-1)]);
        insert(&mut rel, vec![Value::I32(-100)]);
        insert(&mut rel, vec![Value::I32(0)]);

        assert_eq!(rel.len(), 3);
        assert!(rel.contains(&[Value::I32(-1)]));
        assert!(rel.contains(&[Value::I32(-100)]));
        assert!(rel.contains(&[Value::I32(0)]));

        // Verify round-trip through iter
        let vals: Vec<i32> = rel
            .iter()
            .map(|t| {
                if let Value::I32(n) = t[0] {
                    n
                } else {
                    panic!()
                }
            })
            .collect();
        assert!(vals.contains(&-1));
        assert!(vals.contains(&-100));
        assert!(vals.contains(&0));
    }

    #[test]
    fn test_try_packed_col_types() {
        assert!(try_packed_col_types(&[]).is_none());
        assert!(try_packed_col_types(&[Some("i32".into()), Some("i32".into())]).is_some());
        assert!(try_packed_col_types(&[Some("i32".into()), Some("String".into())]).is_some());
        assert!(try_packed_col_types(&[Some("i32".into()), None]).is_none());
        assert!(try_packed_col_types(&[Some("i32".into()), Some("f64".into())]).is_none());
    }
}
