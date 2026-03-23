//! Relation storage for the interpreter.
//!
//! Each relation maintains per-column hash indices for efficient joins.
//! Tuples are stored in a flat contiguous buffer with stride-based access.
//!
//! The [`Relation`] enum wraps either a generic [`RelationStorage`] or
//! (with the `specialized` feature) a [`PackedStorage`] for relations
//! whose columns are all u32-representable.

use std::hash::{Hash, Hasher};

use hashbrown::HashTable;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

#[cfg(feature = "specialized")]
use crate::eval::specialized::{PackedStorage, try_packed_col_types};
use crate::eval::value::{Tuple, Value};

/// Opaque identifier for a fact source (file, REPL input, module, etc.).
///
/// Facts can be tagged with a `SourceId` to enable bulk retraction:
/// retract all facts from a given source without touching the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SourceId(pub u32);

impl SourceId {
    /// The default source for untagged facts.
    pub const ANONYMOUS: SourceId = SourceId(0);
}

/// Compute a hash for a tuple slice using FxHasher.
fn hash_tuple(tuple: &[Value]) -> u64 {
    let mut hasher = FxHasher::default();
    tuple.hash(&mut hasher);
    hasher.finish()
}

/// Storage for a single relation with per-column indices.
///
/// Tuples are stored in a flat `Vec<Value>` buffer. Tuple `i` occupies
/// `data[i * arity .. (i+1) * arity]`. Deduplication uses a `HashTable<usize>`
/// storing tuple indices, with custom hash/eq closures over the flat buffer.
#[derive(Debug, Clone)]
pub struct RelationStorage {
    /// Flat tuple data buffer. Tuple `i` is at `data[i*arity..(i+1)*arity]`.
    pub(crate) data: Vec<Value>,
    /// Number of tuples stored (== data.len() / arity, except when arity == 0).
    pub(crate) count: usize,
    /// Deduplication table: stores tuple indices into `data`.
    pub(crate) dedup: HashTable<usize>,
    /// Indices of tuples added in the current iteration (delta).
    pub(crate) delta: Vec<usize>,
    /// Indices of tuples from the previous iteration (for semi-naive).
    pub(crate) recent: Vec<usize>,
    /// Set version of recent for O(1) membership checks.
    pub(crate) recent_set: FxHashSet<usize>,
    /// Per-column index: (column, value) → list of tuple indices.
    pub(crate) indices: Vec<FxHashMap<Value, Vec<usize>>>,
    /// Per-column index for recent tuples only (rebuilt on advance).
    pub(crate) recent_col_indices: Vec<FxHashMap<Value, Vec<usize>>>,
    /// Number of columns.
    pub(crate) arity: usize,
    /// Whether this is a lattice relation (last column is the lattice value).
    pub(crate) is_lattice: bool,
    /// For lattice relations: key columns hash → tuple index (for merge-by-key).
    pub(crate) key_index: HashTable<usize>,
    /// Source tag per tuple, parallel to the flat buffer (one per tuple index).
    pub(crate) source_tags: Vec<SourceId>,
}

impl Default for RelationStorage {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            count: 0,
            dedup: HashTable::new(),
            delta: Vec::new(),
            recent: Vec::new(),
            recent_set: FxHashSet::default(),
            indices: Vec::new(),
            recent_col_indices: Vec::new(),
            arity: 0,
            is_lattice: false,
            key_index: HashTable::new(),
            source_tags: Vec::new(),
        }
    }
}

impl RelationStorage {
    /// Create a new relation with the given arity.
    pub fn new(arity: usize) -> Self {
        Self::with_lattice(arity, false)
    }

    /// Create a new relation with the given arity, optionally as a lattice.
    pub fn with_lattice(arity: usize, is_lattice: bool) -> Self {
        Self {
            data: Vec::new(),
            count: 0,
            dedup: HashTable::new(),
            delta: Vec::new(),
            recent: Vec::new(),
            recent_set: FxHashSet::default(),
            indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            recent_col_indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            arity,
            is_lattice,
            key_index: HashTable::new(),
            source_tags: Vec::new(),
        }
    }

    /// Get the slice for tuple at the given logical index.
    #[inline]
    fn tuple_slice(&self, idx: usize) -> &[Value] {
        &self.data[idx * self.arity..(idx + 1) * self.arity]
    }

    /// Get the arity (number of columns).
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Check if the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the number of tuples.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Insert a tuple with [`SourceId::ANONYMOUS`]. Returns true if data changed.
    ///
    /// For regular relations: deduplicates by full tuple equality.
    /// For lattice relations: merges by key columns (all except last),
    /// applying lattice join on the last column.
    pub fn insert(&mut self, tuple: Tuple) -> bool {
        self.insert_with_source(tuple, SourceId::ANONYMOUS)
    }

    /// Insert a tuple tagged with a source. Returns true if data changed.
    pub fn insert_with_source(&mut self, tuple: Tuple, source: SourceId) -> bool {
        debug_assert_eq!(tuple.len(), self.arity, "tuple arity mismatch");

        if self.is_lattice && self.arity > 0 {
            return self.insert_lattice(tuple, source);
        }

        // Zero-arity: at most one tuple
        if self.arity == 0 {
            if self.count > 0 {
                return false;
            }
            self.count = 1;
            self.source_tags.push(source);
            self.delta.push(0);
            return true;
        }

        // Check for duplicate using dedup table
        let hash = hash_tuple(&tuple);
        let data = &self.data;
        let arity = self.arity;
        if self
            .dedup
            .find(hash, |&idx| {
                &data[idx * arity..(idx + 1) * arity] == tuple.as_slice()
            })
            .is_some()
        {
            return false;
        }

        // New tuple: append to flat buffer
        let idx = self.count;
        for (col, val) in tuple.iter().enumerate() {
            self.indices[col].entry(val.clone()).or_default().push(idx);
        }
        self.data.extend(tuple);
        self.source_tags.push(source);
        self.count += 1;
        self.delta.push(idx);

        // Insert into dedup table
        let data = &self.data;
        self.dedup.insert_unique(hash, idx, |&i| {
            hash_tuple(&data[i * arity..(i + 1) * arity])
        });
        true
    }

    /// Lattice insert: merge by key columns using lattice join on last column.
    fn insert_lattice(&mut self, tuple: Tuple, source: SourceId) -> bool {
        let last_col = self.arity - 1;
        let key = &tuple[..last_col];
        let new_lat = &tuple[last_col];
        let key_hash = hash_tuple(key);

        let data = &self.data;
        let arity = self.arity;

        if let Some(&idx) = self.key_index.find(key_hash, |&idx| {
            &data[idx * arity..idx * arity + last_col] == key
        }) {
            // Key exists: try to merge lattice values
            let old_lat = &self.data[idx * arity + last_col];
            if let Some(joined) = old_lat.lattice_join(new_lat)
                && joined != *old_lat
            {
                // Lattice value changed: mutate in place
                let old_val =
                    std::mem::replace(&mut self.data[idx * arity + last_col], joined.clone());

                // Update the last-column index
                if let Some(entries) = self.indices[last_col].get_mut(&old_val) {
                    entries.retain(|&i| i != idx);
                }
                self.indices[last_col].entry(joined).or_default().push(idx);

                // Update source to the latest writer
                self.source_tags[idx] = source;

                // Mark as changed
                if !self.delta.contains(&idx) {
                    self.delta.push(idx);
                }
                return true;
            }
            false
        } else {
            // New key: insert fresh tuple
            let idx = self.count;
            for (col, val) in tuple.iter().enumerate() {
                self.indices[col].entry(val.clone()).or_default().push(idx);
            }

            // Insert into key_index
            let data_ref = &self.data;
            self.key_index.insert_unique(key_hash, idx, |&i| {
                hash_tuple(&data_ref[i * arity..i * arity + last_col])
            });

            self.data.extend(tuple);
            self.source_tags.push(source);
            self.count += 1;
            self.delta.push(idx);
            true
        }
    }

    /// Check if a tuple exists.
    pub fn contains(&self, tuple: &[Value]) -> bool {
        if self.arity == 0 {
            return self.count > 0;
        }

        if self.is_lattice && self.arity > 0 {
            // For lattice relations, look up by key and check the full tuple
            let last_col = self.arity - 1;
            let key = &tuple[..last_col];
            let key_hash = hash_tuple(key);
            let data = &self.data;
            let arity = self.arity;
            if let Some(&idx) = self.key_index.find(key_hash, |&idx| {
                &data[idx * arity..idx * arity + last_col] == key
            }) {
                return self.tuple_slice(idx) == tuple;
            }
            return false;
        }

        let hash = hash_tuple(tuple);
        let data = &self.data;
        let arity = self.arity;
        self.dedup
            .find(hash, |&idx| &data[idx * arity..(idx + 1) * arity] == tuple)
            .is_some()
    }

    /// Iterate over all tuples.
    pub fn iter(&self) -> impl Iterator<Item = &[Value]> {
        (0..self.count).map(move |i| self.tuple_slice(i))
    }

    /// Iterate over recent tuples (from last iteration).
    pub fn iter_recent(&self) -> impl Iterator<Item = &[Value]> {
        self.recent.iter().map(|&i| self.tuple_slice(i))
    }

    /// Iterate over all tuples (for rules that don't use semi-naive).
    pub fn iter_full(&self) -> impl Iterator<Item = &[Value]> {
        (0..self.count).map(move |i| self.tuple_slice(i))
    }

    /// Look up tuples matching a value in the given column.
    pub fn lookup(&self, col: usize, value: &Value) -> &[usize] {
        self.indices
            .get(col)
            .and_then(|idx| idx.get(value))
            .map_or(&[], Vec::as_slice)
    }

    /// Look up only recent tuples matching a value in the given column.
    pub fn lookup_recent(&self, col: usize, value: &Value) -> &[usize] {
        self.recent_col_indices
            .get(col)
            .and_then(|idx| idx.get(value))
            .map_or(&[], Vec::as_slice)
    }

    /// Get a tuple by index.
    pub fn get(&self, idx: usize) -> &[Value] {
        self.tuple_slice(idx)
    }

    /// Check if there are new tuples in the delta.
    pub fn has_delta(&self) -> bool {
        !self.delta.is_empty()
    }

    /// Move delta to recent, clear delta. Returns true if there were changes.
    pub fn advance(&mut self) -> bool {
        let had_delta = !self.delta.is_empty();
        self.recent = std::mem::take(&mut self.delta);
        self.recent_set = self.recent.iter().copied().collect();
        // Rebuild per-column indices for recent tuples only
        for col_idx in &mut self.recent_col_indices {
            col_idx.clear();
        }
        for &idx in &self.recent {
            let start = idx * self.arity;
            let end = start + self.arity;
            for (col, val) in self.data[start..end].iter().enumerate() {
                self.recent_col_indices[col]
                    .entry(val.clone())
                    .or_default()
                    .push(idx);
            }
        }
        had_delta
    }

    /// Advance delta to recent without consuming delta.
    ///
    /// Like `advance`, but preserves the delta so downstream strata can still
    /// see it. Used for input relations during incremental stratum evaluation.
    pub fn advance_peek(&mut self) -> bool {
        let had_delta = !self.delta.is_empty();
        self.recent = self.delta.clone();
        self.recent_set = self.recent.iter().copied().collect();
        for col_idx in &mut self.recent_col_indices {
            col_idx.clear();
        }
        for &idx in &self.recent {
            let start = idx * self.arity;
            let end = start + self.arity;
            for (col, val) in self.data[start..end].iter().enumerate() {
                self.recent_col_indices[col]
                    .entry(val.clone())
                    .or_default()
                    .push(idx);
            }
        }
        had_delta
    }

    /// Set delta to contain indices in [start..self.count).
    ///
    /// Used after incremental stratum evaluation to propagate newly-derived
    /// tuples as deltas for downstream strata.
    pub fn set_delta_range(&mut self, start: usize) {
        self.delta.clear();
        for i in start..self.count {
            self.delta.push(i);
        }
    }

    /// Clear the recent set (after all rules have been evaluated).
    pub fn clear_recent(&mut self) {
        self.recent.clear();
        self.recent_set.clear();
    }

    /// Remove all data from this relation, resetting it to empty.
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
        self.dedup = HashTable::new();
        self.delta.clear();
        self.recent.clear();
        self.recent_set.clear();
        for idx in &mut self.indices {
            idx.clear();
        }
        for idx in &mut self.recent_col_indices {
            idx.clear();
        }
        self.key_index = HashTable::new();
        self.source_tags.clear();
    }

    /// Check if a tuple index is in the recent set (O(1)).
    pub fn is_recent(&self, idx: usize) -> bool {
        self.recent_set.contains(&idx)
    }

    /// Iterate over recent tuple indices.
    pub fn recent_indices(&self) -> &[usize] {
        &self.recent
    }

    /// Total number of tuples (for index bounds).
    pub fn tuple_count(&self) -> usize {
        self.count
    }

    /// Get the source tag for a tuple by index.
    pub fn source_of(&self, idx: usize) -> SourceId {
        self.source_tags
            .get(idx)
            .copied()
            .unwrap_or(SourceId::ANONYMOUS)
    }

    /// Remove all tuples tagged with the given source. Returns the number removed.
    ///
    /// Rebuilds internal data structures (indices, dedup) from surviving tuples.
    /// Delta and recent are cleared since they reference pre-retraction indices.
    pub fn retract_source(&mut self, source: SourceId) -> usize {
        let keep: Vec<bool> = self.source_tags.iter().map(|&s| s != source).collect();
        let removed = keep.iter().filter(|&&k| !k).count();
        if removed == 0 {
            return 0;
        }
        self.rebuild_keeping(&keep);
        removed
    }

    /// Remove all tuples tagged with any of the given sources. Returns the number removed.
    ///
    /// More efficient than calling `retract_source` in a loop — does a single rebuild pass.
    pub fn retract_sources(&mut self, sources: &rustc_hash::FxHashSet<SourceId>) -> usize {
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

    /// Rebuild the relation keeping only tuples where `keep[i]` is true.
    fn rebuild_keeping(&mut self, keep: &[bool]) {
        let arity = self.arity;

        // Handle zero-arity specially
        if arity == 0 {
            if self.count > 0 && !keep.first().copied().unwrap_or(true) {
                self.count = 0;
                self.source_tags.clear();
            }
            self.delta.clear();
            self.recent.clear();
            self.recent_set.clear();
            return;
        }

        let mut new_data = Vec::with_capacity(self.data.len());
        let mut new_sources = Vec::with_capacity(self.source_tags.len());
        let mut new_dedup = HashTable::new();
        let mut new_indices: Vec<FxHashMap<Value, Vec<usize>>> =
            (0..arity).map(|_| FxHashMap::default()).collect();
        let mut new_key_index = HashTable::new();
        let mut new_count = 0;

        for (i, &kept) in keep.iter().enumerate().take(self.count) {
            if !kept {
                continue;
            }
            let tuple = &self.data[i * arity..(i + 1) * arity];
            let idx = new_count;

            new_data.extend_from_slice(tuple);
            new_sources.push(self.source_tags[i]);

            for (col, val) in tuple.iter().enumerate() {
                new_indices[col].entry(val.clone()).or_default().push(idx);
            }

            let hash = hash_tuple(tuple);
            new_dedup.insert_unique(hash, idx, |&j| {
                hash_tuple(&new_data[j * arity..(j + 1) * arity])
            });

            if self.is_lattice && arity > 0 {
                let key = &tuple[..arity - 1];
                let key_hash = hash_tuple(key);
                new_key_index.insert_unique(key_hash, idx, |&j| {
                    hash_tuple(&new_data[j * arity..j * arity + arity - 1])
                });
            }

            new_count += 1;
        }

        self.data = new_data;
        self.source_tags = new_sources;
        self.count = new_count;
        self.dedup = new_dedup;
        self.indices = new_indices;
        self.key_index = new_key_index;
        // Delta/recent reference old indices — clear them
        self.delta.clear();
        self.recent.clear();
        self.recent_set.clear();
        for col in &mut self.recent_col_indices {
            col.clear();
        }
    }
}

/// Unified relation type that dispatches to either generic or packed storage.
///
/// Without the `specialized` feature, this is always `Generic`.
/// With the feature, relations whose columns are all u32-representable
/// (i32, u32, interned strings, booleans) use [`PackedStorage`] for
/// faster dedup and index operations.
#[derive(Debug, Clone)]
pub enum Relation {
    Generic(RelationStorage),
    #[cfg(feature = "specialized")]
    Packed(PackedStorage),
}

impl Default for Relation {
    fn default() -> Self {
        Relation::Generic(RelationStorage::default())
    }
}

impl Relation {
    /// Create a generic relation with the given arity.
    pub fn new(arity: usize) -> Self {
        Relation::Generic(RelationStorage::new(arity))
    }

    /// Create a generic relation with the given arity, optionally as a lattice.
    pub fn with_lattice(arity: usize, is_lattice: bool) -> Self {
        Relation::Generic(RelationStorage::with_lattice(arity, is_lattice))
    }

    /// Create a relation, using packed storage when all columns are u32-representable.
    ///
    /// Falls back to generic storage for lattice relations, zero-arity relations,
    /// or relations with non-packable column types.
    pub fn new_auto(arity: usize, is_lattice: bool, _col_types: &[Option<String>]) -> Self {
        #[cfg(feature = "specialized")]
        if !is_lattice && let Some(packed_types) = try_packed_col_types(_col_types) {
            return Relation::Packed(PackedStorage::new(packed_types));
        }
        Relation::Generic(RelationStorage::with_lattice(arity, is_lattice))
    }

    /// Whether this relation uses packed storage.
    pub fn is_packed(&self) -> bool {
        match self {
            Relation::Generic(_) => false,
            #[cfg(feature = "specialized")]
            Relation::Packed(_) => true,
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Relation::Generic(r) => r.arity(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.arity(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Relation::Generic(r) => r.is_empty(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Relation::Generic(r) => r.len(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.len(),
        }
    }

    pub fn insert(&mut self, tuple: Tuple) -> bool {
        self.insert_with_source(tuple, SourceId::ANONYMOUS)
    }

    pub fn insert_with_source(&mut self, tuple: Tuple, source: SourceId) -> bool {
        match self {
            Relation::Generic(r) => r.insert_with_source(tuple, source),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => match p.try_insert_with_source(tuple, source) {
                Ok(inserted) => inserted,
                Err(tuple) => {
                    // Type mismatch: downgrade to generic storage
                    let packed = match std::mem::take(self) {
                        Relation::Packed(p) => p,
                        _ => unreachable!(),
                    };
                    *self = Relation::Generic(packed.into_generic());
                    self.insert_with_source(tuple, source)
                }
            },
        }
    }

    pub fn contains(&self, tuple: &[Value]) -> bool {
        match self {
            Relation::Generic(r) => r.contains(tuple),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.contains(tuple),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &[Value]> + '_ {
        let (data, count, arity) = match self {
            Relation::Generic(r) => (r.data.as_slice(), r.count, r.arity),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => (p.value_data.as_slice(), p.count, p.arity),
        };
        (0..count).map(move |i| &data[i * arity..(i + 1) * arity])
    }

    pub fn iter_recent(&self) -> impl Iterator<Item = &[Value]> + '_ {
        let (data, arity, recent) = match self {
            Relation::Generic(r) => (r.data.as_slice(), r.arity, r.recent.as_slice()),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => (p.value_data.as_slice(), p.arity, p.recent.as_slice()),
        };
        recent
            .iter()
            .map(move |&i| &data[i * arity..(i + 1) * arity])
    }

    pub fn iter_full(&self) -> impl Iterator<Item = &[Value]> + '_ {
        let (data, count, arity) = match self {
            Relation::Generic(r) => (r.data.as_slice(), r.count, r.arity),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => (p.value_data.as_slice(), p.count, p.arity),
        };
        (0..count).map(move |i| &data[i * arity..(i + 1) * arity])
    }

    pub fn lookup(&self, col: usize, value: &Value) -> &[usize] {
        match self {
            Relation::Generic(r) => r.lookup(col, value),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.lookup(col, value),
        }
    }

    pub fn lookup_recent(&self, col: usize, value: &Value) -> &[usize] {
        match self {
            Relation::Generic(r) => r.lookup_recent(col, value),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.lookup_recent(col, value),
        }
    }

    pub fn get(&self, idx: usize) -> &[Value] {
        match self {
            Relation::Generic(r) => r.get(idx),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.get(idx),
        }
    }

    pub fn has_delta(&self) -> bool {
        match self {
            Relation::Generic(r) => r.has_delta(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.has_delta(),
        }
    }

    pub fn advance(&mut self) -> bool {
        match self {
            Relation::Generic(r) => r.advance(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.advance(),
        }
    }

    pub fn advance_peek(&mut self) -> bool {
        match self {
            Relation::Generic(r) => r.advance_peek(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.advance_peek(),
        }
    }

    pub fn set_delta_range(&mut self, start: usize) {
        match self {
            Relation::Generic(r) => r.set_delta_range(start),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.set_delta_range(start),
        }
    }

    pub fn clear_recent(&mut self) {
        match self {
            Relation::Generic(r) => r.clear_recent(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.clear_recent(),
        }
    }

    pub fn clear(&mut self) {
        match self {
            Relation::Generic(r) => r.clear(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.clear(),
        }
    }

    pub fn is_recent(&self, idx: usize) -> bool {
        match self {
            Relation::Generic(r) => r.is_recent(idx),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.recent.contains(&idx),
        }
    }

    pub fn recent_indices(&self) -> &[usize] {
        match self {
            Relation::Generic(r) => r.recent_indices(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.recent_indices(),
        }
    }

    pub fn tuple_count(&self) -> usize {
        match self {
            Relation::Generic(r) => r.tuple_count(),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.tuple_count(),
        }
    }

    pub fn source_of(&self, idx: usize) -> SourceId {
        match self {
            Relation::Generic(r) => r.source_of(idx),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.source_of(idx),
        }
    }

    pub fn retract_source(&mut self, source: SourceId) -> usize {
        match self {
            Relation::Generic(r) => r.retract_source(source),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.retract_source(source),
        }
    }

    pub fn retract_sources(&mut self, sources: &FxHashSet<SourceId>) -> usize {
        match self {
            Relation::Generic(r) => r.retract_sources(sources),
            #[cfg(feature = "specialized")]
            Relation::Packed(p) => p.retract_sources(sources),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::value::Value;

    #[test]
    fn test_insert_and_contains() {
        let mut rel = RelationStorage::new(2);
        let tuple = vec![Value::I32(1), Value::I32(2)];

        assert!(rel.insert(tuple.clone()));
        assert!(!rel.insert(tuple.clone())); // duplicate
        assert!(rel.contains(&tuple));
        assert_eq!(rel.len(), 1);
    }

    #[test]
    fn test_delta_and_advance() {
        let mut rel = RelationStorage::new(1);

        rel.insert(vec![Value::I32(1)]);
        rel.insert(vec![Value::I32(2)]);

        assert!(rel.has_delta());
        assert_eq!(rel.delta.len(), 2);
        assert_eq!(rel.recent.len(), 0);

        rel.advance();

        assert!(!rel.has_delta());
        assert_eq!(rel.delta.len(), 0);
        assert_eq!(rel.recent.len(), 2);

        // Insert more
        rel.insert(vec![Value::I32(3)]);
        rel.advance();

        assert_eq!(rel.recent.len(), 1);
        assert_eq!(rel.len(), 3);
    }

    #[test]
    fn test_lattice_insert_merge() {
        let mut rel = RelationStorage::with_lattice(2, true);

        // First insert: key=1, lattice=10
        assert!(rel.insert(vec![Value::I32(1), Value::I32(10)]));
        assert_eq!(rel.len(), 1);

        // Same key, smaller value: no change (join = max for i32)
        assert!(!rel.insert(vec![Value::I32(1), Value::I32(5)]));
        assert_eq!(rel.len(), 1);

        // Same key, larger value: merge takes max
        assert!(rel.insert(vec![Value::I32(1), Value::I32(20)]));
        assert_eq!(rel.len(), 1);
        assert!(rel.contains(&[Value::I32(1), Value::I32(20)]));

        // Different key: new tuple
        assert!(rel.insert(vec![Value::I32(2), Value::I32(15)]));
        assert_eq!(rel.len(), 2);
    }

    #[test]
    fn test_lattice_dual_merge() {
        let mut rel = RelationStorage::with_lattice(2, true);

        // Dual join = min of inner values
        rel.insert(vec![Value::I32(1), Value::Dual(Box::new(Value::I32(10)))]);
        assert_eq!(rel.len(), 1);

        // Smaller inner value wins for Dual
        assert!(rel.insert(vec![Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));
        assert!(rel.contains(&[Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));

        // Larger inner value: no change
        assert!(!rel.insert(vec![Value::I32(1), Value::Dual(Box::new(Value::I32(20)))]));
        assert!(rel.contains(&[Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));
    }

    #[test]
    fn test_dedup_with_mixed_types() {
        let mut rel = RelationStorage::new(2);

        // Pure i32 inserts
        assert!(rel.insert(vec![Value::I32(1), Value::I32(2)]));
        assert!(!rel.insert(vec![Value::I32(1), Value::I32(2)])); // duplicate
        assert_eq!(rel.len(), 1);

        // Contains works
        assert!(rel.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(!rel.contains(&[Value::I32(3), Value::I32(4)]));

        // Mixed types work seamlessly
        assert!(rel.insert(vec![Value::I32(5), Value::string("hello")]));
        assert_eq!(rel.len(), 2);

        // Dedup still works for both
        assert!(!rel.insert(vec![Value::I32(1), Value::I32(2)]));
        assert!(!rel.insert(vec![Value::I32(5), Value::string("hello")]));
        assert_eq!(rel.len(), 2);

        // Contains works for both
        assert!(rel.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(rel.contains(&[Value::I32(5), Value::string("hello")]));
    }

    #[test]
    fn test_index_lookup() {
        let mut rel = RelationStorage::new(2);
        rel.insert(vec![Value::I32(1), Value::I32(10)]);
        rel.insert(vec![Value::I32(1), Value::I32(20)]);
        rel.insert(vec![Value::I32(2), Value::I32(30)]);

        // Lookup col 0 = 1 → should return 2 tuple indices
        let matches = rel.lookup(0, &Value::I32(1));
        assert_eq!(matches.len(), 2);

        // Lookup col 0 = 2 → should return 1 tuple index
        let matches = rel.lookup(0, &Value::I32(2));
        assert_eq!(matches.len(), 1);

        // Lookup col 1 = 10 → should return 1 tuple index
        let matches = rel.lookup(1, &Value::I32(10));
        assert_eq!(matches.len(), 1);

        // Lookup col 0 = 99 → should return 0
        let matches = rel.lookup(0, &Value::I32(99));
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_zero_arity() {
        let mut rel = RelationStorage::new(0);
        assert!(rel.insert(vec![]));
        assert!(!rel.insert(vec![])); // duplicate
        assert_eq!(rel.len(), 1);
        assert!(rel.contains(&[]));
    }

    #[test]
    fn test_relation_new_auto_generic() {
        // Lattice → always generic
        let rel = Relation::new_auto(2, true, &[Some("i32".into()), Some("i32".into())]);
        assert!(!rel.is_packed());
    }

    #[test]
    fn test_relation_new_auto_unknown_types() {
        // Unknown column type → generic
        let rel = Relation::new_auto(2, false, &[Some("i32".into()), None]);
        assert!(!rel.is_packed());
    }

    #[cfg(feature = "specialized")]
    #[test]
    fn test_relation_new_auto_packed() {
        let rel = Relation::new_auto(2, false, &[Some("i32".into()), Some("i32".into())]);
        assert!(rel.is_packed());
    }

    #[cfg(feature = "specialized")]
    #[test]
    fn test_relation_packed_downgrade() {
        // Create packed relation, insert non-packable value → downgrades to generic
        let mut rel = Relation::new_auto(2, false, &[Some("i32".into()), Some("i32".into())]);
        assert!(rel.is_packed());

        // Normal packed insert
        assert!(rel.insert(vec![Value::I32(1), Value::I32(2)]));
        assert!(rel.is_packed());

        // Insert non-packable value → triggers downgrade
        assert!(rel.insert(vec![Value::I32(3), Value::string("oops")]));
        assert!(!rel.is_packed());

        // All data preserved after downgrade
        assert_eq!(rel.len(), 2);
        assert!(rel.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(rel.contains(&[Value::I32(3), Value::string("oops")]));
    }

    #[cfg(feature = "specialized")]
    #[test]
    fn test_relation_packed_via_enum() {
        // Verify the Relation enum delegates correctly to packed storage
        let mut rel = Relation::new_auto(2, false, &[Some("i32".into()), Some("i32".into())]);
        assert!(rel.is_packed());

        rel.insert(vec![Value::I32(1), Value::I32(10)]);
        rel.insert(vec![Value::I32(1), Value::I32(20)]);
        rel.insert(vec![Value::I32(2), Value::I32(30)]);
        assert_eq!(rel.len(), 3);

        // Index lookup through enum
        assert_eq!(rel.lookup(0, &Value::I32(1)).len(), 2);
        assert_eq!(rel.lookup(0, &Value::I32(2)).len(), 1);

        // Delta/advance through enum
        assert!(rel.has_delta());
        assert!(rel.advance());
        assert!(!rel.has_delta());
        assert_eq!(rel.lookup_recent(0, &Value::I32(1)).len(), 2);

        // Iter through enum
        let count = rel.iter_full().count();
        assert_eq!(count, 3);
    }
}
