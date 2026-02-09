//! Relation storage for the interpreter.
//!
//! Each relation maintains per-column hash indices for efficient joins.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::value::{Tuple, Value};

/// Maximum arity for the i32 stack-array fast path.
const I32_FAST_PATH_MAX_ARITY: usize = 8;

/// Storage for a single relation with per-column indices.
#[derive(Debug, Clone, Default)]
pub struct RelationStorage {
    /// All tuples in the relation.
    tuples: Vec<Vec<Value>>,
    /// Deduplication set (used when `all_i32` is false).
    seen: FxHashSet<Vec<Value>>,
    /// Whether all tuples so far contain only `Value::I32` values.
    all_i32: bool,
    /// i32-specialized dedup set (used when `all_i32` is true).
    seen_i32: FxHashSet<Vec<i32>>,
    /// Indices of tuples added in the current iteration (delta).
    delta: Vec<usize>,
    /// Indices of tuples from the previous iteration (for semi-naive).
    recent: Vec<usize>,
    /// Set version of recent for O(1) membership checks.
    recent_set: FxHashSet<usize>,
    /// Per-column index: (column, value) → list of tuple indices.
    indices: Vec<FxHashMap<Value, Vec<usize>>>,
    /// Per-column index for recent tuples only (rebuilt on advance).
    recent_col_indices: Vec<FxHashMap<Value, Vec<usize>>>,
    /// Number of columns.
    arity: usize,
    /// Whether this is a lattice relation (last column is the lattice value).
    is_lattice: bool,
    /// For lattice relations: key columns → tuple index (for merge-by-key).
    key_index: FxHashMap<Vec<Value>, usize>,
}

impl RelationStorage {
    /// Create a new relation with the given arity.
    pub fn new(arity: usize) -> Self {
        Self::with_lattice(arity, false)
    }

    /// Create a new relation with the given arity, optionally as a lattice.
    pub fn with_lattice(arity: usize, is_lattice: bool) -> Self {
        Self {
            tuples: Vec::new(),
            seen: FxHashSet::default(),
            all_i32: !is_lattice,
            seen_i32: FxHashSet::default(),
            delta: Vec::new(),
            recent: Vec::new(),
            recent_set: FxHashSet::default(),
            indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            recent_col_indices: (0..arity).map(|_| FxHashMap::default()).collect(),
            arity,
            is_lattice,
            key_index: FxHashMap::default(),
        }
    }

    /// Get the arity (number of columns).
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Check if the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Get the number of tuples.
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Insert a tuple. Returns true if data changed.
    ///
    /// For regular relations: deduplicates by full tuple equality.
    /// For lattice relations: merges by key columns (all except last),
    /// applying lattice join on the last column.
    pub fn insert(&mut self, tuple: Tuple) -> bool {
        debug_assert_eq!(tuple.len(), self.arity, "tuple arity mismatch");

        if self.is_lattice && self.arity > 0 {
            return self.insert_lattice(tuple);
        }

        // i32 fast path: use specialized dedup set for all-i32 tuples with arity ≤ 8
        if self.all_i32 && self.arity <= I32_FAST_PATH_MAX_ARITY {
            if let Some(is_new) = self.try_insert_i32(&tuple) {
                if !is_new {
                    return false;
                }
                // New tuple via i32 fast path — skip `seen`, go straight to storage
                let idx = self.tuples.len();
                for (col, val) in tuple.iter().enumerate() {
                    self.indices[col].entry(val.clone()).or_default().push(idx);
                }
                self.tuples.push(tuple);
                self.delta.push(idx);
                return true;
            }
            // Non-i32 value encountered: migrate to Value mode
            self.migrate_to_value_mode();
        }

        if !self.seen.insert(tuple.clone()) {
            return false;
        }
        let idx = self.tuples.len();
        // Update per-column indices
        for (col, val) in tuple.iter().enumerate() {
            self.indices[col].entry(val.clone()).or_default().push(idx);
        }
        self.tuples.push(tuple);
        self.delta.push(idx);
        true
    }

    /// Try to extract i32 values from a tuple into a stack buffer.
    /// Returns `None` if any value is not `Value::I32`.
    fn extract_i32s(tuple: &[Value]) -> Option<[i32; I32_FAST_PATH_MAX_ARITY]> {
        let mut buf = [0i32; I32_FAST_PATH_MAX_ARITY];
        for (i, v) in tuple.iter().enumerate() {
            match v {
                Value::I32(n) => buf[i] = *n,
                _ => return None,
            }
        }
        Some(buf)
    }

    /// Try to insert using the i32 fast path.
    /// Returns `Some(true)` if inserted, `Some(false)` if duplicate, `None` if non-i32.
    fn try_insert_i32(&mut self, tuple: &[Value]) -> Option<bool> {
        let buf = Self::extract_i32s(tuple)?;
        let slice = &buf[..self.arity];
        if self.seen_i32.contains(slice) {
            Some(false)
        } else {
            self.seen_i32.insert(slice.to_vec());
            Some(true)
        }
    }

    /// Migrate from i32-specialized dedup to generic Value dedup.
    /// Rebuilds `seen` from existing tuples and clears `seen_i32`.
    fn migrate_to_value_mode(&mut self) {
        self.all_i32 = false;
        self.seen = self.tuples.iter().cloned().collect();
        self.seen_i32 = FxHashSet::default();
    }

    /// Lattice insert: merge by key columns using lattice join on last column.
    fn insert_lattice(&mut self, tuple: Tuple) -> bool {
        let key: Vec<Value> = tuple[..self.arity - 1].to_vec();
        let new_lat = &tuple[self.arity - 1];

        if let Some(&idx) = self.key_index.get(&key) {
            // Key exists: try to merge lattice values
            let old_lat = &self.tuples[idx][self.arity - 1];
            if let Some(joined) = old_lat.lattice_join(new_lat)
                && joined != *old_lat
            {
                // Lattice value changed: mutate in place
                let last_col = self.arity - 1;
                let old_val = std::mem::replace(&mut self.tuples[idx][last_col], joined.clone());

                // Update the last-column index
                if let Some(entries) = self.indices[last_col].get_mut(&old_val) {
                    entries.retain(|&i| i != idx);
                }
                self.indices[last_col].entry(joined).or_default().push(idx);

                // Mark as changed
                if !self.delta.contains(&idx) {
                    self.delta.push(idx);
                }
                return true;
            }
            false
        } else {
            // New key: insert fresh tuple
            let idx = self.tuples.len();
            for (col, val) in tuple.iter().enumerate() {
                self.indices[col].entry(val.clone()).or_default().push(idx);
            }
            self.key_index.insert(key, idx);
            self.tuples.push(tuple);
            self.delta.push(idx);
            true
        }
    }

    /// Check if a tuple exists.
    pub fn contains(&self, tuple: &[Value]) -> bool {
        if self.is_lattice && self.arity > 0 {
            // For lattice relations, look up by key and check the full tuple
            let key = &tuple[..self.arity - 1];
            if let Some(&idx) = self.key_index.get(key) {
                return *self.tuples[idx] == *tuple;
            }
            return false;
        }
        // i32 fast path
        if self.all_i32
            && self.arity <= I32_FAST_PATH_MAX_ARITY
            && let Some(buf) = Self::extract_i32s(tuple)
        {
            return self.seen_i32.contains(&buf[..self.arity]);
        }
        self.seen.contains(tuple)
    }

    /// Iterate over all tuples.
    pub fn iter(&self) -> impl Iterator<Item = &[Value]> {
        self.tuples.iter().map(|t| t.as_slice())
    }

    /// Iterate over recent tuples (from last iteration).
    pub fn iter_recent(&self) -> impl Iterator<Item = &[Value]> {
        self.recent.iter().map(|&i| self.tuples[i].as_slice())
    }

    /// Iterate over all tuples (for rules that don't use semi-naive).
    pub fn iter_full(&self) -> impl Iterator<Item = &[Value]> {
        self.tuples.iter().map(|t| t.as_slice())
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
        &self.tuples[idx]
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
            for (col, val) in self.tuples[idx].iter().enumerate() {
                self.recent_col_indices[col]
                    .entry(val.clone())
                    .or_default()
                    .push(idx);
            }
        }
        had_delta
    }

    /// Clear the recent set (after all rules have been evaluated).
    pub fn clear_recent(&mut self) {
        self.recent.clear();
        self.recent_set.clear();
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
        self.tuples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

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
    fn test_i32_fast_path() {
        let mut rel = RelationStorage::new(2);
        assert!(rel.all_i32);

        // Pure i32 inserts use fast path
        assert!(rel.insert(vec![Value::I32(1), Value::I32(2)]));
        assert!(rel.all_i32);
        assert_eq!(rel.seen_i32.len(), 1);
        assert!(rel.seen.is_empty()); // Value dedup set not used

        // Duplicate rejected via fast path
        assert!(!rel.insert(vec![Value::I32(1), Value::I32(2)]));
        assert_eq!(rel.len(), 1);

        // Contains works via fast path
        assert!(rel.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(!rel.contains(&[Value::I32(3), Value::I32(4)]));
    }

    #[test]
    fn test_i32_migration_on_mixed_types() {
        let mut rel = RelationStorage::new(2);

        // Insert i32 tuples
        rel.insert(vec![Value::I32(1), Value::I32(2)]);
        rel.insert(vec![Value::I32(3), Value::I32(4)]);
        assert!(rel.all_i32);
        assert_eq!(rel.seen_i32.len(), 2);

        // Insert non-i32 tuple triggers migration
        rel.insert(vec![
            Value::I32(5),
            Value::String(String::from("hello").into()),
        ]);
        assert!(!rel.all_i32);
        assert!(rel.seen_i32.is_empty());
        assert_eq!(rel.seen.len(), 3); // All 3 tuples in Value dedup set
        assert_eq!(rel.len(), 3);

        // Contains still works after migration
        assert!(rel.contains(&[Value::I32(1), Value::I32(2)]));
        assert!(rel.contains(&[Value::I32(5), Value::String(String::from("hello").into())]));
    }

    #[test]
    fn test_i32_migration_preserves_dedup() {
        let mut rel = RelationStorage::new(1);

        rel.insert(vec![Value::I32(1)]);
        rel.insert(vec![Value::I32(2)]);

        // Force migration
        rel.insert(vec![Value::String(String::from("x").into())]);

        // Duplicate after migration should still be rejected
        assert!(!rel.insert(vec![Value::I32(1)]));
        assert_eq!(rel.len(), 3);
    }

    #[test]
    fn test_lattice_skips_i32_optimization() {
        let rel = RelationStorage::with_lattice(2, true);
        assert!(!rel.all_i32);
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
}
