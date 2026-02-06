//! Relation storage for the interpreter.
//!
//! Each relation maintains per-column hash indices for efficient joins.

use std::collections::{HashMap, HashSet};

use crate::value::{Tuple, Value};

/// Storage for a single relation with per-column indices.
#[derive(Debug, Clone, Default)]
pub struct RelationStorage {
    /// All tuples in the relation, stored as a Vec for index-based access.
    tuples: Vec<Tuple>,
    /// Deduplication set (used for non-lattice relations).
    seen: HashSet<Tuple>,
    /// Indices of tuples added in the current iteration (delta).
    delta: Vec<usize>,
    /// Indices of tuples from the previous iteration (for semi-naive).
    recent: Vec<usize>,
    /// Set version of recent for O(1) membership checks.
    recent_set: HashSet<usize>,
    /// Per-column index: (column, value) → list of tuple indices.
    indices: Vec<HashMap<Value, Vec<usize>>>,
    /// Number of columns.
    arity: usize,
    /// Whether this is a lattice relation (last column is the lattice value).
    is_lattice: bool,
    /// For lattice relations: key columns → tuple index (for merge-by-key).
    key_index: HashMap<Vec<Value>, usize>,
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
            seen: HashSet::new(),
            delta: Vec::new(),
            recent: Vec::new(),
            recent_set: HashSet::new(),
            indices: (0..arity).map(|_| HashMap::new()).collect(),
            arity,
            is_lattice,
            key_index: HashMap::new(),
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

        if self.seen.insert(tuple.clone()) {
            let idx = self.tuples.len();
            // Update per-column indices
            for (col, val) in tuple.iter().enumerate() {
                self.indices[col].entry(val.clone()).or_default().push(idx);
            }
            self.tuples.push(tuple);
            self.delta.push(idx);
            true
        } else {
            false
        }
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
                // Lattice value changed: update in place
                let old_val = self.tuples[idx][self.arity - 1].clone();
                self.tuples[idx][self.arity - 1] = joined.clone();

                // Update the last-column index
                let last_col = self.arity - 1;
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
    pub fn contains(&self, tuple: &Tuple) -> bool {
        if self.is_lattice && self.arity > 0 {
            // For lattice relations, look up by key and check the full tuple
            let key = &tuple[..self.arity - 1];
            if let Some(&idx) = self.key_index.get(key) {
                return self.tuples[idx] == *tuple;
            }
            return false;
        }
        self.seen.contains(tuple)
    }

    /// Iterate over all tuples.
    pub fn iter(&self) -> impl Iterator<Item = &Tuple> {
        self.tuples.iter()
    }

    /// Iterate over recent tuples (from last iteration).
    pub fn iter_recent(&self) -> impl Iterator<Item = &Tuple> {
        self.recent.iter().map(|&i| &self.tuples[i])
    }

    /// Iterate over all tuples (for rules that don't use semi-naive).
    pub fn iter_full(&self) -> impl Iterator<Item = &Tuple> {
        self.tuples.iter()
    }

    /// Look up tuples matching a value in the given column.
    pub fn lookup(&self, col: usize, value: &Value) -> &[usize] {
        self.indices
            .get(col)
            .and_then(|idx| idx.get(value))
            .map_or(&[], Vec::as_slice)
    }

    /// Get a tuple by index.
    pub fn get(&self, idx: usize) -> &Tuple {
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
        assert!(rel.contains(&vec![Value::I32(1), Value::I32(20)]));

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
        assert!(rel.contains(&vec![Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));

        // Larger inner value: no change
        assert!(!rel.insert(vec![Value::I32(1), Value::Dual(Box::new(Value::I32(20)))]));
        assert!(rel.contains(&vec![Value::I32(1), Value::Dual(Box::new(Value::I32(5)))]));
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
