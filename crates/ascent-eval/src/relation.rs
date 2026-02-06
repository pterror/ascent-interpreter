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
    /// Deduplication set.
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
}

impl RelationStorage {
    /// Create a new relation with the given arity.
    pub fn new(arity: usize) -> Self {
        Self {
            tuples: Vec::new(),
            seen: HashSet::new(),
            delta: Vec::new(),
            recent: Vec::new(),
            recent_set: HashSet::new(),
            indices: (0..arity).map(|_| HashMap::new()).collect(),
            arity,
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

    /// Insert a tuple. Returns true if it was new.
    pub fn insert(&mut self, tuple: Tuple) -> bool {
        debug_assert_eq!(tuple.len(), self.arity, "tuple arity mismatch");
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

    /// Check if a tuple exists.
    pub fn contains(&self, tuple: &Tuple) -> bool {
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
