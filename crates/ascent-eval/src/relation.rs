//! Relation storage for the interpreter.

use std::collections::HashSet;

use crate::value::Tuple;

/// Storage for a single relation.
#[derive(Debug, Clone, Default)]
pub struct RelationStorage {
    /// All tuples in the relation.
    full: HashSet<Tuple>,
    /// Tuples added in the current iteration (delta).
    delta: HashSet<Tuple>,
    /// Tuples from the previous iteration (for semi-naive).
    recent: HashSet<Tuple>,
    /// Number of columns.
    arity: usize,
}

impl RelationStorage {
    /// Create a new relation with the given arity.
    pub fn new(arity: usize) -> Self {
        Self {
            full: HashSet::new(),
            delta: HashSet::new(),
            recent: HashSet::new(),
            arity,
        }
    }

    /// Get the arity (number of columns).
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Check if the relation is empty.
    pub fn is_empty(&self) -> bool {
        self.full.is_empty()
    }

    /// Get the number of tuples.
    pub fn len(&self) -> usize {
        self.full.len()
    }

    /// Insert a tuple. Returns true if it was new.
    pub fn insert(&mut self, tuple: Tuple) -> bool {
        debug_assert_eq!(tuple.len(), self.arity, "tuple arity mismatch");
        if self.full.insert(tuple.clone()) {
            self.delta.insert(tuple);
            true
        } else {
            false
        }
    }

    /// Check if a tuple exists.
    pub fn contains(&self, tuple: &Tuple) -> bool {
        self.full.contains(tuple)
    }

    /// Iterate over all tuples.
    pub fn iter(&self) -> impl Iterator<Item = &Tuple> {
        self.full.iter()
    }

    /// Iterate over recent tuples (from last iteration).
    pub fn iter_recent(&self) -> impl Iterator<Item = &Tuple> {
        self.recent.iter()
    }

    /// Iterate over all tuples (for rules that don't use semi-naive).
    pub fn iter_full(&self) -> impl Iterator<Item = &Tuple> {
        self.full.iter()
    }

    /// Check if there are new tuples in the delta.
    pub fn has_delta(&self) -> bool {
        !self.delta.is_empty()
    }

    /// Move delta to recent, clear delta. Returns true if there were changes.
    pub fn advance(&mut self) -> bool {
        let had_delta = !self.delta.is_empty();
        self.recent = std::mem::take(&mut self.delta);
        had_delta
    }

    /// Clear the recent set (after all rules have been evaluated).
    pub fn clear_recent(&mut self) {
        self.recent.clear();
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
}
