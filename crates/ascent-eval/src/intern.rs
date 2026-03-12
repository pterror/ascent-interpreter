//! Global string interner for O(1) equality, hashing, and cloning of string values.
//!
//! All `Value::String` instances store a `SymbolId` (a `u32` index) instead of
//! heap-allocated string data. The interner is thread-local and leak-based:
//! interned strings live for the lifetime of the thread, so `resolve()` returns
//! `&'static str` with no borrow gymnastics.

use std::fmt;

use rustc_hash::FxHashMap;

/// An interned string identifier. Copy-cheap, with O(1) equality and hashing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub(crate) u32);

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", resolve(*self))
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", resolve(*self))
    }
}

struct Interner {
    strings: Vec<&'static str>,
    ids: FxHashMap<&'static str, u32>,
}

thread_local! {
    static INTERNER: std::cell::RefCell<Interner> = std::cell::RefCell::new(Interner {
        strings: Vec::new(),
        ids: FxHashMap::default(),
    });
}

/// Intern a string, returning its `SymbolId`. If the string was already
/// interned, returns the existing id (O(1) amortized hash lookup).
pub fn intern(s: &str) -> SymbolId {
    INTERNER.with(|cell| {
        let mut inner = cell.borrow_mut();
        if let Some(&id) = inner.ids.get(s) {
            return SymbolId(id);
        }
        let id = inner.strings.len() as u32;
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        inner.strings.push(leaked);
        inner.ids.insert(leaked, id);
        SymbolId(id)
    })
}

/// Resolve a `SymbolId` back to its string. Panics if the id is invalid.
pub fn resolve(id: SymbolId) -> &'static str {
    INTERNER.with(|cell| cell.borrow().strings[id.0 as usize])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_id() {
        let a = intern("hello");
        let b = intern("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn different_strings_different_ids() {
        let a = intern("foo");
        let b = intern("bar");
        assert_ne!(a, b);
    }

    #[test]
    fn resolve_round_trips() {
        let id = intern("world");
        assert_eq!(resolve(id), "world");
    }

    #[test]
    fn display_and_debug() {
        let id = intern("test");
        assert_eq!(format!("{id}"), "test");
        assert_eq!(format!("{id:?}"), "\"test\"");
    }
}
