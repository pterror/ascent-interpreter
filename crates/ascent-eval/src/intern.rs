//! String interning via a thread-local `StringTable`.
//!
//! `Value::Interned(Rc<dyn InternTable>, u32)` is the universal representation
//! for interned values. Strings use this module's `StringTable`; other types
//! use `specialized::HashInternTable`.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::value::{InternTable, Value};

/// Thread-local intern table for strings.
///
/// Uses `Box::leak` for zero-copy `&'static str` resolution, matching the
/// original `Interner` design. `pack` verifies table identity via pointer
/// comparison so values from different tables are never confused.
///
/// **Memory lifecycle:** Interned strings are intentionally leaked via
/// `Box::leak` and are never freed. This is acceptable for interpreter/REPL
/// use where the string set is bounded by program size, but callers should
/// be aware that repeated interning of dynamic strings will grow memory
/// monotonically.
pub struct StringTable {
    to_id: RefCell<FxHashMap<&'static str, u32>>,
    to_val: RefCell<Vec<&'static str>>,
}

impl StringTable {
    fn new() -> Self {
        Self {
            to_id: RefCell::new(FxHashMap::default()),
            to_val: RefCell::new(Vec::new()),
        }
    }

    /// Intern a string and return its u32 id.
    pub fn intern(&self, s: &str) -> u32 {
        if let Some(&id) = self.to_id.borrow().get(s) {
            return id;
        }
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        let id = self.to_val.borrow().len() as u32;
        self.to_val.borrow_mut().push(leaked);
        self.to_id.borrow_mut().insert(leaked, id);
        id
    }

    /// Resolve a u32 id back to its `&'static str`. Panics on invalid id.
    pub fn resolve(&self, id: u32) -> &'static str {
        self.to_val.borrow().get(id as usize).copied().unwrap_or_else(|| {
            panic!(
                "invalid intern id {id}: StringTable contains {} entries",
                self.to_val.borrow().len()
            )
        })
    }
}

impl InternTable for StringTable {
    fn pack(&self, val: &Value) -> Option<u32> {
        if let Value::Interned(table, id) = val {
            // Only accept values that belong to this exact table instance.
            let self_ptr = self as *const StringTable as *const ();
            let table_ptr = Rc::as_ptr(table) as *const ();
            if std::ptr::eq(self_ptr, table_ptr) {
                return Some(*id);
            }
        }
        None
    }

    fn fmt_display(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.resolve(id))
    }

    fn fmt_debug(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.resolve(id))
    }

    fn cmp_ids(&self, a: u32, b: u32) -> Ordering {
        self.resolve(a).cmp(self.resolve(b))
    }

    fn resolve_str(&self, id: u32) -> Option<&str> {
        Some(self.resolve(id))
    }

    fn type_name(&self) -> &str {
        "String"
    }
}

thread_local! {
    static STRING_TABLE: Rc<StringTable> = Rc::new(StringTable::new());
}

/// Return a clone of the thread-local `StringTable` `Rc`.
/// Cheap — just increments the reference count.
pub fn string_table() -> Rc<StringTable> {
    STRING_TABLE.with(Rc::clone)
}

/// Construct a `Value::Interned` for a string, interning it in the
/// thread-local `StringTable`.
///
/// The string is leaked via `Box::leak` and will not be freed for the
/// lifetime of the thread. See [`StringTable`] for details.
pub fn string_value(s: &str) -> Value {
    STRING_TABLE.with(|table| {
        let id = table.intern(s);
        Value::Interned(Rc::clone(table) as Rc<dyn InternTable>, id)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_returns_same_id() {
        let table = StringTable::new();
        let a = table.intern("hello");
        let b = table.intern("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn different_strings_different_ids() {
        let table = StringTable::new();
        let a = table.intern("foo");
        let b = table.intern("bar");
        assert_ne!(a, b);
    }

    #[test]
    fn resolve_round_trips() {
        let table = StringTable::new();
        let id = table.intern("world");
        assert_eq!(table.resolve(id), "world");
    }

    #[test]
    fn string_value_display_and_debug() {
        let v = Value::string("test");
        assert_eq!(format!("{v}"), "test");
        assert_eq!(format!("{v:?}"), "\"test\"");
    }

    #[test]
    fn string_value_equality() {
        let a = Value::string("hello");
        let b = Value::string("hello");
        let c = Value::string("world");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
