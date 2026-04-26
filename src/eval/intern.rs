//! String interning via a global `StringTable`.
//!
//! `Value::Interned(Arc<dyn InternTable>, u32)` is the universal representation
//! for interned values. Strings use this module's `StringTable`; other types
//! use `specialized::HashInternTable`.

use std::cmp::Ordering;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

use rustc_hash::FxHashMap;

use crate::eval::value::{InternTable, Value};

/// Global intern table for strings.
///
/// Uses `Box::leak` for zero-copy `&'static str` resolution. All threads share
/// one table so `Value::Interned` ids are consistent across thread boundaries.
///
/// **Memory lifecycle:** Interned strings are intentionally leaked and never freed.
pub struct StringTable {
    to_id: RwLock<FxHashMap<&'static str, u32>>,
    to_val: RwLock<Vec<&'static str>>,
}

impl StringTable {
    fn new() -> Self {
        Self {
            to_id: RwLock::new(FxHashMap::default()),
            to_val: RwLock::new(Vec::new()),
        }
    }

    /// Intern a string and return its u32 id.
    pub fn intern(&self, s: &str) -> u32 {
        // Fast path: already interned
        if let Some(&id) = self.to_id.read().unwrap().get(s) {
            return id;
        }
        // Slow path: acquire write locks and insert
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        let mut id_guard = self.to_id.write().unwrap();
        // Re-check under write lock (another thread may have won the race)
        if let Some(&id) = id_guard.get(leaked) {
            return id;
        }
        let mut val_guard = self.to_val.write().unwrap();
        let id = val_guard.len() as u32;
        val_guard.push(leaked);
        id_guard.insert(leaked, id);
        id
    }

    /// Resolve a u32 id back to its `&'static str`. Panics on invalid id.
    pub fn resolve(&self, id: u32) -> &'static str {
        self.to_val
            .read()
            .unwrap()
            .get(id as usize)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "invalid intern id {id}: StringTable contains {} entries",
                    self.to_val.read().unwrap().len()
                )
            })
    }
}

impl InternTable for StringTable {
    fn pack(&self, val: &Value) -> Option<u32> {
        if let Value::Interned(table, id) = val {
            let self_ptr = self as *const StringTable as *const ();
            let table_ptr = Arc::as_ptr(table) as *const ();
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

static GLOBAL_STRING_TABLE: OnceLock<Arc<StringTable>> = OnceLock::new();

fn global_table() -> Arc<StringTable> {
    GLOBAL_STRING_TABLE
        .get_or_init(|| Arc::new(StringTable::new()))
        .clone()
}

/// Return a clone of the global `StringTable` `Arc`.
pub fn string_table() -> Arc<StringTable> {
    global_table()
}

/// Construct a `Value::Interned` for a string, interning it in the global `StringTable`.
pub fn string_value(s: &str) -> Value {
    let table = global_table();
    let id = table.intern(s);
    Value::Interned(table as Arc<dyn InternTable>, id)
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
