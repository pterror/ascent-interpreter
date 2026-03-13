//! `extern "C"` trampoline functions for JIT-generated code.
//!
//! JIT code handles control flow (loops, branches) but calls these helpers
//! for all Rust-side operations: relation access, Value manipulation, and
//! head emission. This keeps the JIT IR simple and avoids duplicating
//! complex Rust logic in Cranelift.

use crate::compiled::{CCondition, CExpr, CHeadClause, eval_cexpr};
use crate::eval::{Bindings, TypeRegistry, VarInterner};
use crate::relation::Relation;
use crate::value::{Tuple, Value};

/// Result of a relation index lookup — pointer + length to a `&[usize]`.
#[repr(C)]
pub struct LookupResult {
    pub ptr: *const usize,
    pub len: usize,
}

/// Result of expression evaluation — value pointer + success flag.
#[repr(C)]
pub struct EvalResult {
    /// Pointer to heap-allocated Value (caller must free), or null on failure.
    pub ptr: *mut Value,
    pub ok: bool,
}

/// Runtime context passed from Rust to JIT-generated code.
///
/// All pointers must remain valid for the duration of the JIT function call.
/// The JIT function is called synchronously within `derive_tuples`, so
/// lifetimes are tied to the engine borrow.
#[repr(C)]
pub struct JitContext {
    /// Array of relation pointers, pre-resolved by relation name order in the rule.
    pub rels: *const *const Relation,
    pub rels_len: u32,
    /// Raw pointer to the start of `Bindings.slots` Vec data.
    pub bindings: *mut Bindings,
    /// Pointer to results accumulator.
    pub results: *mut Vec<(usize, Tuple)>,
    /// Array of head clause pointers.
    pub heads: *const *const CHeadClause,
    pub heads_len: u32,
    /// Type registry for expression evaluation.
    pub registry: *const TypeRegistry,
    /// Variable interner for expression evaluation.
    pub interner: *const VarInterner,
}

// ─── Relation operations ────────────────────────────────────────────

/// Look up tuples matching a value in a column.
/// `use_recent`: 0 = full index, 1 = recent-only index.
///
/// # Safety
/// `rel` must point to a valid Relation. `val` must point to a valid Value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_lookup(
    rel: *const Relation,
    col: u32,
    val: *const Value,
    use_recent: u32,
) -> LookupResult {
    let rel = unsafe { &*rel };
    let val = unsafe { &*val };
    let indices = if use_recent != 0 {
        rel.lookup_recent(col as usize, val)
    } else {
        rel.lookup(col as usize, val)
    };
    LookupResult {
        ptr: indices.as_ptr(),
        len: indices.len(),
    }
}

/// Get a tuple by index, returning pointer to the first Value in the tuple slice.
///
/// # Safety
/// `rel` must point to a valid Relation. `tuple_idx` must be in bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_get_tuple(rel: *const Relation, tuple_idx: usize) -> *const Value {
    let rel = unsafe { &*rel };
    rel.get(tuple_idx).as_ptr()
}

/// Get the count of tuples (full or recent).
/// `use_recent`: 0 = total count, 1 = recent count.
///
/// # Safety
/// `rel` must point to a valid Relation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_count(rel: *const Relation, use_recent: u32) -> usize {
    let rel = unsafe { &*rel };
    if use_recent != 0 {
        rel.recent_indices().len()
    } else {
        rel.tuple_count()
    }
}

/// Get a tuple by sequential index during full scan.
/// For full scan: index 0..count maps to tuples 0..count.
/// For recent scan: index 0..recent_count maps to recent tuple indices.
///
/// # Safety
/// `rel` must point to a valid Relation.
/// For full scan: `seq_idx` must be < tuple_count().
/// For recent scan: `seq_idx` must be < recent_indices().len().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_tuple_at(
    rel: *const Relation,
    seq_idx: usize,
    use_recent: u32,
) -> *const Value {
    let rel = unsafe { &*rel };
    if use_recent != 0 {
        let idx = rel.recent_indices()[seq_idx];
        rel.get(idx).as_ptr()
    } else {
        rel.get(seq_idx).as_ptr()
    }
}

/// Check if a relation contains a tuple.
///
/// # Safety
/// `rel` must point to a valid Relation.
/// `tuple_ptr` must point to `arity` contiguous Values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_contains(
    rel: *const Relation,
    tuple_ptr: *const Value,
    arity: u32,
) -> bool {
    let rel = unsafe { &*rel };
    let tuple = unsafe { std::slice::from_raw_parts(tuple_ptr, arity as usize) };
    rel.contains(tuple)
}

// ─── Value operations ───────────────────────────────────────────────

/// Clone a Value from src to dst.
///
/// # Safety
/// `src` must point to a valid, initialized Value.
/// `dst` must point to writable memory of at least VALUE_SIZE bytes.
/// `dst` must NOT contain a live Value (would leak). Caller must clear first.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_value_clone(src: *const Value, dst: *mut Value) {
    let src = unsafe { &*src };
    let cloned = src.clone();
    unsafe { std::ptr::write(dst, cloned) };
}

/// Compare two Values for equality.
///
/// # Safety
/// Both pointers must point to valid, initialized Values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_value_eq(a: *const Value, b: *const Value) -> bool {
    let a = unsafe { &*a };
    let b = unsafe { &*b };
    a == b
}

/// Clear a binding slot: drop the Option<Value> and set to None.
///
/// # Safety
/// `slot_ptr` must point to a valid `Option<Value>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_slot_clear(slot_ptr: *mut Option<Value>) {
    unsafe {
        std::ptr::drop_in_place(slot_ptr);
        std::ptr::write(slot_ptr, None);
    }
}

/// Write a cloned Value into a binding slot (set to Some(value.clone())).
///
/// # Safety
/// `slot_ptr` must point to a valid `Option<Value>` that is currently `None`.
/// `value_ptr` must point to a valid Value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_slot_set(slot_ptr: *mut Option<Value>, value_ptr: *const Value) {
    let value = unsafe { &*value_ptr };
    unsafe {
        std::ptr::write(slot_ptr, Some(value.clone()));
    }
}

/// Read a binding slot, returning pointer to the Value if Some, or null if None.
///
/// # Safety
/// `slot_ptr` must point to a valid `Option<Value>`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_slot_get(slot_ptr: *const Option<Value>) -> *const Value {
    let slot = unsafe { &*slot_ptr };
    match slot {
        Some(v) => v as *const Value,
        None => std::ptr::null(),
    }
}

// ─── Condition evaluation ───────────────────────────────────────────

/// Evaluate a CCondition against current bindings.
/// Only handles `CCondition::If` — IfLet/Let are not JIT-eligible.
///
/// # Safety
/// All pointers must be valid. `cond` must point to a CCondition::If.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_eval_condition(
    cond: *const CCondition,
    bindings: *mut Bindings,
    registry: *const TypeRegistry,
    interner: *const VarInterner,
) -> bool {
    let cond = unsafe { &*cond };
    let bindings = unsafe { &*bindings };
    let registry = unsafe { &*registry };
    let interner = unsafe { &*interner };
    match cond {
        CCondition::If(expr) => eval_cexpr(expr, bindings, Some(registry), interner)
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        // IfLet/Let conditions should not reach here (not JIT-eligible)
        _ => false,
    }
}

// ─── Head emission ──────────────────────────────────────────────────

/// Evaluate and emit a head tuple into the results vector.
///
/// # Safety
/// All pointers must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_emit_head(
    head: *const CHeadClause,
    head_idx: usize,
    bindings: *mut Bindings,
    results: *mut Vec<(usize, Tuple)>,
    registry: *const TypeRegistry,
    interner: *const VarInterner,
) {
    let head = unsafe { &*head };
    let bindings = unsafe { &*bindings };
    let results = unsafe { &mut *results };
    let registry = unsafe { &*registry };
    let interner = unsafe { &*interner };

    let mut tuple = Vec::with_capacity(head.args.len());
    for arg in &head.args {
        if let Some(value) = eval_cexpr(arg, bindings, Some(registry), interner) {
            tuple.push(value);
        } else {
            return;
        }
    }
    results.push((head_idx, tuple));
}

/// Emit all heads for a rule.
///
/// # Safety
/// `ctx` must point to a valid JitContext.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_emit_all_heads(ctx: *mut JitContext) {
    let ctx = unsafe { &mut *ctx };
    let heads = unsafe { std::slice::from_raw_parts(ctx.heads, ctx.heads_len as usize) };
    let bindings = unsafe { &*ctx.bindings };
    let results = unsafe { &mut *ctx.results };
    let registry = unsafe { &*ctx.registry };
    let interner = unsafe { &*ctx.interner };

    for (head_idx, &head_ptr) in heads.iter().enumerate() {
        let head = unsafe { &*head_ptr };
        let mut tuple = Vec::with_capacity(head.args.len());
        let mut ok = true;
        for arg in &head.args {
            if let Some(value) = eval_cexpr(arg, bindings, Some(registry), interner) {
                tuple.push(value);
            } else {
                ok = false;
                break;
            }
        }
        if ok {
            results.push((head_idx, tuple));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::RelationStorage;

    #[test]
    fn test_jit_value_eq_helper() {
        let a = Value::I32(42);
        let b = Value::I32(42);
        let c = Value::I32(99);
        unsafe {
            assert!(jit_value_eq(&a, &b));
            assert!(!jit_value_eq(&a, &c));
        }
    }

    #[test]
    fn test_jit_value_clone_helper() {
        let src = Value::I32(42);
        let mut dst = std::mem::MaybeUninit::<Value>::uninit();
        unsafe {
            jit_value_clone(&src, dst.as_mut_ptr());
            let dst = dst.assume_init();
            assert_eq!(dst, Value::I32(42));
        }
    }

    #[test]
    fn test_jit_slot_ops() {
        let mut slot: Option<Value> = None;
        let val = Value::I32(42);

        // slot_get on None returns null
        unsafe {
            assert!(jit_slot_get(&slot).is_null());

            // slot_set writes Some
            jit_slot_set(&mut slot, &val);
            assert_eq!(slot, Some(Value::I32(42)));

            // slot_get on Some returns pointer to value
            let ptr = jit_slot_get(&slot);
            assert!(!ptr.is_null());
            assert_eq!(&*ptr, &Value::I32(42));

            // slot_clear drops and sets to None
            jit_slot_clear(&mut slot);
            assert_eq!(slot, None);
        }
    }

    #[test]
    fn test_jit_rel_lookup_helper() {
        let mut rel = RelationStorage::new(2);
        rel.insert(vec![Value::I32(1), Value::I32(10)]);
        rel.insert(vec![Value::I32(1), Value::I32(20)]);
        rel.insert(vec![Value::I32(2), Value::I32(30)]);

        let rel = Relation::Generic(rel);
        let key = Value::I32(1);
        unsafe {
            let result = jit_rel_lookup(&rel, 0, &key, 0);
            let indices = std::slice::from_raw_parts(result.ptr, result.len);
            assert_eq!(indices.len(), 2);
        }
    }

    #[test]
    fn test_jit_rel_count_and_tuple_at() {
        let mut rel = RelationStorage::new(2);
        rel.insert(vec![Value::I32(1), Value::I32(10)]);
        rel.insert(vec![Value::I32(2), Value::I32(20)]);
        let rel = Relation::Generic(rel);

        unsafe {
            assert_eq!(jit_rel_count(&rel, 0), 2);

            let tuple = jit_rel_tuple_at(&rel, 0, 0);
            assert_eq!(&*tuple, &Value::I32(1));
            assert_eq!(&*tuple.add(1), &Value::I32(10));
        }
    }

    #[test]
    fn test_jit_rel_contains_helper() {
        let mut rel = RelationStorage::new(2);
        rel.insert(vec![Value::I32(1), Value::I32(10)]);
        let rel = Relation::Generic(rel);

        let present = [Value::I32(1), Value::I32(10)];
        let absent = [Value::I32(1), Value::I32(99)];
        unsafe {
            assert!(jit_rel_contains(&rel, present.as_ptr(), 2));
            assert!(!jit_rel_contains(&rel, absent.as_ptr(), 2));
        }
    }
}
