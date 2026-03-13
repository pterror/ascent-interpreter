//! Typed helpers for the packed u32 JIT compiler.
//!
//! Only available when both `jit` and `specialized` features are enabled.
//!
//! These work directly on PackedStorage's `packed_data: Vec<u32>` buffer,
//! bypassing the Value enum entirely. All bindings are flat `u32` arrays.
//! This eliminates Value cloning, Option<Value> overhead, and enum dispatch.

use crate::specialized::PackedStorage;

/// Result of a packed index lookup — pointer + length to a `&[usize]`.
#[repr(C)]
pub struct PackedLookupResult {
    pub ptr: *const usize,
    pub len: usize,
}

/// Runtime context for typed packed JIT functions.
///
/// All values in the `bindings` array are raw u32 (intern IDs for strings,
/// bit-cast i32 for signed ints, 0/1 for bools).
///
/// repr(C) layout on 64-bit:
///   rels     (ptr)  @ offset  0
///   rels_len (u32)  @ offset  8
///   _pad     (u32)  @ offset 12
///   bindings (ptr)  @ offset 16
///   results  (ptr)  @ offset 24
#[repr(C)]
pub struct PackedJitContext {
    /// Array of PackedStorage pointers, one per clause relation.
    pub rels: *const *const PackedStorage,
    pub rels_len: u32,
    pub _pad: u32,
    /// Flat u32 binding scratch: `bindings[var_id]` = current packed value.
    pub bindings: *mut u32,
    /// Results: (head_idx, packed_tuple).
    pub results: *mut Vec<(usize, Vec<u32>)>,
}

/// Type alias for packed JIT function pointer.
pub type PackedJitFn = unsafe extern "C" fn(*mut PackedJitContext);

// ─── Compile-time layout verification ───────────────────────────────

const _: () = {
    assert!(std::mem::offset_of!(PackedJitContext, rels) == 0);
    assert!(std::mem::offset_of!(PackedJitContext, rels_len) == 8);
    assert!(std::mem::offset_of!(PackedJitContext, bindings) == 16);
    assert!(std::mem::offset_of!(PackedJitContext, results) == 24);
};

// ─── Helpers called by JIT-generated code ───────────────────────────

/// Return the number of tuples (full count or recent count).
///
/// # Safety
/// `rel` must point to a valid PackedStorage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_count(rel: *const PackedStorage, use_recent: u32) -> usize {
    let rel = unsafe { &*rel };
    if use_recent != 0 {
        rel.recent.len()
    } else {
        rel.count
    }
}

/// Return pointer to the start of the packed u32 data buffer.
///
/// # Safety
/// `rel` must point to a valid PackedStorage with non-empty packed_data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_data_ptr(rel: *const PackedStorage) -> *const u32 {
    let rel = unsafe { &*rel };
    rel.packed_data.as_ptr()
}

/// For a recent scan, return the actual tuple index at sequential position `seq_idx`.
///
/// Maps `seq_idx` (0..recent.len()) to `rel.recent[seq_idx]`.
///
/// # Safety
/// `rel` must point to a valid PackedStorage. `seq_idx` must be < recent.len().
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_recent_idx(rel: *const PackedStorage, seq_idx: usize) -> usize {
    let rel = unsafe { &*rel };
    rel.recent[seq_idx]
}

/// Look up tuples matching `key` in column `col`.
///
/// Returns a pointer+length pair pointing into the relation's index Vec.
/// The slice is valid as long as the PackedStorage is not mutated.
///
/// # Safety
/// `rel` must point to a valid PackedStorage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_lookup(
    rel: *const PackedStorage,
    col: u32,
    key: u32,
    use_recent: u32,
) -> PackedLookupResult {
    let rel = unsafe { &*rel };
    let col = col as usize;
    let indices = rel.lookup_packed(col, key, use_recent != 0);
    PackedLookupResult {
        ptr: indices.as_ptr(),
        len: indices.len(),
    }
}

/// Push a packed tuple into the results accumulator.
///
/// Copies `arity` u32 words from `tuple` into a new Vec<u32> and pushes
/// `(head_idx, tuple_vec)` onto results.
///
/// # Safety
/// `results` must point to a valid Vec. `tuple` must point to `arity` valid u32s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_push_result(
    results: *mut Vec<(usize, Vec<u32>)>,
    head_idx: usize,
    tuple: *const u32,
    arity: u32,
) {
    let results = unsafe { &mut *results };
    let slice = unsafe { std::slice::from_raw_parts(tuple, arity as usize) };
    results.push((head_idx, slice.to_vec()));
}

// ─── Stratum meta-function flush+advance helper ──────────────────────

/// Per-rule info needed to flush JIT results into relations.
pub struct RuleFlushInfo {
    /// Pointer to the rule's results buffer (`Vec<(usize, Vec<u32>)>`).
    pub results: *mut Vec<(usize, Vec<u32>)>,
    /// Backing storage for head_rels slice.
    /// head_rels[i] = *mut PackedStorage for rule.heads[i].
    pub head_rels: Vec<*mut PackedStorage>,
}

/// Runtime state for the stratum meta-function flush+advance operation.
pub struct StratumFlusher {
    /// Per-rule flush info (one entry per rule in the stratum).
    pub rules: Vec<RuleFlushInfo>,
    /// All packed relations to advance each iteration.
    pub all_packed_rels: Vec<*mut PackedStorage>,
}

/// Flush all pending rule results into their target relations, then advance
/// all packed relations in the stratum.
///
/// Returns 1 if any relation gained new tuples this iteration, 0 otherwise.
///
/// # Safety
/// All pointers in `flusher` must be valid for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_stratum_flush_advance(flusher: *mut StratumFlusher) -> u8 {
    let flusher = unsafe { &mut *flusher };

    for rule_info in &mut flusher.rules {
        let results = unsafe { &mut *rule_info.results };
        for (head_idx, packed_vals) in results.drain(..) {
            if let Some(&rel_ptr) = rule_info.head_rels.get(head_idx) {
                let rel = unsafe { &mut *rel_ptr };
                rel.insert_packed_raw(&packed_vals);
            }
        }
    }

    let mut changed = false;
    for &rel_ptr in &flusher.all_packed_rels {
        let rel = unsafe { &mut *rel_ptr };
        if rel.advance() {
            changed = true;
        }
    }

    changed as u8
}

/// Context struct passed to the stratum meta-function.
///
/// Layout (repr C, 64-bit):
///   offset  0: full_fns    *const PackedJitFn           (8 bytes)
///   offset  8: full_ctxs   *const *mut PackedJitContext  (8 bytes)
///   offset 16: num_full    u32                           (4 bytes)
///   offset 20: num_recent  u32                           (4 bytes)
///   offset 24: recent_fns  *const PackedJitFn            (8 bytes)
///   offset 32: recent_ctxs *const *mut PackedJitContext  (8 bytes)
///   offset 40: flusher     *mut StratumFlusher            (8 bytes)
#[repr(C)]
pub struct StratumMetaCtx {
    pub full_fns: *const PackedJitFn,
    pub full_ctxs: *const *mut PackedJitContext,
    pub num_full: u32,
    pub num_recent: u32,
    pub recent_fns: *const PackedJitFn,
    pub recent_ctxs: *const *mut PackedJitContext,
    pub flusher: *mut StratumFlusher,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(StratumMetaCtx, full_fns) == 0);
    assert!(std::mem::offset_of!(StratumMetaCtx, full_ctxs) == 8);
    assert!(std::mem::offset_of!(StratumMetaCtx, num_full) == 16);
    assert!(std::mem::offset_of!(StratumMetaCtx, num_recent) == 20);
    assert!(std::mem::offset_of!(StratumMetaCtx, recent_fns) == 24);
    assert!(std::mem::offset_of!(StratumMetaCtx, recent_ctxs) == 32);
    assert!(std::mem::offset_of!(StratumMetaCtx, flusher) == 40);
};

/// Type alias for the stratum meta-function pointer.
pub type StratumMetaFn = unsafe extern "C" fn(*mut StratumMetaCtx);
