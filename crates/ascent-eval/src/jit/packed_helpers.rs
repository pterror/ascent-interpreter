//! Typed helpers for the packed u32 JIT compiler.
//!
//! Only available when both `jit` and `specialized` features are enabled.
//!
//! These work directly on PackedStorage's `packed_data: Vec<u32>` buffer,
//! bypassing the Value enum entirely. All bindings are flat `u32` arrays.
//! This eliminates Value cloning, Option<Value> overhead, and enum dispatch.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr;

use crate::jit::storage;
use crate::jit_index::{JitDedupHandle, JitLookupHandle};
use crate::specialized::PackedStorage;

// ─── JIT head write buffer ────────────────────────────────────────────────

/// Pre-allocated write buffer for a single head relation.
///
/// The JIT writes new tuples inline (after the inline dedup probe); after each
/// stratum iteration, Rust bulk-inserts them into `PackedStorage`.
///
/// `repr(C)` layout on 64-bit (24 bytes):
///   offset  0: data  *mut u32  — write destination (row-major, stride=arity)
///   offset  8: len   u32       — current tuple count (number written so far)
///   offset 12: cap   u32       — capacity in tuples
///   offset 16: arity u32       — arity (compile-time known, stored for Rust flush)
///   offset 20: _pad  u32
#[repr(C)]
pub struct JitHeadBuf {
    /// Pointer to the flat tuple data (row-major, stride = arity u32s per tuple).
    pub data: *mut u32,
    /// Number of tuples written so far.
    pub len: u32,
    /// Capacity in tuples.
    pub cap: u32,
    /// Arity (number of u32 words per tuple).
    pub arity: u32,
    pub _pad: u32,
}

unsafe impl Send for JitHeadBuf {}
unsafe impl Sync for JitHeadBuf {}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(JitHeadBuf, data) == 0);
    assert!(std::mem::offset_of!(JitHeadBuf, len) == 8);
    assert!(std::mem::offset_of!(JitHeadBuf, cap) == 12);
    assert!(std::mem::offset_of!(JitHeadBuf, arity) == 16);
    assert!(std::mem::size_of::<JitHeadBuf>() == 24);
};

/// Called when `JitHeadBuf` is full: doubles capacity and appends the tuple.
///
/// # Safety
/// `buf` must be a valid `*mut JitHeadBuf`. `tuple` must point to `arity` valid u32s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_head_buf_grow_and_insert(
    buf: *mut JitHeadBuf,
    tuple: *const u32,
    arity: u32,
) {
    let buf = unsafe { &mut *buf };
    let new_cap = (buf.cap as usize * 2).max(16);
    let layout = Layout::array::<u32>(new_cap * arity as usize).expect("JitHeadBuf layout");
    let new_data = unsafe { alloc_zeroed(layout) } as *mut u32;
    // Copy old data and free old allocation.
    if !buf.data.is_null() && buf.len > 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                buf.data,
                new_data,
                buf.len as usize * arity as usize,
            );
        }
        let old_layout =
            Layout::array::<u32>(buf.cap as usize * arity as usize).expect("JitHeadBuf layout");
        unsafe { dealloc(buf.data as *mut u8, old_layout) };
    }
    buf.data = new_data;
    buf.cap = new_cap as u32;
    // Append the new tuple.
    let dst = unsafe { new_data.add(buf.len as usize * arity as usize) };
    unsafe { ptr::copy_nonoverlapping(tuple, dst, arity as usize) };
    buf.len += 1;
}

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

/// Return a pointer to the start of the recent-index array (`rel.recent`).
///
/// The array contains `usize` elements (pointer-sized); element `i` is the
/// tuple index of the i-th recent tuple.  Callers must not mutate through
/// this pointer.  The pointer is stable for the lifetime of the current
/// semi-naive iteration (no `advance()` call between fetch and use).
///
/// # Safety
/// `rel` must point to a valid PackedStorage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_recent_ptr(rel: *const PackedStorage) -> *const usize {
    let rel = unsafe { &*rel };
    rel.recent.as_ptr()
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

// ─── Stage 3: direct-insert rule variants ───────────────────────────

/// Runtime context for Stage 3 packed JIT rule variants.
///
/// Head relations are written to directly — no results buffer needed.
/// Bindings are held in Cranelift Variables (register-allocated), not in a heap array.
///
/// repr(C) layout on 64-bit:
///   rels               (ptr)  @ offset  0
///   rels_len           (u32)  @ offset  8
///   _pad               (u32)  @ offset 12
///   head_rels          (ptr)  @ offset 16
///   lookup_handles     (ptr)  @ offset 24  ← flat array of JitLookupHandle
///   head_dedup_handles (ptr)  @ offset 32  ← one *mut JitDedupHandle per head
#[repr(C)]
pub struct PackedJitContextV3 {
    /// Array of PackedStorage pointers, one per clause relation.
    pub rels: *const *const PackedStorage,
    pub rels_len: u32,
    pub _pad: u32,
    /// Array of *mut PackedStorage, one per head relation (direct insert target).
    pub head_rels: *const *mut PackedStorage,
    /// Pointer to flat array of JitLookupHandle, indexed by `clause_offset * 2 + use_recent`.
    pub lookup_handles: *const JitLookupHandle,
    /// Array of *mut JitDedupHandle, one per head relation.
    /// Points into the `jit_dedup.handle` field of each head's PackedStorage.
    /// Used by the JIT to probe the dedup snapshot before calling packed_try_insert.
    pub head_dedup_handles: *const *mut JitDedupHandle,
}

pub type PackedJitFnV3 = unsafe extern "C" fn(*mut PackedJitContextV3);

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(PackedJitContextV3, rels) == 0);
    assert!(std::mem::offset_of!(PackedJitContextV3, rels_len) == 8);
    assert!(std::mem::offset_of!(PackedJitContextV3, head_rels) == 16);
    assert!(std::mem::offset_of!(PackedJitContextV3, lookup_handles) == 24);
    assert!(std::mem::offset_of!(PackedJitContextV3, head_dedup_handles) == 32);
};

/// Context for Stage 3 stratum meta-function (direct-insert variant).
///
/// Layout (repr C, 64-bit):
///   offset  0: full_fns    *const PackedJitFnV3           (8 bytes)
///   offset  8: full_ctxs   *const *mut PackedJitContextV3 (8 bytes)
///   offset 16: num_full    u32                            (4 bytes)
///   offset 20: num_recent  u32                            (4 bytes)
///   offset 24: recent_fns  *const PackedJitFnV3           (8 bytes)
///   offset 32: recent_ctxs *const *mut PackedJitContextV3 (8 bytes)
///   offset 40: all_rels    *const *mut PackedStorage       (8 bytes)
///   offset 48: n_all_rels  u32                            (4 bytes)
///   offset 52: _pad        u32                            (4 bytes)
#[repr(C)]
pub struct StratumStage3Ctx {
    pub full_fns: *const PackedJitFnV3,
    pub full_ctxs: *const *mut PackedJitContextV3,
    pub num_full: u32,
    pub num_recent: u32,
    pub recent_fns: *const PackedJitFnV3,
    pub recent_ctxs: *const *mut PackedJitContextV3,
    pub all_rels: *const *mut PackedStorage,
    pub n_all_rels: u32,
    pub _pad: u32,
}

pub type StratumStage3Fn = unsafe extern "C" fn(*mut StratumStage3Ctx);

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(StratumStage3Ctx, full_fns) == 0);
    assert!(std::mem::offset_of!(StratumStage3Ctx, full_ctxs) == 8);
    assert!(std::mem::offset_of!(StratumStage3Ctx, num_full) == 16);
    assert!(std::mem::offset_of!(StratumStage3Ctx, num_recent) == 20);
    assert!(std::mem::offset_of!(StratumStage3Ctx, recent_fns) == 24);
    assert!(std::mem::offset_of!(StratumStage3Ctx, recent_ctxs) == 32);
    assert!(std::mem::offset_of!(StratumStage3Ctx, all_rels) == 40);
    assert!(std::mem::offset_of!(StratumStage3Ctx, n_all_rels) == 48);
};

/// Specifies how to refresh one `JitLookupHandle` after an advance.
///
/// Layout (repr C, 16 bytes):
///   offset  0: rel        *const PackedStorage  (8 bytes)
///   offset  8: col        u32                   (4 bytes)
///   offset 12: use_recent u32                   (4 bytes — non-zero means recent)
#[repr(C)]
pub struct LookupSpec {
    pub rel: *const PackedStorage,
    pub col: u32,
    pub use_recent: u32,
}

unsafe impl Send for LookupSpec {}
unsafe impl Sync for LookupSpec {}

const _: () = {
    assert!(std::mem::size_of::<LookupSpec>() == 16);
    assert!(std::mem::offset_of!(LookupSpec, rel) == 0);
    assert!(std::mem::offset_of!(LookupSpec, col) == 8);
    assert!(std::mem::offset_of!(LookupSpec, use_recent) == 12);
};

/// Context for Stage 4 stratum function: rule bodies inlined, no fn-ptr arrays.
///
/// Layout (repr C, 64-bit):
///   offset  0: rule_ctxs       *const *mut PackedJitContextV3 (8 bytes)
///   offset  8: num_rules       u32                            (4 bytes)
///   offset 12: _pad            u32                            (4 bytes)
///   offset 16: all_rels        *const *mut PackedStorage       (8 bytes)
///   offset 24: n_all_rels      u32                            (4 bytes)
///   offset 28: _pad2           u32                            (4 bytes)
///   offset 32: handles_buf     *mut JitLookupHandle            (8 bytes)
///   offset 40: lookup_specs    *const LookupSpec              (8 bytes)
///   offset 48: total_handles   u32                            (4 bytes)
///   offset 52: _pad3           u32                            (4 bytes)
///   offset 56: tuple_sets_buf  *const *const JitTupleSet      (8 bytes)
///   offset 64: head_write_bufs *const *mut JitHeadBuf         (8 bytes)
///             flat array[total_heads] — one per (rule_i, head_i) pair, in rule order.
///   offset 72: head_rel_ptrs   *const *mut PackedStorage      (8 bytes)
///             parallel to head_write_bufs; maps buf index → target PackedStorage.
///   offset 80: total_heads     u32                            (4 bytes)
///   offset 84: _pad4           u32                            (4 bytes)
#[repr(C)]
pub struct StratumStage4Ctx {
    /// Pointer to array of *mut PackedJitContextV3, one per rule.
    pub rule_ctxs: *const *mut PackedJitContextV3,
    pub num_rules: u32,
    pub _pad: u32,
    /// All packed relations for advance().
    pub all_rels: *const *mut PackedStorage,
    pub n_all_rels: u32,
    pub _pad2: u32,
    /// Flat array of all lookup handles (all rules concatenated).
    pub handles_buf: *mut JitLookupHandle,
    /// One spec per handle, parallel to handles_buf.
    pub lookup_specs: *const LookupSpec,
    /// Total number of handles (= sum of num_clauses * 2 over all rules).
    pub total_handles: u32,
    pub _pad3: u32,
    /// Flat array parallel to handles_buf; each entry is a *const JitTupleSet
    /// for the total relation of that clause, or null if not applicable.
    pub tuple_sets_buf: *const *const storage::JitTupleSet,
    /// Flat array of `*mut JitHeadBuf`, one per (rule_i, head_i) pair.
    /// Index = `rule_head_base[rule_i] + hi`.
    pub head_write_bufs: *const *mut JitHeadBuf,
    /// Flat array of `*mut PackedStorage`, parallel to `head_write_bufs`.
    /// Maps each head buf index to its target `PackedStorage`.
    pub head_rel_ptrs: *const *mut PackedStorage,
    /// Total number of (rule, head) pairs (= len of head_write_bufs / head_rel_ptrs).
    pub total_heads: u32,
    pub _pad4: u32,
}

unsafe impl Send for StratumStage4Ctx {}
unsafe impl Sync for StratumStage4Ctx {}

pub type StratumStage4Fn = unsafe extern "C" fn(*mut StratumStage4Ctx);

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::offset_of!(StratumStage4Ctx, rule_ctxs) == 0);
    assert!(std::mem::offset_of!(StratumStage4Ctx, num_rules) == 8);
    assert!(std::mem::offset_of!(StratumStage4Ctx, all_rels) == 16);
    assert!(std::mem::offset_of!(StratumStage4Ctx, n_all_rels) == 24);
    assert!(std::mem::offset_of!(StratumStage4Ctx, handles_buf) == 32);
    assert!(std::mem::offset_of!(StratumStage4Ctx, lookup_specs) == 40);
    assert!(std::mem::offset_of!(StratumStage4Ctx, total_handles) == 48);
    assert!(std::mem::offset_of!(StratumStage4Ctx, tuple_sets_buf) == 56);
    assert!(std::mem::offset_of!(StratumStage4Ctx, head_write_bufs) == 64);
    assert!(std::mem::offset_of!(StratumStage4Ctx, head_rel_ptrs) == 72);
    assert!(std::mem::offset_of!(StratumStage4Ctx, total_heads) == 80);
};

/// Insert a packed u32 tuple directly into a relation.
///
/// Returns 1 if new, 0 if duplicate. Equivalent to `insert_packed_raw` but
/// callable from JIT-generated code.
///
/// # Safety
/// `rel` must point to a valid `PackedStorage`. `tuple` must point to `arity` valid u32s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_try_insert(
    rel: *mut PackedStorage,
    tuple: *const u32,
    arity: u32,
) -> u8 {
    let rel = unsafe { &mut *rel };
    let slice = unsafe { std::slice::from_raw_parts(tuple, arity as usize) };
    rel.insert_packed_raw(slice) as u8
}

/// JIT insert that skips interpreter-only structures (indices, value_data, source_tags).
/// After JIT evaluation completes, call `rebuild_interpreter_state()` on derived relations.
///
/// # Safety
/// `rel` must point to a valid `PackedStorage`, `tuple` to `arity` valid u32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn packed_try_insert_jit(
    rel: *mut PackedStorage,
    tuple: *const u32,
    arity: u32,
) -> u8 {
    let rel = unsafe { &mut *rel };
    let slice = unsafe { std::slice::from_raw_parts(tuple, arity as usize) };
    rel.insert_packed_raw_for_jit(slice) as u8
}

/// Advance all packed relations and return 1 if any gained new tuples.
///
/// Called once per semi-naive iteration in Stage 3 (no flush step needed).
///
/// # Safety
/// `rels` must point to `n_rels` valid `*mut PackedStorage` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_stratum_advance(rels: *const *mut PackedStorage, n_rels: u32) -> u8 {
    let mut changed = false;
    for i in 0..n_rels as usize {
        let rel = unsafe { &mut **rels.add(i) };
        if rel.advance() {
            changed = true;
        }
    }
    changed as u8
}

/// Flush all head write buffers into their target `PackedStorage` relations.
///
/// Called once per semi-naive iteration in Stage 4, BEFORE `advance()`.
/// For each non-empty `JitHeadBuf`, appends each tuple directly to the target
/// `PackedStorage` without re-checking the dedup table (the JIT already did
/// the inline probe and wrote to the dedup table).  After appending, resets
/// `buf.len = 0`.
///
/// # Safety
/// `ctx` must point to a valid `StratumStage4Ctx` with all sub-pointers valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_flush_head_bufs(ctx: *mut StratumStage4Ctx) {
    let ctx = unsafe { &*ctx };
    for i in 0..ctx.total_heads as usize {
        let buf = unsafe { &mut **ctx.head_write_bufs.add(i) };
        if buf.len == 0 {
            continue;
        }
        let rel = unsafe { &mut **ctx.head_rel_ptrs.add(i) };
        let arity = buf.arity as usize;
        let data = unsafe { std::slice::from_raw_parts(buf.data, buf.len as usize * arity) };
        for t in 0..buf.len as usize {
            let packed = &data[t * arity..(t + 1) * arity];
            rel.insert_packed_raw_no_dedup(packed);
        }
        buf.len = 0;
    }
}

/// Advance all packed relations and refresh all JIT lookup handles.
///
/// Replaces `jit_stratum_advance` in Stage 4: after `advance()`, `PackedStorage`
/// rebuilds its JIT indices — this function copies the new pointers into the
/// handle array so JIT code sees fresh data on the next iteration.
///
/// Flushes head write buffers first (so newly-emitted tuples enter `delta`
/// and become `recent` for the next iteration).
///
/// Returns 1 if any relation gained new tuples, 0 otherwise.
///
/// # Safety
/// `ctx` must point to a valid `StratumStage4Ctx` with all sub-pointers valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_stratum_advance_s4(ctx: *mut StratumStage4Ctx) -> u8 {
    // Flush head write bufs into PackedStorage delta BEFORE advance.
    unsafe { jit_flush_head_bufs(ctx) };
    let ctx = unsafe { &*ctx };
    let changed = unsafe { jit_stratum_advance(ctx.all_rels, ctx.n_all_rels) };
    // Refresh handles — jit_stratum_advance already called advance() on all rels
    // which rebuilds jit_indices / jit_recent_indices; now copy fresh pointers.
    for i in 0..ctx.total_handles as usize {
        let spec = unsafe { &*ctx.lookup_specs.add(i) };
        let ps = unsafe { &*spec.rel };
        let idx: &crate::jit_index::JitHashIndex = if spec.use_recent != 0 {
            &ps.jit_recent_indices[spec.col as usize]
        } else {
            &ps.jit_indices[spec.col as usize]
        };
        let handle = unsafe { &mut *ctx.handles_buf.add(i) };
        handle.entries = idx.entries_ptr;
        handle.values = idx.values_ptr;
        handle.mask = idx.mask;
        handle._pad = 0;
    }
    changed
}
