//! Zero-callback JIT storage layer.
//!
//! Provides `#[repr(C)]` structs with fixed, asserted offsets that
//! dynasm-generated x86-64 code can address directly without any Rust
//! function calls in the inner loop.
//!
//! All heap allocations use `alloc_zeroed` / `dealloc` directly so that
//! `Drop` can recover them by reconstructing the original `Layout`.

// Not all storage types/methods are used yet — dead_code allowed intentionally.
#![allow(dead_code)]

use std::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use std::ptr;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Sentinel key value meaning "empty bucket" in `JitColIndex`.
pub const EMPTY_KEY: u32 = u32::MAX;

/// Sentinel tag value meaning "empty slot" in `JitTupleSet`.
pub const EMPTY_TAG: u32 = 0;

// ─── Hash helpers ───────────────────────────────────────────────────────────

/// 32-bit Knuth multiplicative hash — must match what the JIT emits.
#[inline(always)]
fn col_hash(key: u32) -> u32 {
    key.wrapping_mul(0x9e3779b9)
}

/// Multiply-accumulate tuple hash — must match what the JIT emits.
/// Returns a value != 0 (0 is reserved as `EMPTY_TAG`).
#[inline(always)]
fn tuple_hash(words: &[u32]) -> u32 {
    let mut h: u32 = 0x9e3779b9;
    for &w in words {
        h = h.wrapping_mul(0x9e3779b9).wrapping_add(w);
    }
    if h == 0 { 1 } else { h }
}

// ─── Raw allocation helpers ──────────────────────────────────────────────────

/// Allocate `count` zeroed `u32` values and return the raw pointer.
///
/// The returned pointer must be freed by calling `free_u32_slice(ptr, count)`.
/// For `count == 0` returns a dangling non-null pointer (no allocation).
fn alloc_u32_zeroed(count: usize) -> *mut u32 {
    if count == 0 {
        return ptr::NonNull::dangling().as_ptr();
    }
    let layout = Layout::array::<u32>(count).expect("layout overflow");
    let ptr = unsafe { alloc_zeroed(layout) } as *mut u32;
    assert!(!ptr.is_null(), "allocation failed");
    ptr
}

/// Allocate `count` zeroed `u64` values and return the raw pointer.
fn alloc_u64_zeroed(count: usize) -> *mut u64 {
    if count == 0 {
        return ptr::NonNull::dangling().as_ptr();
    }
    let layout = Layout::array::<u64>(count).expect("layout overflow");
    let ptr = unsafe { alloc_zeroed(layout) } as *mut u64;
    assert!(!ptr.is_null(), "allocation failed");
    ptr
}

/// Free a `u32` slice previously allocated with `alloc_u32_zeroed`.
/// No-op for `count == 0`.
unsafe fn free_u32_slice(ptr: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    let layout = Layout::array::<u32>(count).expect("layout overflow");
    unsafe { dealloc(ptr as *mut u8, layout) };
}

/// Free a `u64` slice previously allocated with `alloc_u64_zeroed`.
/// No-op for `count == 0`.
unsafe fn free_u64_slice(ptr: *mut u64, count: usize) {
    if count == 0 {
        return;
    }
    let layout = Layout::array::<u64>(count).expect("layout overflow");
    unsafe { dealloc(ptr as *mut u8, layout) };
}

/// Next power of two >= `n`, minimum 16.
fn next_pow2_min16(n: usize) -> usize {
    let mut cap = 16usize;
    while cap < n {
        cap <<= 1;
    }
    cap
}

/// Allocate `count` bytes filled with `fill` and return the raw pointer.
fn alloc_u8_filled(count: usize, fill: u8) -> *mut u8 {
    if count == 0 {
        return ptr::NonNull::dangling().as_ptr();
    }
    let layout = Layout::array::<u8>(count).expect("layout overflow");
    let ptr = unsafe { alloc_zeroed(layout) };
    assert!(!ptr.is_null(), "allocation failed");
    if fill != 0 {
        unsafe { ptr::write_bytes(ptr, fill, count) };
    }
    ptr
}

/// Free a `u8` slice previously allocated with `alloc_u8_filled`.
/// No-op for `count == 0`.
unsafe fn free_u8_slice(ptr: *mut u8, count: usize) {
    if count == 0 {
        return;
    }
    let layout = Layout::array::<u8>(count).expect("layout overflow");
    unsafe { dealloc(ptr, layout) };
}

// ─── JitColIndex ────────────────────────────────────────────────────────────

/// Per-column open-addressed hash map.
///
/// Maps a `u32` column value to a contiguous slice of `u32` values in `vals`.
/// For arity-2 relations the stored values are the *other* column's values;
/// for higher arities they are row indices.
///
/// Field offsets are fixed and verified by static assertions below.
#[repr(C)]
pub struct JitColIndex {
    /// Hash-table key array; `EMPTY_KEY` (0xFFFF_FFFF) = empty bucket.
    /// offset 0
    pub keys: *mut u32,
    /// Range array: `ranges[i] = start | (count << 32)`.
    /// offset 8
    pub ranges: *mut u64,
    /// Flat value array (values for all keys, grouped by key).
    /// offset 16
    pub vals: *mut u32,
    /// `capacity - 1`; capacity is always a power of 2.
    /// offset 24
    pub mask: u32,
    /// Number of occupied buckets.
    /// offset 28
    pub len: u32,
}

// Safety: JitColIndex is only used from a single JIT-callback thread.
unsafe impl Send for JitColIndex {}
unsafe impl Sync for JitColIndex {}

impl JitColIndex {
    /// Returns a zeroed, unallocated struct.  Must not be dereferenced until
    /// `build` has been called.
    pub fn new_empty() -> Self {
        JitColIndex {
            keys: ptr::null_mut(),
            ranges: ptr::null_mut(),
            vals: ptr::null_mut(),
            mask: 0,
            len: 0,
        }
    }

    /// Build a column index from a flat row-major tuple slice.
    ///
    /// `data[i * arity + col]` is the `col`-th field of tuple `i`.
    ///
    /// Stored values:
    /// - arity == 2 → the *other* column's value
    /// - arity != 2 → row index (`i as u32`)
    ///
    /// Values within each key group are sorted ascending (binary-search compatible).
    pub fn build(data: &[u32], arity: usize, col: usize) -> Box<Self> {
        assert!(arity >= 1);
        assert!(col < arity);
        let n_tuples = data.len() / arity;

        // ── pass 1: collect (key, value) pairs ──────────────────────────
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(n_tuples);
        for i in 0..n_tuples {
            let key = data[i * arity + col];
            let val = if arity == 2 {
                data[i * arity + (1 - col)] // other column
            } else {
                i as u32 // row index
            };
            pairs.push((key, val));
        }
        // Sort by key then value for grouping + binary-search compatibility.
        pairs.sort_unstable();

        // ── pass 2: build key→(start, count) groups ──────────────────────
        let mut groups: Vec<(u32, u32, u32)> = Vec::new(); // (key, start, count)
        {
            let mut i = 0usize;
            while i < pairs.len() {
                let key = pairs[i].0;
                let start = i as u32;
                while i < pairs.len() && pairs[i].0 == key {
                    i += 1;
                }
                let count = (i as u32) - start;
                groups.push((key, start, count));
            }
        }
        let n_keys = groups.len();
        let n_vals = pairs.len();

        // ── allocate hash table ──────────────────────────────────────────
        // Target load factor < 0.7; we need room for n_keys with that constraint.
        let cap = next_pow2_min16((n_keys * 10 / 7) + 1);
        let mask = (cap - 1) as u32;

        let keys_ptr = alloc_u32_zeroed(cap);
        // Fill with EMPTY_KEY.
        for i in 0..cap {
            unsafe { *keys_ptr.add(i) = EMPTY_KEY };
        }

        let ranges_ptr = alloc_u64_zeroed(cap);
        let vals_ptr = alloc_u32_zeroed(n_vals.max(1));

        // ── write vals array ─────────────────────────────────────────────
        for (idx, &(_, val)) in pairs.iter().enumerate() {
            unsafe { *vals_ptr.add(idx) = val };
        }

        // ── insert groups into hash table (linear probing) ───────────────
        for &(key, start, count) in &groups {
            let mut slot = (col_hash(key) & mask) as usize;
            loop {
                let existing = unsafe { *keys_ptr.add(slot) };
                if existing == EMPTY_KEY {
                    unsafe {
                        *keys_ptr.add(slot) = key;
                        *ranges_ptr.add(slot) = (start as u64) | ((count as u64) << 32);
                    }
                    break;
                }
                slot = (slot + 1) & (mask as usize);
            }
        }

        Box::new(JitColIndex {
            keys: keys_ptr,
            ranges: ranges_ptr,
            vals: vals_ptr,
            mask,
            len: n_keys as u32,
        })
    }

    /// Capacity (number of hash-table slots).
    #[inline]
    pub fn cap(&self) -> usize {
        (self.mask as usize) + 1
    }

    /// Total number of values stored across all key groups.
    ///
    /// Computed by summing `count` fields from all occupied buckets.
    pub fn vals_len(&self) -> usize {
        if self.keys.is_null() {
            return 0;
        }
        let cap = self.cap();
        let mut total = 0usize;
        for i in 0..cap {
            let k = unsafe { *self.keys.add(i) };
            if k != EMPTY_KEY {
                let r = unsafe { *self.ranges.add(i) };
                let count = (r >> 32) as u32;
                total += count as usize;
            }
        }
        total
    }

    /// Clone by directly copying the underlying arrays without resorting.
    ///
    /// `n_vals` is the total number of values in the `vals` array (= number of tuples
    /// stored in the source relation for arity-2; same for higher arities).
    /// This is O(n) memcpy instead of O(n log n) sort+build.
    pub fn clone_from_raw(&self, n_vals: usize) -> Box<Self> {
        if self.keys.is_null() {
            return Box::new(Self::new_empty());
        }
        let cap = (self.mask as usize) + 1;
        let keys = alloc_u32_zeroed(cap);
        let ranges = alloc_u64_zeroed(cap);
        let vals = alloc_u32_zeroed(n_vals.max(1));
        unsafe {
            ptr::copy_nonoverlapping(self.keys, keys, cap);
            ptr::copy_nonoverlapping(self.ranges, ranges, cap);
            ptr::copy_nonoverlapping(self.vals, vals, n_vals.max(1));
        }
        Box::new(JitColIndex { keys, ranges, vals, mask: self.mask, len: self.len })
    }
}

impl Drop for JitColIndex {
    fn drop(&mut self) {
        if self.keys.is_null() {
            return;
        }
        let cap = self.cap();
        let n_vals = self.vals_len();
        unsafe {
            free_u32_slice(self.keys, cap);
            free_u64_slice(self.ranges, cap);
            free_u32_slice(self.vals, n_vals.max(1));
        }
        self.keys = ptr::null_mut();
        self.ranges = ptr::null_mut();
        self.vals = ptr::null_mut();
    }
}

// ─── JitTupleSet ────────────────────────────────────────────────────────────

/// Full-tuple open-addressed hash set.
///
/// Each slot is `arity + 1` consecutive `u32` words:
/// `[hash_tag, col0, col1, …, colN]`. `hash_tag == 0` (`EMPTY_TAG`) means empty.
///
/// Field offsets are fixed and verified by static assertions below.
#[repr(C)]
pub struct JitTupleSet {
    /// Inline storage; stride = arity + 1 words.
    /// offset 0
    pub slots: *mut u32,
    /// `(cap_in_slots - 1)` where cap_in_slots is a power of 2.
    /// offset 8
    pub mask: u64,
    /// Number of occupied slots.
    /// offset 16
    pub len: u64,
}

// Safety: only used from the JIT-callback thread.
unsafe impl Send for JitTupleSet {}
unsafe impl Sync for JitTupleSet {}

impl JitTupleSet {
    /// Returns a zeroed, unallocated struct.
    pub fn new_empty() -> Self {
        JitTupleSet {
            slots: ptr::null_mut(),
            mask: 0,
            len: 0,
        }
    }

    /// Build a `JitTupleSet` from a flat row-major tuple slice.
    pub fn build(data: &[u32], arity: usize) -> Box<Self> {
        let n_tuples = if arity == 0 { 0 } else { data.len() / arity };
        let stride = arity + 1;

        let cap = next_pow2_min16((n_tuples * 10 / 7) + 1);
        let mask = (cap - 1) as u64;

        // Zeroed allocation → all hash_tags are 0 == EMPTY_TAG.
        let slots_ptr = alloc_u32_zeroed(cap * stride);

        let mut occupied = 0u64;
        for i in 0..n_tuples {
            let tuple = &data[i * arity..(i + 1) * arity];
            let h = tuple_hash(tuple);
            let mut slot = (h as u64 & mask) as usize;
            loop {
                let tag = unsafe { *slots_ptr.add(slot * stride) };
                if tag == EMPTY_TAG {
                    // Write into empty slot.
                    unsafe {
                        *slots_ptr.add(slot * stride) = h;
                        for (j, &v) in tuple.iter().enumerate() {
                            *slots_ptr.add(slot * stride + 1 + j) = v;
                        }
                    }
                    occupied += 1;
                    break;
                }
                // Check for duplicate tuple (dedup).
                let existing_matches = {
                    let mut m = true;
                    for (j, &tv) in tuple.iter().enumerate() {
                        if unsafe { *slots_ptr.add(slot * stride + 1 + j) } != tv {
                            m = false;
                            break;
                        }
                    }
                    m
                };
                if existing_matches {
                    break; // already present — skip
                }
                slot = ((slot as u64 + 1) & mask) as usize;
            }
        }

        Box::new(JitTupleSet {
            slots: slots_ptr,
            mask,
            len: occupied,
        })
    }

    /// Probe the set (Rust-side correctness checks).
    ///
    /// # Safety
    /// `self` must have been built via `build` and not yet dropped.
    pub unsafe fn contains(&self, tuple: &[u32]) -> bool {
        if self.slots.is_null() {
            return false;
        }
        let arity = tuple.len();
        let stride = arity + 1;
        let h = tuple_hash(tuple);
        let mut slot = (h as u64 & self.mask) as usize;
        loop {
            let tag = unsafe { *self.slots.add(slot * stride) };
            if tag == EMPTY_TAG {
                return false;
            }
            if tag == h {
                let mut matches = true;
                for (j, &tv) in tuple.iter().enumerate() {
                    if unsafe { *self.slots.add(slot * stride + 1 + j) } != tv {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    return true;
                }
            }
            slot = ((slot as u64 + 1) & self.mask) as usize;
        }
    }

    /// Capacity in slots.
    #[inline]
    pub fn cap_in_slots(&self) -> usize {
        (self.mask + 1) as usize
    }
}

/// Insert a single tuple into a raw `JitTupleSet` slot buffer without grow checks.
///
/// `slots` must be a zeroed allocation with `(mask + 1)` slots of `arity + 1` words each.
/// The caller must ensure load factor < 70% before calling (no automatic grow).
///
/// # Safety
/// `slots` must be valid, zeroed, and have capacity `(mask + 1) * (arity + 1)` u32 words.
#[inline]
unsafe fn jit_tuple_set_insert_unchecked(slots: *mut u32, mask: u64, tuple: &[u32]) {
    let arity = tuple.len();
    let stride = arity + 1;
    let h = tuple_hash(tuple);
    let mut slot = (h as u64 & mask) as usize;
    loop {
        let tag = unsafe { *slots.add(slot * stride) };
        if tag == EMPTY_TAG {
            unsafe {
                *slots.add(slot * stride) = h;
                for (j, &v) in tuple.iter().enumerate() {
                    *slots.add(slot * stride + 1 + j) = v;
                }
            }
            return;
        }
        // Dedup: skip if already present (shouldn't happen for fresh inserts, but be safe).
        let matches = (0..arity).all(|j| unsafe { *slots.add(slot * stride + 1 + j) == tuple[j] });
        if matches {
            return;
        }
        slot = ((slot as u64 + 1) & mask) as usize;
    }
}

// JitTupleSet does NOT implement Drop for freeing slots — the owner
// (JitRelData) holds the arity and is responsible for freeing slots.
// Dropping a standalone JitTupleSet built via `build` will leak.
// This is intentional: JitTupleSet is always embedded in JitRelData.
impl Drop for JitTupleSet {
    fn drop(&mut self) {
        // Intentional no-op: JitRelData::drop handles freeing slots.
        // Direct use of JitTupleSet::build (outside JitRelData) leaks the
        // slots allocation; that is acceptable for the current use case.
    }
}

// ─── JitSwissTable ──────────────────────────────────────────────────────────

/// Swiss-table existence set for O(1) SIMD-probed membership queries.
///
/// Control bytes (1 per slot, packed 16 per cache line) let the JIT check 16
/// candidates with a single `pcmpeqb`/`pmovmskb` pair before touching tuple
/// data.  Layout:
/// - `ctrl[0..cap]`: per-slot tag byte (0xFF = empty, 0x00-0x7F = occupied H2)
/// - `ctrl[cap..cap+16]`: mirror of ctrl[0..16] for safe SIMD overread at wrap
/// - `data[0..cap*arity]`: packed tuple data, slot `i` at `data[i*arity]`
///
/// Field offsets are fixed and verified by static assertions below.
#[repr(C)]
pub struct JitSwissTable {
    /// Control-byte array (cap + 16 bytes).  0xFF = empty.
    /// offset 0
    pub ctrl: *mut u8,
    /// Packed tuple data (cap * arity u32 words).
    /// offset 8
    pub data: *mut u32,
    /// cap - 1 where cap is a power of two ≥ 16.
    /// offset 16
    pub mask: u64,
}

// Safety: only accessed from the JIT-callback thread.
unsafe impl Send for JitSwissTable {}
unsafe impl Sync for JitSwissTable {}

/// SIMD-empty sentinel for `JitSwissTable` control bytes.
pub const SWISS_EMPTY: u8 = 0xFF;

impl JitSwissTable {
    /// Null/unbuilt sentinel.
    pub const NULL: Self = JitSwissTable { ctrl: ptr::null_mut(), data: ptr::null_mut(), mask: 0 };

    pub fn is_null(&self) -> bool { self.ctrl.is_null() }

    /// Build from flat row-major packed data.  `arity` must be in 1–3.
    ///
    /// H2 tag = top 7 bits of 32-bit `tuple_hash`, always < 0x80.
    /// Load factor is kept below 7/8; capacity is next power-of-two ≥ 16.
    pub fn build(packed: &[u32], arity: usize) -> Self {
        debug_assert!((1..=3).contains(&arity));
        let n = packed.len() / arity;
        let cap = next_pow2_min16((n * 8 / 7) + 1);
        let mask = (cap - 1) as u64;

        // Control bytes: cap + 16 (extra 16 = SIMD overread guard), all 0xFF.
        let ctrl = alloc_u8_filled(cap + 16, SWISS_EMPTY);
        // Tuple data: cap * arity u32s (zeroed for safety; only read on H2 match).
        let data = alloc_u32_zeroed(cap * arity);

        let st = JitSwissTable { ctrl, data, mask };

        // Insert all tuples.
        for i in 0..n {
            let tuple = &packed[i * arity..(i + 1) * arity];
            st.insert(tuple, arity, cap);
        }

        // Mirror first 16 control bytes at [cap..cap+16] for safe wrap-around reads.
        unsafe { ptr::copy_nonoverlapping(ctrl, ctrl.add(cap), 16) };

        st
    }

    /// Insert a single tuple (no dedup check — caller must ensure uniqueness).
    fn insert(&self, tuple: &[u32], arity: usize, _cap: usize) {
        let h = tuple_hash(tuple);
        let h2 = (h >> 25) as u8; // top 7 bits, always 0x00-0x7F
        let mask = self.mask as usize;

        // Group-aligned probe start.
        let mut group = (h as usize & mask) & !15;
        loop {
            for bit in 0..16 {
                let slot = (group + bit) & mask;
                let cb = unsafe { *self.ctrl.add(slot) };
                if cb == SWISS_EMPTY {
                    unsafe {
                        *self.ctrl.add(slot) = h2;
                        for (j, &word) in tuple.iter().enumerate().take(arity) {
                            *self.data.add(slot * arity + j) = word;
                        }
                    }
                    return;
                }
            }
            group = (group + 16) & mask;
            debug_assert_ne!(group, (h as usize & mask) & !15, "JitSwissTable: full table");
        }
    }

    /// Rust-side membership check (for tests / correctness verification).
    pub unsafe fn contains(&self, tuple: &[u32]) -> bool {
        if self.ctrl.is_null() { return false; }
        let arity = tuple.len();
        let h = tuple_hash(tuple);
        let h2 = (h >> 25) as u8;
        let mask = self.mask as usize;
        let cap = mask + 1;
        let mut group = (h as usize & mask) & !15;
        loop {
            for bit in 0..16 {
                let slot = (group + bit) & mask;
                let cb = unsafe { *self.ctrl.add(slot) };
                if cb == SWISS_EMPTY { return false; }
                if cb == h2 {
                    let matches = (0..arity)
                        .all(|j| unsafe { *self.data.add(slot * arity + j) == tuple[j] });
                    if matches { return true; }
                }
            }
            group = (group + 16) & mask;
            if group == (h as usize & mask) & !15 { return false; } // full scan
            let _ = cap;
        }
    }

    /// Free owned allocations.  No-op if null.
    pub unsafe fn free(&mut self, arity: usize) {
        if self.ctrl.is_null() { return; }
        let cap = (self.mask + 1) as usize;
        unsafe { free_u8_slice(self.ctrl, cap + 16) };
        if arity > 0 {
            unsafe { free_u32_slice(self.data, cap * arity) };
        }
        *self = JitSwissTable::NULL;
    }
}

// ─── JitRelData ─────────────────────────────────────────────────────────────

/// One relation version (total, delta, or new).
///
/// Layout (C-visible, 88 bytes):
/// - offset  0: `data`        (*mut u32)
/// - offset  8: `len`         (u64)
/// - offset 16: `cap`         (u64)
/// - offset 24: `col_indices` (*mut *mut JitColIndex)
/// - offset 32: `tuple_set`   (JitTupleSet, 24 bytes)
/// - offset 56: `arity`       (u32)
/// - offset 60: `_pad`        (u32)
/// - offset 64: `swiss`       (JitSwissTable, 24 bytes) — SIMD existence probe
///
/// Private fields follow at offset 88 (not visible to JIT code).
#[repr(C)]
pub struct JitRelData {
    /// Packed tuples, row-major, stride = arity.
    /// offset 0
    pub data: *mut u32,
    /// Tuple count.
    /// offset 8
    pub len: u64,
    /// Capacity in tuples.
    /// offset 16
    pub cap: u64,
    /// `array[arity]` of pointers to `JitColIndex` (null per column if not built).
    /// offset 24
    pub col_indices: *mut *mut JitColIndex,
    /// Full-tuple membership set (scalar open-address; used for head dedup on `new`).
    /// offset 32  (JitTupleSet is 24 bytes → ends at 56)
    pub tuple_set: JitTupleSet,
    /// Relation arity.
    /// offset 56
    pub arity: u32,
    #[doc(hidden)]
    pub _pad: u32,
    /// SIMD Swiss-table existence probe (for fully-bound body-clause checks on `total`).
    /// Built for arity 1–3 when `build_tuple_set=true`; null otherwise.
    /// offset 64  (JitSwissTable is 24 bytes → ends at 88)
    pub swiss: JitSwissTable,
    // ── private fields beyond the 88-byte C region ───────────────────────
    // Total words allocated in tuple_set.slots (cap_in_slots * (arity + 1)).
    // Used by Drop to reconstruct the layout for dealloc.
    // Sentinel value `usize::MAX` means the slots are aliased (not owned) and
    // must NOT be freed by Drop — set by alias_tuple_set().
    _ts_slots_words: usize,
}

// Safety: only used from the JIT-callback thread.
unsafe impl Send for JitRelData {}
unsafe impl Sync for JitRelData {}

impl JitRelData {
    /// Returns a zeroed, unallocated `Box<JitRelData>`.
    pub fn new_empty(arity: usize) -> Box<Self> {
        Box::new(JitRelData {
            data: ptr::null_mut(),
            len: 0,
            cap: 0,
            col_indices: ptr::null_mut(),
            tuple_set: JitTupleSet::new_empty(),
            arity: arity as u32,
            _pad: 0,
            swiss: JitSwissTable::NULL,
            _ts_slots_words: 0,
        })
    }

    /// Build from a flat row-major packed tuple slice.
    ///
    /// If `build_indices` is true, column indices are built for all `arity` columns.
    /// The `JitTupleSet` is always built (used for cross-iteration head dedup on total).
    pub fn build_from_packed(data: &[u32], arity: usize, build_indices: bool) -> Box<Self> {
        Self::build_from_packed_impl(data, arity, build_indices, true)
    }

    /// Like `build_from_packed`, but skips building the `JitTupleSet`.
    ///
    /// Use this for `recent` buffers that are only iterated over, never probed as a
    /// membership set (the JIT only probes `total.tuple_set` for head dedup, not
    /// `recent.tuple_set`).
    pub fn build_from_packed_no_tupleset(data: &[u32], arity: usize, build_indices: bool) -> Box<Self> {
        Self::build_from_packed_impl(data, arity, build_indices, false)
    }

    fn build_from_packed_impl(data: &[u32], arity: usize, build_indices: bool, build_tuple_set: bool) -> Box<Self> {
        let n_tuples = if arity == 0 { 0 } else { data.len() / arity };

        // ── data array ───────────────────────────────────────────────────
        let data_words = n_tuples * arity.max(1);
        let data_cap = n_tuples.max(1); // capacity in tuples
        let data_ptr = alloc_u32_zeroed(data_cap * arity.max(1));
        if !data.is_empty() {
            unsafe { ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data_words) };
        }

        // ── tuple set ────────────────────────────────────────────────────
        // Build, then manually decompose so we can store the slots_words for Drop.
        // Skipped for `recent` buffers that are only iterated, never probed.
        let (ts_slots, ts_mask, ts_len, ts_slots_words) = if build_tuple_set {
            let ts = JitTupleSet::build(data, arity);
            let stride = arity + 1;
            let ts_cap = ts.cap_in_slots();
            let ts_slots_words = ts_cap * stride;
            let ts_slots = ts.slots;
            let ts_mask = ts.mask;
            let ts_len = ts.len;
            // Suppress the (no-op) drop — we own the memory.
            std::mem::forget(ts);
            (ts_slots, ts_mask, ts_len, ts_slots_words)
        } else {
            (ptr::null_mut(), 0u64, 0u64, 0usize)
        };

        // ── Swiss table ───────────────────────────────────────────────────
        // SIMD existence probe for fully-bound body clauses.  Built alongside
        // tuple_set for arity 1–3; null for arity 0 or arity > 3.
        let swiss = if build_tuple_set && (1..=3).contains(&arity) {
            JitSwissTable::build(data, arity)
        } else {
            JitSwissTable::NULL
        };

        // ── column index pointer array ────────────────────────────────────
        // We allocate an array of `arity` raw pointers.  `*mut *mut JitColIndex`
        // has the same size as `*mut usize` on the target, so we allocate
        // enough bytes and cast.
        let ptr_bytes = arity * std::mem::size_of::<*mut JitColIndex>();
        let col_indices_ptr: *mut *mut JitColIndex = if arity > 0 {
            // Allocate as bytes via a u8 layout for exact sizing.
            let layout =
                Layout::array::<*mut JitColIndex>(arity).expect("layout overflow");
            let raw = unsafe { alloc_zeroed(layout) } as *mut *mut JitColIndex;
            assert!(!raw.is_null(), "allocation failed");
            if build_indices {
                for col in 0..arity {
                    let idx = JitColIndex::build(data, arity, col);
                    unsafe { *raw.add(col) = Box::into_raw(idx) };
                }
            }
            raw
        } else {
            ptr::null_mut()
        };
        let _ = ptr_bytes; // used only for documentation

        Box::new(JitRelData {
            data: data_ptr,
            len: n_tuples as u64,
            cap: data_cap as u64,
            col_indices: col_indices_ptr,
            tuple_set: JitTupleSet {
                slots: ts_slots,
                mask: ts_mask,
                len: ts_len,
            },
            arity: arity as u32,
            _pad: 0,
            swiss,
            _ts_slots_words: ts_slots_words,
        })
    }
}

impl JitRelData {
    /// Reset this JitRelData's `new` buffer in-place for the next fixpoint iteration.
    ///
    /// Sets `len = 0` and zeros the `tuple_set` slots without reallocating, so the
    /// JIT can reuse the same memory for the next iteration's batch of new tuples.
    /// Called instead of `std::mem::replace(&mut native.new, build_from_packed(&[], …))`
    /// to avoid one alloc+free per advance step per relation.
    pub fn reset_for_new_iteration(&mut self) {
        self.len = 0;
        if !self.tuple_set.slots.is_null() && self.tuple_set.mask > 0 {
            // Compute the actual slot count from mask (always in sync, even after
            // jit_tuple_set_grow — unlike _ts_slots_words which isn't updated by grow).
            let stride = self.arity as usize + 1;
            let cap = (self.tuple_set.mask + 1) as usize;
            // SAFETY: slots points to cap * stride u32 values.
            unsafe {
                std::ptr::write_bytes(self.tuple_set.slots, 0, cap * stride);
            }
            self.tuple_set.len = 0;
        }
    }

    /// Pre-size this buffer (data + tuple_set) to hold at least `n_tuples` without growing.
    ///
    /// Called once per stratum run using count hints from a prior run.  Applies to both
    /// `native.new` (head buffer) and `native.total` (accumulated tuples) — both start at
    /// capacity 16 on a fresh engine and grow 7× for triangle n=20 (1140 tuples) without hints.
    /// Pre-sizing replaces the 7 grow-rehash cascades with a single upfront alloc + memset.
    ///
    /// # Safety
    /// `self` must be a fully initialized `JitRelData` (built by `build_from_packed`).
    pub unsafe fn pre_size(&mut self, n_tuples: usize) {
        if n_tuples == 0 {
            return;
        }
        let arity = self.arity as usize;
        let arity_max1 = arity.max(1);

        // Grow data buffer if needed (no zeroing needed — just capacity).
        if n_tuples > self.cap as usize {
            let old_cap = self.cap as usize;
            let mut new_cap = old_cap.max(1);
            while new_cap < n_tuples {
                new_cap *= 2;
            }
            let old_layout = Layout::array::<u32>(old_cap * arity_max1).expect("layout");
            let new_ptr = unsafe {
                realloc(
                    self.data as *mut u8,
                    old_layout,
                    new_cap * arity_max1 * std::mem::size_of::<u32>(),
                ) as *mut u32
            };
            assert!(!new_ptr.is_null(), "pre_size: data realloc failed");
            self.data = new_ptr;
            self.cap = new_cap as u64;
        }

        // Grow tuple_set if needed.  Must be zeroed (open-addressed hash needs empty markers).
        if arity > 0 {
            let stride = arity + 1;
            let needed_cap = next_pow2_min16((n_tuples * 10 / 7) + 1);
            let current_cap = if !self.tuple_set.slots.is_null() {
                // Mask encodes current capacity: cap = mask + 1.
                (self.tuple_set.mask as usize) + 1
            } else {
                0
            };
            if needed_cap > current_cap {
                // Free old allocation.
                if !self.tuple_set.slots.is_null() && current_cap > 0 {
                    let old_layout = Layout::array::<u32>(current_cap * stride).expect("layout");
                    unsafe { dealloc(self.tuple_set.slots as *mut u8, old_layout) };
                }
                // Alloc new zeroed allocation.
                let new_slots = alloc_u32_zeroed(needed_cap * stride);
                self.tuple_set.slots = new_slots;
                self.tuple_set.mask = (needed_cap - 1) as u64;
                self.tuple_set.len = 0;
                self._ts_slots_words = needed_cap * stride;
            }
        }
    }

    /// Point `tuple_set` at `jit_dedup`'s backing storage instead of maintaining
    /// a separate owned copy.
    ///
    /// After calling this, the JIT's `total.tuple_set` probe reads directly from
    /// the `JitDedupTable` entries.  This is valid when:
    /// 1. `jit_dedup_hash` and `tuple_hash` produce identical values (unified
    ///    after the 2026-03-19 hash unification: both start at `0x9e3779b9`,
    ///    both use sentinel 0).
    /// 2. `jit_dedup` contains every tuple that was inserted into this relation
    ///    (guaranteed by `insert_packed_raw_native_flush` which calls
    ///    `jit_dedup.insert` before this method is called).
    ///
    /// The aliased memory is NOT freed by `Drop`; the owning `JitDedupTable`
    /// is responsible for its lifetime.
    ///
    /// # Safety
    /// `entries` must point to a valid flat slot array of at least `(mask+1)*(arity+1)` u32s,
    /// owned by a `JitDedupTable` that outlives `self`.
    pub unsafe fn alias_tuple_set(&mut self, entries: *mut u32, mask: u64) {
        // Free any previously owned tuple_set allocation.
        if !self.tuple_set.slots.is_null() && self.tuple_set.mask > 0
            && self._ts_slots_words != usize::MAX
        {
            unsafe { free_u32_slice(self.tuple_set.slots, self._ts_slots_words) };
        }
        self.tuple_set.slots = entries;
        self.tuple_set.mask = mask;
        self.tuple_set.len = 0; // not used during probe
        // Sentinel: usize::MAX means the slot array is aliased (not owned).
        self._ts_slots_words = usize::MAX;
    }

    /// Appends `new_tuples` (flat row-major, stride=arity) to this relation's data buffer,
    /// rebuilds all column indices from the extended data, and optionally updates the
    /// `tuple_set` to include the new tuples.
    ///
    /// Pass `build_tuple_set = false` when the caller will alias `tuple_set` to
    /// `jit_dedup` immediately after this call (skips the ~12µs tuple_set update
    /// for IDB sink relations in the native advance path).
    ///
    /// # Safety
    /// `self` must be a fully initialized `JitRelData` from `build_from_packed`.
    pub unsafe fn extend_and_rebuild_indices(
        &mut self,
        new_tuples: &[u32],
        arity: usize,
        build_indices: bool,
        build_tuple_set: bool,
    ) {
        let arity_max1 = arity.max(1);
        let n_new = new_tuples.len() / arity_max1;
        if n_new == 0 {
            return;
        }

        let old_len = self.len as usize;
        let new_len = old_len + n_new;

        // Grow data buffer if needed.
        if new_len > self.cap as usize {
            let mut new_cap = (self.cap as usize).max(1);
            while new_cap < new_len {
                new_cap *= 2;
            }
            let old_layout = Layout::array::<u32>((self.cap as usize) * arity_max1)
                .expect("layout overflow");
            let new_ptr = unsafe {
                realloc(
                    self.data as *mut u8,
                    old_layout,
                    new_cap * arity_max1 * std::mem::size_of::<u32>(),
                ) as *mut u32
            };
            assert!(!new_ptr.is_null(), "extend_and_rebuild_indices: realloc failed");
            // Zero-initialize newly added region.
            unsafe {
                std::ptr::write_bytes(
                    new_ptr.add(old_len * arity_max1),
                    0,
                    (new_cap - old_len) * arity_max1,
                );
            }
            self.data = new_ptr;
            self.cap = new_cap as u64;
        }

        // Append new tuples.
        unsafe {
            ptr::copy_nonoverlapping(
                new_tuples.as_ptr(),
                self.data.add(old_len * arity_max1),
                new_tuples.len(),
            );
        }
        self.len = new_len as u64;

        // Update tuple_set: required for cross-iteration head dedup (JIT probes total.tuple_set).
        // Skipped when the caller will alias tuple_set to jit_dedup after this call.
        if build_tuple_set && arity > 0 {
            let stride = arity + 1;

            // Check if existing capacity can accommodate new_len at <70% load factor.
            // If so, just insert the n_new new tuples (old_len..new_len) without
            // zeroing or reinserting the old tuples — they are still correct in the set.
            let current_cap = if !self.tuple_set.slots.is_null() {
                self._ts_slots_words / stride
            } else {
                0
            };

            if current_cap > 0 && (new_len * 10 / 7) < current_cap {
                // Incremental path: current allocation fits new_len at <70% load.
                // Insert only the n_new new tuples into the existing set.
                let slots = self.tuple_set.slots;
                let mask = self.tuple_set.mask;
                for i in old_len..new_len {
                    let tuple = unsafe {
                        std::slice::from_raw_parts(self.data.add(i * arity), arity)
                    };
                    unsafe { jit_tuple_set_insert_unchecked(slots, mask, tuple) };
                }
                self.tuple_set.len = new_len as u64;
            } else {
                // Growth path: allocate a new larger set, zero it, and reinsert all tuples.
                let needed_cap = next_pow2_min16((new_len * 10 / 7) + 1);
                let needed_words = needed_cap * stride;
                if !self.tuple_set.slots.is_null() && self._ts_slots_words > 0 {
                    unsafe { free_u32_slice(self.tuple_set.slots, self._ts_slots_words) };
                }
                let new_slots = alloc_u32_zeroed(needed_words);
                self._ts_slots_words = needed_words;
                let mask = (needed_cap - 1) as u64;
                self.tuple_set.slots = new_slots;
                self.tuple_set.mask = mask;
                self.tuple_set.len = 0;
                // Reinsert all new_len tuples into the fresh set.
                for i in 0..new_len {
                    let tuple = unsafe {
                        std::slice::from_raw_parts(self.data.add(i * arity), arity)
                    };
                    unsafe { jit_tuple_set_insert_unchecked(new_slots, mask, tuple) };
                }
                self.tuple_set.len = new_len as u64;
            }
        }

        if !build_indices || arity == 0 {
            return;
        }

        // Rebuild all column indices from the full extended data.
        let full_slice =
            unsafe { std::slice::from_raw_parts(self.data, new_len * arity) };
        let col_indices_ptr = self.col_indices;
        for col in 0..arity {
            // Drop old index.
            let old_ptr = unsafe { *col_indices_ptr.add(col) };
            if !old_ptr.is_null() {
                drop(unsafe { Box::from_raw(old_ptr) });
            }
            // Build and store new index.
            let new_idx = JitColIndex::build(full_slice, arity, col);
            unsafe { *col_indices_ptr.add(col) = Box::into_raw(new_idx) };
        }
    }
}

impl Drop for JitRelData {
    fn drop(&mut self) {
        let arity = self.arity as usize;

        // Free data array.
        if !self.data.is_null() {
            unsafe { free_u32_slice(self.data, (self.cap as usize) * arity.max(1)) };
            self.data = ptr::null_mut();
        }

        // Free column indices (and the pointer array itself).
        if !self.col_indices.is_null() {
            for col in 0..arity {
                let idx_ptr = unsafe { *self.col_indices.add(col) };
                if !idx_ptr.is_null() {
                    // Reconstruct Box so JitColIndex::drop runs.
                    drop(unsafe { Box::from_raw(idx_ptr) });
                }
            }
            // Free the pointer array.
            let layout = Layout::array::<*mut JitColIndex>(arity).expect("layout overflow");
            unsafe { dealloc(self.col_indices as *mut u8, layout) };
            self.col_indices = ptr::null_mut();
        }

        // Free tuple_set slots (only if owned — not an alias into jit_dedup).
        // _ts_slots_words == usize::MAX is the sentinel meaning "aliased, do not free".
        if !self.tuple_set.slots.is_null() && self.tuple_set.mask > 0
            && self._ts_slots_words != usize::MAX
        {
            let stride = arity + 1;
            let cap = (self.tuple_set.mask + 1) as usize;
            unsafe { free_u32_slice(self.tuple_set.slots, cap * stride) };
            self.tuple_set.slots = ptr::null_mut();
        }

        // Free Swiss table (if not null).
        unsafe { self.swiss.free(arity) };
    }
}

// ─── JitNativeRelData ────────────────────────────────────────────────────────

/// Three `JitRelData` views of a `PackedStorage` relation: total, recent, and new.
///
/// Kept fresh by `PackedStorage::advance_jit()`:
/// - `total`  — all tuples (`packed_data[0..count*arity]`), column indices built.
/// - `recent` — only the tuples that became recent this iteration.
/// - `new`    — empty write buffer for JIT-generated tuples; reset after each advance.
pub struct JitNativeRelData {
    pub total: Box<JitRelData>,
    pub recent: Box<JitRelData>,
    pub new: Box<JitRelData>,
    /// Number of tuples in `total` at the time it was last built.
    /// Used by `advance_jit` to extend incrementally instead of full rebuild.
    pub total_built_count: usize,
    /// Whether to build `JitColIndex` arrays during advance.
    ///
    /// `true` for the asm native path (which reads `col_indices` directly).
    /// `false` for lean Cranelift projections (which only read `data` and `len`).
    pub build_indices: bool,
}

// Safety: JitRelData contains raw pointers and is !Send by default, but all
// three views are only ever accessed from the single JIT-evaluation thread.
unsafe impl Send for JitNativeRelData {}
unsafe impl Sync for JitNativeRelData {}

impl std::fmt::Debug for JitNativeRelData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitNativeRelData")
            .field("total_len", &self.total.len)
            .field("recent_len", &self.recent.len)
            .field("new_len", &self.new.len)
            .finish()
    }
}

/// Clone a `JitRelData` by directly copying its raw data buffer, tuple_set, and (optionally)
/// JitColIndex arrays. O(n) memcpy instead of O(n log n) sort+rebuild.
///
/// When `copy_tuple_set` is false, the clone gets a null tuple_set (safe for `recent` buffers
/// that are never probed for membership, or for IDB sink relations where aliasing follows).
///
/// # Safety
/// `src` must be a valid, fully-initialized `JitRelData`.
pub(crate) unsafe fn clone_jit_rel_data_with_indices(
    src: &JitRelData,
    arity: usize,
    copy_tuple_set: bool,
) -> Box<JitRelData> {
    let n = src.len as usize;
    let arity_max1 = arity.max(1);

    // ── data buffer ─────────────────────────────────────────────────────────
    let data_cap = (src.cap as usize).max(1);
    let data_ptr = alloc_u32_zeroed(data_cap * arity_max1);
    if n > 0 && !src.data.is_null() {
        unsafe { ptr::copy_nonoverlapping(src.data, data_ptr, n * arity_max1) };
    }

    // ── tuple_set ────────────────────────────────────────────────────────────
    let (ts_slots, ts_mask, ts_len, ts_slots_words) = if copy_tuple_set
        && arity > 0
        && !src.tuple_set.slots.is_null()
        && src.tuple_set.mask > 0
        && src._ts_slots_words != usize::MAX // not aliased
    {
        let stride = arity + 1;
        let cap = (src.tuple_set.mask + 1) as usize;
        let words = cap * stride;
        let slots = alloc_u32_zeroed(words);
        unsafe { ptr::copy_nonoverlapping(src.tuple_set.slots, slots, words) };
        (slots, src.tuple_set.mask, src.tuple_set.len, words)
    } else {
        (ptr::null_mut(), 0u64, 0u64, 0usize)
    };

    // ── col_indices ──────────────────────────────────────────────────────────
    let col_indices_ptr: *mut *mut JitColIndex = if arity > 0
        && !src.col_indices.is_null()
    {
        let layout = Layout::array::<*mut JitColIndex>(arity).expect("layout overflow");
        let raw = unsafe { alloc_zeroed(layout) } as *mut *mut JitColIndex;
        assert!(!raw.is_null(), "allocation failed");
        for col in 0..arity {
            let src_idx_ptr = unsafe { *src.col_indices.add(col) };
            if !src_idx_ptr.is_null() {
                let src_idx = unsafe { &*src_idx_ptr };
                // n_vals = n_tuples (one value per tuple in the vals array).
                let cloned = src_idx.clone_from_raw(n);
                unsafe { *raw.add(col) = Box::into_raw(cloned) };
            }
        }
        raw
    } else {
        ptr::null_mut()
    };

    // ── Swiss table ──────────────────────────────────────────────────────────
    // Rebuild from the copied data (cheaper than a byte-for-byte ctrl copy since
    // the table is small relative to tuple data, and ensures correct state).
    let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, n * arity_max1) };
    let swiss = if copy_tuple_set && (1..=3).contains(&arity) && n > 0 {
        JitSwissTable::build(data_slice, arity)
    } else {
        JitSwissTable::NULL
    };

    Box::new(JitRelData {
        data: data_ptr,
        len: src.len,
        cap: data_cap as u64,
        col_indices: col_indices_ptr,
        tuple_set: JitTupleSet { slots: ts_slots, mask: ts_mask, len: ts_len },
        arity: src.arity,
        _pad: 0,
        swiss,
        _ts_slots_words: ts_slots_words,
    })
}

impl JitNativeRelData {
    /// Deep-clone this `JitNativeRelData` by directly copying pre-built arrays.
    ///
    /// Uses O(n) memcpy for both data buffers, tuple_set, and JitColIndex arrays
    /// instead of the O(n log n) sort+rebuild that `build_native_projection` would do.
    /// This eliminates the ~10µs sort cost per `jit_hot` benchmark iteration.
    ///
    /// Used by `PackedStorage::Clone` to preserve the prebuilt native projection across
    /// clones so that the first `jit_advance_native` call in a cloned engine does not pay
    /// the full `build_native_projection` rebuild cost.
    pub fn deep_clone(&self) -> Self {
        let arity = self.total.arity as usize;
        let build_indices = !self.total.col_indices.is_null();

        // Clone total: copy data buffer + tuple_set + col_indices (no sort needed).
        let total_clone = unsafe { clone_jit_rel_data_with_indices(&self.total, arity, true) };

        // Clone recent: copy data buffer only (no tuple_set, no col_indices).
        // `recent` is only iterated, never probed as a membership set.
        let recent_clone = if self.recent.len > 0 {
            unsafe { clone_jit_rel_data_with_indices(&self.recent, arity, false) }
        } else {
            JitRelData::build_from_packed_no_tupleset(&[], arity, false)
        };

        // `new` is always an empty write buffer — no data to copy.
        let new_clone = JitRelData::build_from_packed(&[], arity, false);

        let _ = build_indices; // determined by col_indices presence in clone_jit_rel_data
        JitNativeRelData {
            total: total_clone,
            recent: recent_clone,
            new: new_clone,
            total_built_count: self.total_built_count,
            build_indices: self.build_indices,
        }
    }
}

// ─── JIT growth callbacks ────────────────────────────────────────────────────
//
// These are the ONLY Rust functions the new JIT hot path calls — and only on
// capacity overflow (rare).  All other data access is direct memory reads/writes
// using the fixed offsets verified by the assertions below.

/// Called from JIT when `JitRelData.len == JitRelData.cap`.
/// Doubles the `data` buffer capacity; updates `rel.data` and `rel.cap` in place.
///
/// # Safety
/// `rel` must be a valid non-null `*mut JitRelData` with a properly initialised
/// `data`, `cap`, and `arity` field.  The caller must NOT have incremented `len`
/// yet — this function does not touch `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rel_data_grow(rel: *mut JitRelData) {
    let rel = unsafe { &mut *rel };
    let arity = rel.arity as usize;
    let old_cap = rel.cap as usize;
    let new_cap = (old_cap * 2).max(1);

    // Each tuple is `arity` u32 words (minimum 1 word to satisfy allocator).
    let word_size = arity.max(1);
    let old_words = old_cap * word_size;
    let new_words = new_cap * word_size;

    let old_layout = Layout::array::<u32>(old_words).expect("layout overflow");
    let new_layout = Layout::array::<u32>(new_words).expect("layout overflow");

    let new_ptr = unsafe {
        realloc(rel.data as *mut u8, old_layout, new_layout.size()) as *mut u32
    };
    assert!(!new_ptr.is_null(), "jit_rel_data_grow: allocation failed");

    // Zero-initialise the newly added region so stale bytes are not visible.
    unsafe {
        std::ptr::write_bytes(new_ptr.add(old_words), 0, new_words - old_words);
    }

    rel.data = new_ptr;
    rel.cap = new_cap as u64;
}

/// Called from JIT when `JitTupleSet` load factor exceeds threshold
/// (`len * 10 > cap * 7`).  Doubles capacity and rehashes all existing entries.
///
/// # Safety
/// `ts` must be a valid non-null `*mut JitTupleSet` with a properly initialised
/// `slots`, `mask`, and `len` field.  `arity` must be >= 1 and match the arity
/// that was used when the slot buffer was originally allocated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_tuple_set_grow(ts: *mut JitTupleSet, arity: u32) {
    let ts = unsafe { &mut *ts };
    let arity = arity as usize;
    let stride = arity + 1; // words per slot: [hash_tag, col0…colN]

    let old_cap = (ts.mask + 1) as usize; // must be power of 2
    let new_cap = old_cap * 2;
    let new_mask = (new_cap - 1) as u64;

    // Allocate new zeroed slot array (zero ↔ EMPTY_TAG for all hash_tag words).
    let new_words = new_cap * stride;
    let new_ptr = alloc_u32_zeroed(new_words);

    // Rehash all occupied entries from the old table into the new one.
    let old_slots = ts.slots;
    for i in 0..old_cap {
        let tag = unsafe { *old_slots.add(i * stride) };
        if tag == EMPTY_TAG {
            continue;
        }
        // Linear-probe insert into new table using the stored hash tag.
        let mut slot = (tag as u64 & new_mask) as usize;
        loop {
            let existing_tag = unsafe { *new_ptr.add(slot * stride) };
            if existing_tag == EMPTY_TAG {
                // Copy the entire slot (tag + arity fields).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old_slots.add(i * stride),
                        new_ptr.add(slot * stride),
                        stride,
                    );
                }
                break;
            }
            slot = ((slot as u64 + 1) & new_mask) as usize;
        }
    }

    // Free the old slot allocation.
    let old_words = old_cap * stride;
    unsafe { free_u32_slice(old_slots, old_words) };

    // Update the JitTupleSet in place (len is unchanged).
    ts.slots = new_ptr;
    ts.mask = new_mask;
}

// ─── Static offset assertions ────────────────────────────────────────────────
//
// JIT code hardcodes these byte offsets — they must never change.

const _: () = {
    use std::mem::{offset_of, size_of};

    // ── JitColIndex (32 bytes) ────────────────────────────────────────────
    assert!(offset_of!(JitColIndex, keys) == 0);
    assert!(offset_of!(JitColIndex, ranges) == 8);
    assert!(offset_of!(JitColIndex, vals) == 16);
    assert!(offset_of!(JitColIndex, mask) == 24);
    assert!(offset_of!(JitColIndex, len) == 28);
    assert!(size_of::<JitColIndex>() == 32);

    // ── JitTupleSet (24 bytes) ────────────────────────────────────────────
    assert!(offset_of!(JitTupleSet, slots) == 0);
    assert!(offset_of!(JitTupleSet, mask) == 8);
    assert!(offset_of!(JitTupleSet, len) == 16);
    assert!(size_of::<JitTupleSet>() == 24);

    // ── JitSwissTable (24 bytes) ──────────────────────────────────────────
    assert!(offset_of!(JitSwissTable, ctrl) == 0);
    assert!(offset_of!(JitSwissTable, data) == 8);
    assert!(offset_of!(JitSwissTable, mask) == 16);
    assert!(size_of::<JitSwissTable>() == 24);

    // ── JitRelData C-visible region (first 88 bytes) ──────────────────────
    assert!(offset_of!(JitRelData, data) == 0);
    assert!(offset_of!(JitRelData, len) == 8);
    assert!(offset_of!(JitRelData, cap) == 16);
    assert!(offset_of!(JitRelData, col_indices) == 24);
    assert!(offset_of!(JitRelData, tuple_set) == 32);
    assert!(offset_of!(JitRelData, arity) == 56);
    assert!(offset_of!(JitRelData, swiss) == 64);
    // Total struct size is 88 (C region) + size_of::<usize>() (private field).
    assert!(size_of::<JitRelData>() == 88 + size_of::<usize>());
};
