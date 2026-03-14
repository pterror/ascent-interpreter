#![allow(clippy::items_after_test_module)]
//! Open-addressed hash index for inline JIT probing.
//!
//! Provides a C-visible hash table layout (`JitHashIndex`) that JIT-generated
//! Cranelift code can probe inline without calling back into Rust.
//!
//! Values are stored as contiguous per-key slices: after the hash probe finds
//! the entry for a key, the inner iteration is a sequential scan over a
//! flat `u32` array.  This eliminates the pointer-chasing of linked-list
//! chains, matching the cache behaviour of `ascent_macro`'s `HashMap<K, Vec<V>>`.

/// One slot in the flat open-addressed entries array.
///
/// `#[repr(C)]` layout (16 bytes):
/// - offset  0: `key`      u32              — EMPTY_KEY = 0xFFFFFFFF means slot is empty
/// - offset  4: `len`      u32              — number of tuple_idxs in this key's slice
/// - offset  8: `data_ptr` *const u32       — pointer to contiguous tuple_idx array
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JitIndexEntry {
    pub key: u32,
    pub len: u32,
    pub data_ptr: *const u32,
}

// Safety: JitIndexEntry contains a raw pointer that we own and never alias concurrently.
unsafe impl Send for JitIndexEntry {}
unsafe impl Sync for JitIndexEntry {}

pub const EMPTY_KEY: u32 = 0xFFFF_FFFF;

const _: () = {
    assert!(std::mem::size_of::<JitIndexEntry>() == 16);
    assert!(std::mem::offset_of!(JitIndexEntry, key) == 0);
    assert!(std::mem::offset_of!(JitIndexEntry, len) == 4);
    assert!(std::mem::offset_of!(JitIndexEntry, data_ptr) == 8);
};

impl Default for JitIndexEntry {
    fn default() -> Self {
        JitIndexEntry { key: EMPTY_KEY, len: 0, data_ptr: std::ptr::null() }
    }
}

/// Open-addressed hash index for packed u32 data.
///
/// Each entry in the hash table stores a `key` and a pointer into a flat
/// contiguous `u32` slice.  After the key lookup, inner-loop iteration is a
/// sequential scan over `data_ptr[0..len]` — no pointer chasing.
///
/// The first 16 bytes are C-visible (accessible from JIT code):
/// - offset  0: `entries_ptr` `*const JitIndexEntry`
/// - offset  8: `mask`        u32   — capacity − 1 (power of 2)
/// - offset 12: `len`         u32   — occupied slots (unique keys)
///
/// Per-key data lives in `vals_vecs`: one `Vec<u32>` per unique key.  The
/// `data_ptr` in each entry points into the corresponding `Vec`; it is
/// refreshed after every mutation that may reallocate a Vec.
#[repr(C)]
pub struct JitHashIndex {
    pub entries_ptr: *const JitIndexEntry,
    pub mask: u32,
    pub len: u32,
    entries: Vec<JitIndexEntry>,
    /// One Vec<u32> per unique key; index = vals_idx stored implicitly via data_ptr.
    /// Stored as (key, Vec<u32>) pairs to support rehashing.
    vals_vecs: Vec<(u32, Vec<u32>)>,
}

// Safety: JitHashIndex contains raw pointers that we own and never alias.
unsafe impl Send for JitHashIndex {}
unsafe impl Sync for JitHashIndex {}

impl std::fmt::Debug for JitHashIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitHashIndex")
            .field("mask", &self.mask)
            .field("len", &self.len)
            .finish()
    }
}

impl Clone for JitHashIndex {
    fn clone(&self) -> Self {
        // Full rebuild from the vals_vecs data.
        let mut new_idx = Self::empty();
        for (key, vals) in &self.vals_vecs {
            for &tuple_idx in vals {
                new_idx.insert(*key, tuple_idx);
            }
        }
        new_idx
    }
}

/// Knuth multiplicative hash for u32 keys.
#[inline]
fn knuth_hash(key: u32) -> usize {
    let h = (key as u64).wrapping_mul(2_654_435_761);
    h as usize
}

impl JitHashIndex {
    /// Build an index from (key, tuple_idx) pairs.
    ///
    /// `pairs` need not be sorted; keys may repeat.
    ///
    /// If key `EMPTY_KEY` (0xFFFFFFFF) appears (e.g. from i32 value -1), it is
    /// stored in a dedicated overflow slot at index `capacity` (one past the end
    /// of the regular table).  The JIT probe handles this via the overflow check
    /// in `empty_exit`.
    #[allow(dead_code)]
    pub fn build(pairs: &[(u32, u32)]) -> Self {
        if pairs.is_empty() {
            return Self::empty();
        }

        // Collect into per-key vecs first.
        let mut key_order: Vec<u32> = Vec::new();
        let mut key_map: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();
        for &(key, tuple_idx) in pairs {
            let entry = key_map.entry(key).or_insert_with(|| {
                key_order.push(key);
                Vec::new()
            });
            entry.push(tuple_idx);
        }

        let n_keys = key_order.len();
        let cap = (n_keys * 2).next_power_of_two().max(4);
        let mask = (cap - 1) as u32;

        let mut entries: Vec<JitIndexEntry> = vec![JitIndexEntry::default(); cap + 1];
        let mut vals_vecs: Vec<(u32, Vec<u32>)> = Vec::with_capacity(n_keys);

        for key in key_order {
            let data = key_map.remove(&key).unwrap();
            vals_vecs.push((key, data));
            let vi = vals_vecs.len() - 1;
            let (_, ref vec) = vals_vecs[vi];
            let data_ptr = vec.as_ptr();
            let len = vec.len() as u32;

            if key == EMPTY_KEY {
                entries[cap].key = key;
                entries[cap].len = len;
                entries[cap].data_ptr = data_ptr;
            } else {
                let hash = knuth_hash(key);
                let mut slot_idx = hash & (cap - 1);
                loop {
                    if entries[slot_idx].key == EMPTY_KEY {
                        entries[slot_idx].key = key;
                        entries[slot_idx].len = len;
                        entries[slot_idx].data_ptr = data_ptr;
                        break;
                    }
                    slot_idx = (slot_idx + 1) & (cap - 1);
                }
            }
        }

        let entries_ptr = entries.as_ptr();
        let len = vals_vecs.len() as u32;

        JitHashIndex { entries_ptr, mask, len, entries, vals_vecs }
    }

    /// Return an empty index (all probes will find nothing).
    ///
    /// Allocates 2 slots (regular + overflow) both set to EMPTY_KEY.
    pub fn empty() -> Self {
        let entries = vec![JitIndexEntry::default(); 2];
        let entries_ptr = entries.as_ptr();
        JitHashIndex {
            entries_ptr,
            mask: 0,
            len: 0,
            entries,
            vals_vecs: Vec::new(),
        }
    }

    /// Insert a (key, tuple_idx) pair into this index incrementally.
    ///
    /// O(1) amortized. Rehashes if load factor > 0.5 before adding a new key.
    pub fn insert(&mut self, key: u32, tuple_idx: u32) {
        let cap = (self.mask as usize) + 1;

        if key == EMPTY_KEY {
            let ovf = &mut self.entries[cap];
            if ovf.key == EMPTY_KEY && ovf.data_ptr.is_null() {
                // New key: allocate a new vec.
                self.vals_vecs.push((key, vec![tuple_idx]));
                let vi = self.vals_vecs.len() - 1;
                let (_, ref vec) = self.vals_vecs[vi];
                ovf.key = key;
                ovf.len = 1;
                ovf.data_ptr = vec.as_ptr();
                self.len += 1;
            } else {
                // Existing key: find its vec and push.
                // The overflow slot's data_ptr points into vals_vecs; find it.
                let vi = self.find_vals_idx_for_key(EMPTY_KEY).unwrap();
                self.vals_vecs[vi].1.push(tuple_idx);
                let (_, ref vec) = self.vals_vecs[vi];
                self.entries[cap].len = vec.len() as u32;
                self.entries[cap].data_ptr = vec.as_ptr();
            }
            self.entries_ptr = self.entries.as_ptr();
            return;
        }

        // Probe to find existing slot or empty slot.
        let hash = knuth_hash(key);
        let mut slot_idx = hash & (cap - 1);
        loop {
            let slot_key = self.entries[slot_idx].key;
            if slot_key == EMPTY_KEY {
                // Check load factor before inserting new key.
                if self.len as usize >= cap / 2 {
                    self.rehash();
                    // After rehash, re-probe.
                    let new_cap = (self.mask as usize) + 1;
                    let new_hash = knuth_hash(key);
                    slot_idx = new_hash & (new_cap - 1);
                    loop {
                        if self.entries[slot_idx].key == EMPTY_KEY {
                            break;
                        }
                        slot_idx = (slot_idx + 1) & (new_cap - 1);
                    }
                }
                // New key: allocate a new vec.
                self.vals_vecs.push((key, vec![tuple_idx]));
                let vi = self.vals_vecs.len() - 1;
                let (_, ref vec) = self.vals_vecs[vi];
                self.entries[slot_idx].key = key;
                self.entries[slot_idx].len = 1;
                self.entries[slot_idx].data_ptr = vec.as_ptr();
                self.len += 1;
                break;
            } else if slot_key == key {
                // Existing key: push to its vec.
                let vi = self.find_vals_idx_for_key(key).unwrap();
                self.vals_vecs[vi].1.push(tuple_idx);
                let (_, ref vec) = self.vals_vecs[vi];
                self.entries[slot_idx].len = vec.len() as u32;
                self.entries[slot_idx].data_ptr = vec.as_ptr();
                break;
            }
            slot_idx = (slot_idx + 1) & (cap - 1);
        }

        self.entries_ptr = self.entries.as_ptr();
    }

    /// Find the index into `vals_vecs` for the given key.
    fn find_vals_idx_for_key(&self, key: u32) -> Option<usize> {
        for (i, (k, _)) in self.vals_vecs.iter().enumerate() {
            if *k == key {
                return Some(i);
            }
        }
        None
    }

    /// Clear all entries for a full rebuild (e.g. recent index).
    ///
    /// Resets all slots to empty, clears vals_vecs, resets len to 0.
    pub fn clear_for_rebuild(&mut self) {
        for e in self.entries.iter_mut() {
            e.key = EMPTY_KEY;
            e.len = 0;
            e.data_ptr = std::ptr::null();
        }
        self.vals_vecs.clear();
        self.len = 0;
        self.entries_ptr = self.entries.as_ptr();
    }

    /// Double the hash table capacity and re-insert all existing entries.
    fn rehash(&mut self) {
        let old_cap = (self.mask as usize) + 1;
        let new_cap = old_cap * 2;
        let new_mask = (new_cap - 1) as u32;

        let mut new_entries = vec![JitIndexEntry::default(); new_cap + 1];

        // Copy overflow slot as-is (index old_cap → new_cap).
        let ovf = self.entries[old_cap];
        if ovf.key == EMPTY_KEY && !ovf.data_ptr.is_null() {
            new_entries[new_cap] = ovf;
        }

        // Re-insert regular entries.
        for i in 0..old_cap {
            let old = self.entries[i];
            if old.key == EMPTY_KEY {
                continue;
            }
            let hash = knuth_hash(old.key);
            let mut slot = hash & (new_cap - 1);
            loop {
                if new_entries[slot].key == EMPTY_KEY {
                    new_entries[slot] = old;
                    break;
                }
                slot = (slot + 1) & (new_cap - 1);
            }
        }

        self.entries = new_entries;
        self.mask = new_mask;
        self.entries_ptr = self.entries.as_ptr();
    }
}

/// A 16-byte C-visible handle pointing into a `JitHashIndex`.
///
/// `#[repr(C)]` layout:
/// - offset  0: `entries` `*const JitIndexEntry`  (8 bytes)
/// - offset  8: `mask`    u32                     (4 bytes)
/// - offset 12: `_pad`    u32                     (4 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JitLookupHandle {
    pub entries: *const JitIndexEntry,
    pub mask: u32,
    pub _pad: u32,
}

unsafe impl Send for JitLookupHandle {}
unsafe impl Sync for JitLookupHandle {}

const _: () = {
    assert!(std::mem::size_of::<JitLookupHandle>() == 16);
    assert!(std::mem::offset_of!(JitLookupHandle, entries) == 0);
    assert!(std::mem::offset_of!(JitLookupHandle, mask) == 8);
};

impl JitLookupHandle {
    /// Build a handle from an existing index.
    pub fn from_index(idx: &JitHashIndex) -> Self {
        JitLookupHandle {
            entries: idx.entries_ptr,
            mask: idx.mask,
            _pad: 0,
        }
    }

    /// Null handle — all probes will find no match (empty sentinel immediately).
    pub fn null() -> Self {
        // We still need a valid entries pointer that starts with EMPTY_KEY.
        // Use a static empty sentinel.
        static EMPTY_SENTINEL: JitIndexEntry = JitIndexEntry {
            key: EMPTY_KEY,
            len: 0,
            data_ptr: std::ptr::null(),
        };
        JitLookupHandle {
            entries: &EMPTY_SENTINEL as *const JitIndexEntry,
            mask: 0,
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect all tuple indices stored for a given key by direct entry lookup.
    fn collect_key_data(entries_ptr: *const JitIndexEntry, mask: u32, key: u32) -> Vec<u32> {
        let cap = (mask as usize) + 1;
        // Try regular slots first
        let hash = knuth_hash(key);
        let mut slot = hash & (cap - 1);
        loop {
            let entry = unsafe { &*entries_ptr.add(slot) };
            if entry.key == EMPTY_KEY {
                // Not found in regular slots; check overflow slot
                break;
            }
            if entry.key == key {
                let slice = unsafe {
                    std::slice::from_raw_parts(entry.data_ptr, entry.len as usize)
                };
                return slice.to_vec();
            }
            slot = (slot + 1) & (cap - 1);
        }
        // Check overflow slot at index cap
        let ovf = unsafe { &*entries_ptr.add(cap) };
        if ovf.key == key && !ovf.data_ptr.is_null() {
            let slice = unsafe {
                std::slice::from_raw_parts(ovf.data_ptr, ovf.len as usize)
            };
            return slice.to_vec();
        }
        vec![]
    }

    #[test]
    fn test_build_and_probe() {
        let pairs = vec![(1u32, 0u32), (2u32, 1u32), (1u32, 2u32), (3u32, 3u32)];
        let idx = JitHashIndex::build(&pairs);

        let mut found = collect_key_data(idx.entries_ptr, idx.mask, 1);
        found.sort();
        assert_eq!(found, [0, 2]);
    }

    #[test]
    fn test_empty() {
        let idx = JitHashIndex::empty();
        let entry = unsafe { &*idx.entries_ptr };
        assert_eq!(entry.key, EMPTY_KEY);
    }

    #[test]
    fn test_insert_incremental() {
        let mut idx = JitHashIndex::empty();
        idx.insert(1, 0);
        idx.insert(2, 1);
        idx.insert(1, 2);
        idx.insert(3, 3);

        let mut found = collect_key_data(idx.entries_ptr, idx.mask, 1);
        found.sort();
        assert_eq!(found, [0, 2]);
    }

    #[test]
    fn test_clear_for_rebuild() {
        let mut idx = JitHashIndex::build(&[(1, 0), (2, 1)]);
        idx.clear_for_rebuild();
        assert_eq!(idx.len, 0);
        assert_eq!(idx.vals_vecs.len(), 0);
        // All entries should be empty.
        for e in &idx.entries {
            assert_eq!(e.key, EMPTY_KEY);
            assert_eq!(e.len, 0);
            assert!(e.data_ptr.is_null());
        }
    }

    #[test]
    fn test_overflow_key() {
        // EMPTY_KEY = 0xFFFFFFFF = i32 value -1 should be stored in overflow slot.
        let pairs = vec![(EMPTY_KEY, 7u32), (1u32, 0u32), (EMPTY_KEY, 8u32)];
        let idx = JitHashIndex::build(&pairs);
        let cap = (idx.mask as usize) + 1;
        let ovf = unsafe { &*idx.entries_ptr.add(cap) };
        assert_eq!(ovf.key, EMPTY_KEY);
        assert_eq!(ovf.len, 2);
        let mut found: Vec<u32> = unsafe {
            std::slice::from_raw_parts(ovf.data_ptr, ovf.len as usize)
        }.to_vec();
        found.sort();
        assert_eq!(found, [7, 8]);
    }

    #[test]
    fn test_build_matches_incremental() {
        let pairs: Vec<(u32, u32)> = vec![(5, 0), (3, 1), (5, 2), (7, 3), (3, 4)];

        let built = JitHashIndex::build(&pairs);
        let mut incremental = JitHashIndex::empty();
        for &(k, v) in &pairs {
            incremental.insert(k, v);
        }

        // Both should have same unique key count.
        assert_eq!(built.len, incremental.len);

        let unique_keys: std::collections::BTreeSet<u32> = pairs.iter().map(|&(k, _)| k).collect();
        for key in unique_keys {
            let mut b = collect_key_data(built.entries_ptr, built.mask, key);
            let mut i = collect_key_data(incremental.entries_ptr, incremental.mask, key);
            b.sort();
            i.sort();
            assert_eq!(b, i, "mismatch for key {key}");
        }
    }

    #[test]
    fn test_rehash_on_insert() {
        let mut idx = JitHashIndex::empty();
        for i in 0..32u32 {
            idx.insert(i, i * 10);
        }
        for i in 0..32u32 {
            let found = collect_key_data(idx.entries_ptr, idx.mask, i);
            assert!(found.contains(&(i * 10)), "key {i} not found after rehash");
        }
    }

    #[test]
    fn test_sequential_data_access() {
        // Verify that data_ptr values are contiguous and all accessible.
        let mut idx = JitHashIndex::empty();
        for i in 0u32..5 {
            idx.insert(42, i);
        }
        let found = collect_key_data(idx.entries_ptr, idx.mask, 42);
        let mut sorted = found.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        // Verify contiguity by checking the data_ptr directly.
        let cap = (idx.mask as usize) + 1;
        let hash = knuth_hash(42);
        let mut slot = hash & (cap - 1);
        loop {
            let entry = unsafe { &*idx.entries_ptr.add(slot) };
            if entry.key == 42 {
                assert_eq!(entry.len, 5);
                // Sequential load test
                for j in 0..5usize {
                    let v = unsafe { *entry.data_ptr.add(j) };
                    assert!(v < 5, "unexpected value {v} at index {j}");
                }
                break;
            }
            slot = (slot + 1) & (cap - 1);
        }
    }
}

// ─── JIT-accessible dedup hash table ────────────────────────────────────────

/// Sentinel value: a slot whose hash field equals `JITDEDUP_EMPTY` is vacant.
pub const JITDEDUP_EMPTY: u32 = 0xFFFF_FFFF;

/// Compute the JIT dedup hash for a packed u32 tuple.
///
/// Uses a simple multiply-accumulate to keep the JIT-side implementation to a
/// handful of `imul_imm` + `iadd` instructions — no length prefix, no FxHasher
/// complexity.  The output `0xFFFF_FFFF` is remapped to `0xFFFF_FFFE` so it
/// cannot collide with the empty-slot sentinel.
///
/// **Must match the hash emitted by the JIT in `gen_emit_heads_v3`.**
pub fn jit_dedup_hash(packed: &[u32]) -> u32 {
    let mut h: u32 = 0;
    for &v in packed {
        h = h.wrapping_mul(0x9e3779b9).wrapping_add(v);
    }
    if h == JITDEDUP_EMPTY { h = JITDEDUP_EMPTY - 1; }
    h
}

/// JIT-visible handle for the dedup table (16 bytes, `repr(C)`).
///
/// The JIT code reads `entries` and `cap` from this handle.  The handle is
/// embedded inside `JitDedupTable` and its address is stable for the lifetime
/// of the owning `PackedStorage`.  After each grow, only the *contents* of
/// `entries` and `cap` are updated in-place; the handle address itself never
/// changes.
///
/// Flat slot layout: each slot is `stride = arity + 1` consecutive `u32`s:
///   `slot[0]`          = hash (`JITDEDUP_EMPTY` = vacant)
///   `slot[1..stride]`  = packed tuple data (`arity` u32s)
///
/// The JIT knows `stride` at compile time (it equals `head.args.len() + 1`).
#[repr(C)]
pub struct JitDedupHandle {
    /// Pointer to the flat slot array; `null` when `cap == 0`.
    pub entries: *mut u32,
    /// Number of slots (always a power of two; mask = cap − 1).
    pub cap: u32,
    pub _pad: u32,
}

unsafe impl Send for JitDedupHandle {}
unsafe impl Sync for JitDedupHandle {}

const _: () = {
    assert!(std::mem::size_of::<JitDedupHandle>() == 16);
    assert!(std::mem::offset_of!(JitDedupHandle, entries) == 0);
    assert!(std::mem::offset_of!(JitDedupHandle, cap) == 8);
};

/// Rust-managed owner of the JIT dedup hash table.
///
/// All mutation happens exclusively in `update_jit_indices()` (called from
/// `advance()`).  JIT-generated code reads the table through the embedded
/// `handle` field but **never writes to it**, so the `entries` pointer loaded
/// at rule-variant function entry is stable for the entire duration of that
/// call.
pub struct JitDedupTable {
    /// Flat slot storage; stride = arity + 1 u32s per slot.
    entries_vec: Vec<u32>,
    /// Stable handle embedded in this struct — address never changes.
    pub handle: JitDedupHandle,
    /// Slot width in u32s: `1` (hash) + `arity` (data words).
    stride: usize,
    /// Number of occupied (non-empty) slots.
    count: usize,
}

unsafe impl Send for JitDedupTable {}
unsafe impl Sync for JitDedupTable {}

impl std::fmt::Debug for JitDedupTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitDedupTable")
            .field("count", &self.count)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl Clone for JitDedupTable {
    fn clone(&self) -> Self {
        // Rebuild from scratch — the clone starts empty; it will be repopulated
        // by `update_jit_indices` on the next iteration.
        Self::new(self.stride.saturating_sub(1))
    }
}

impl JitDedupTable {
    pub fn new(arity: usize) -> Self {
        Self {
            entries_vec: Vec::new(),
            handle: JitDedupHandle { entries: std::ptr::null_mut(), cap: 0, _pad: 0 },
            stride: arity + 1,
            count: 0,
        }
    }

    /// Probe the table for a packed tuple. Returns true if present.
    pub fn probe(&self, hash: u32, packed: &[u32]) -> bool {
        debug_assert_eq!(packed.len() + 1, self.stride);
        let cap = self.handle.cap as usize;
        if cap == 0 { return false; }
        let mask = cap - 1;
        let mut slot = (hash as usize) & mask;
        loop {
            let base = slot * self.stride;
            let h = self.entries_vec[base];
            if h == JITDEDUP_EMPTY { return false; }
            if h == hash && self.entries_vec[base + 1..base + self.stride] == *packed {
                return true;
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Probe then insert atomically. Returns true if new (inserted), false if duplicate.
    ///
    /// May reallocate; updates `handle.entries` in place so the stable handle address
    /// remains valid for JIT code that reloads from the handle on each invocation.
    pub fn insert_if_new(&mut self, hash: u32, packed: &[u32]) -> bool {
        debug_assert_eq!(packed.len() + 1, self.stride);
        let cap = self.handle.cap as usize;
        if cap > 0 {
            let mask = cap - 1;
            let mut slot = (hash as usize) & mask;
            loop {
                let base = slot * self.stride;
                let h = self.entries_vec[base];
                if h == JITDEDUP_EMPTY { break; }
                if h == hash && self.entries_vec[base + 1..base + self.stride] == *packed {
                    return false;
                }
                slot = (slot + 1) & mask;
            }
        }
        // Not found — grow if needed then insert.
        self.maybe_grow();
        let cap = self.handle.cap as usize;
        let mask = cap - 1;
        let mut slot = (hash as usize) & mask;
        loop {
            let base = slot * self.stride;
            if self.entries_vec[base] == JITDEDUP_EMPTY {
                self.entries_vec[base] = hash;
                self.entries_vec[base + 1..base + self.stride].copy_from_slice(packed);
                self.count += 1;
                return true;
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Reset to empty without deallocating. Handle address and capacity are preserved.
    pub fn clear(&mut self) {
        self.entries_vec.fill(JITDEDUP_EMPTY);
        self.count = 0;
    }

    /// Insert a packed tuple (unconditional — caller guarantees no duplicate).
    pub fn insert(&mut self, hash: u32, packed: &[u32]) {
        debug_assert_eq!(packed.len() + 1, self.stride);
        self.maybe_grow();
        let cap = self.entries_vec.len() / self.stride;
        let mask = cap - 1;
        let mut slot = (hash as usize) & mask;
        loop {
            let base = slot * self.stride;
            if self.entries_vec[base] == JITDEDUP_EMPTY {
                self.entries_vec[base] = hash;
                self.entries_vec[base + 1..base + self.stride].copy_from_slice(packed);
                self.count += 1;
                return;
            }
            slot = (slot + 1) & mask;
        }
    }

    fn maybe_grow(&mut self) {
        let cur_cap = if self.stride == 0 { 0 } else { self.entries_vec.len() / self.stride };
        // Grow when load factor > 75 %.
        if cur_cap == 0 || (self.count + 1) * 4 > cur_cap * 3 {
            let new_cap = if cur_cap == 0 { 16 } else { cur_cap * 2 };
            let stride = self.stride;
            let mut new_entries = vec![JITDEDUP_EMPTY; new_cap * stride];
            let mask = new_cap - 1;
            for i in 0..cur_cap {
                let base = i * stride;
                let h = self.entries_vec[base];
                if h != JITDEDUP_EMPTY {
                    let mut slot = (h as usize) & mask;
                    loop {
                        let nb = slot * stride;
                        if new_entries[nb] == JITDEDUP_EMPTY {
                            new_entries[nb..nb + stride]
                                .copy_from_slice(&self.entries_vec[base..base + stride]);
                            break;
                        }
                        slot = (slot + 1) & mask;
                    }
                }
            }
            self.entries_vec = new_entries;
            // Update handle in-place so the stable pointer sees fresh data.
            self.handle.entries = self.entries_vec.as_mut_ptr();
            self.handle.cap = new_cap as u32;
        }
    }
}
