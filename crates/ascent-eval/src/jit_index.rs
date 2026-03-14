#![allow(clippy::items_after_test_module)]
//! Open-addressed hash index for inline JIT probing.
//!
//! Provides a C-visible hash table layout (`JitHashIndex`) that JIT-generated
//! Cranelift code can probe inline without calling back into Rust.
//!
//! Values are stored as a linked-list: each node is two consecutive u32s in
//! the `values` vector:
//!   values[node * 2 + 0] = tuple_idx
//!   values[node * 2 + 1] = next node index (SENTINEL = end of chain)
//!
//! This layout allows O(1) amortized incremental insertion: new tuples are
//! prepended to the chain without rebuilding the full index.

/// One slot in the flat open-addressed entries array.
///
/// `#[repr(C)]` layout (16 bytes):
/// - offset  0: `key`   u32  — EMPTY_KEY = 0xFFFFFFFF means slot is empty
/// - offset  4: `head`  u32  — index of first node in linked-list chain; SENTINEL = no entry
/// - offset  8: `_pad0` u32
/// - offset 12: `_pad1` u32
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JitIndexEntry {
    pub key: u32,
    pub head: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

pub const EMPTY_KEY: u32 = 0xFFFF_FFFF;
pub const SENTINEL: u32 = 0xFFFF_FFFF;

const _: () = {
    assert!(std::mem::size_of::<JitIndexEntry>() == 16);
    assert!(std::mem::offset_of!(JitIndexEntry, key) == 0);
    assert!(std::mem::offset_of!(JitIndexEntry, head) == 4);
    assert!(std::mem::offset_of!(JitIndexEntry, _pad0) == 8);
    assert!(std::mem::offset_of!(JitIndexEntry, _pad1) == 12);
};

impl Default for JitIndexEntry {
    fn default() -> Self {
        JitIndexEntry { key: EMPTY_KEY, head: SENTINEL, _pad0: 0, _pad1: 0 }
    }
}

/// Open-addressed hash index for packed u32 data.
///
/// The first 24 bytes are C-visible (accessible from JIT code):
/// - offset  0: `entries_ptr` `*const JitIndexEntry` — heap-allocated entries
/// - offset  8: `values_ptr`  `*const u32`            — flat linked-list value nodes
/// - offset 16: `mask`        u32                     — capacity − 1 (power of 2)
/// - offset 20: `len`         u32                     — occupied slots (unique keys)
///
/// The `entries` and `values` fields hold ownership (Vec for resizability).
#[repr(C)]
pub struct JitHashIndex {
    pub entries_ptr: *const JitIndexEntry,
    pub values_ptr: *const u32,
    pub mask: u32,
    pub len: u32,
    entries: Vec<JitIndexEntry>,
    values: Vec<u32>,
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
        let mut entries = self.entries.clone();
        let values = self.values.clone();
        let entries_ptr = entries.as_mut_ptr() as *const JitIndexEntry;
        let values_ptr = if values.is_empty() {
            std::ptr::null()
        } else {
            values.as_ptr()
        };
        JitHashIndex {
            entries_ptr,
            values_ptr,
            mask: self.mask,
            len: self.len,
            entries,
            values,
        }
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

        // Upper bound for unique keys = pairs.len(); use 2* for load factor ≤ 0.5.
        let cap = (pairs.len() * 2).next_power_of_two().max(4);
        let mask = (cap - 1) as u32;

        // Allocate cap + 1 entries; index cap is the overflow slot for EMPTY_KEY.
        let mut entries: Vec<JitIndexEntry> = vec![JitIndexEntry::default(); cap + 1];
        let mut values: Vec<u32> = Vec::with_capacity(pairs.len() * 2);
        let mut len: u32 = 0;

        for &(key, tuple_idx) in pairs {
            // Append node (tuple_idx, SENTINEL) temporarily — chain will be updated below.
            let node_idx = (values.len() / 2) as u32;
            values.push(tuple_idx);
            values.push(SENTINEL);

            if key == EMPTY_KEY {
                let slot = &mut entries[cap];
                if slot.head == SENTINEL {
                    // First entry for this key.
                    slot.key = key;
                    slot.head = node_idx;
                    len += 1;
                } else {
                    // Prepend: new node's next = old head, slot.head = new node.
                    values[node_idx as usize * 2 + 1] = slot.head;
                    slot.head = node_idx;
                }
            } else {
                let hash = knuth_hash(key);
                let mut slot_idx = hash & (cap - 1);
                loop {
                    let slot = &mut entries[slot_idx];
                    if slot.key == EMPTY_KEY {
                        // Empty slot: insert new entry.
                        slot.key = key;
                        slot.head = node_idx;
                        len += 1;
                        break;
                    } else if slot.key == key {
                        // Existing entry: prepend new node.
                        values[node_idx as usize * 2 + 1] = slot.head;
                        slot.head = node_idx;
                        break;
                    }
                    slot_idx = (slot_idx + 1) & (cap - 1);
                }
            }
        }

        let entries_ptr = entries.as_ptr();
        let values_ptr = values.as_ptr();

        JitHashIndex {
            entries_ptr,
            values_ptr,
            mask,
            len,
            entries,
            values,
        }
    }

    /// Return an empty index (all probes will find nothing).
    ///
    /// Allocates 2 slots (regular + overflow) both set to EMPTY_KEY / SENTINEL.
    pub fn empty() -> Self {
        let entries = vec![JitIndexEntry::default(); 2];
        let entries_ptr = entries.as_ptr();
        JitHashIndex {
            entries_ptr,
            values_ptr: std::ptr::null(),
            mask: 0,
            len: 0,
            entries,
            values: Vec::new(),
        }
    }

    /// Insert a (key, tuple_idx) pair into this index incrementally.
    ///
    /// O(1) amortized. Rehashes if load factor > 0.5 before adding a new key.
    pub fn insert(&mut self, key: u32, tuple_idx: u32) {
        let node_idx = (self.values.len() / 2) as u32;
        self.values.push(tuple_idx);
        self.values.push(SENTINEL);
        self.values_ptr = self.values.as_ptr();

        let cap = (self.mask as usize) + 1;

        if key == EMPTY_KEY {
            let ovf = &mut self.entries[cap];
            if ovf.head == SENTINEL {
                ovf.key = key;
                ovf.head = node_idx;
                self.len += 1;
            } else {
                self.values[node_idx as usize * 2 + 1] = ovf.head;
                ovf.head = node_idx;
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
                    // Rehash first, then re-probe.
                    self.rehash();
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
                self.entries[slot_idx].key = key;
                self.entries[slot_idx].head = node_idx;
                self.len += 1;
                break;
            } else if slot_key == key {
                // Prepend to existing chain.
                self.values[node_idx as usize * 2 + 1] = self.entries[slot_idx].head;
                self.entries[slot_idx].head = node_idx;
                break;
            }
            slot_idx = (slot_idx + 1) & (cap - 1);
        }

        self.entries_ptr = self.entries.as_ptr();
    }

    /// Clear all entries for a full rebuild (e.g. recent index).
    ///
    /// Resets all slots to empty/sentinel, clears values (keeps capacity),
    /// resets len to 0, and updates raw pointers.
    pub fn clear_for_rebuild(&mut self) {
        let total = self.entries.len();
        for e in self.entries.iter_mut() {
            e.key = EMPTY_KEY;
            e.head = SENTINEL;
            e._pad0 = 0;
            e._pad1 = 0;
        }
        self.values.clear();
        self.len = 0;
        // entries capacity stays; ensure pointer still valid after mutable borrow
        let _ = total; // suppress unused warning
        self.entries_ptr = self.entries.as_ptr();
        self.values_ptr = std::ptr::null();
    }

    /// Double the hash table capacity and re-insert all existing keys.
    ///
    /// Does not move `values` — node indices remain valid.
    fn rehash(&mut self) {
        let old_cap = (self.mask as usize) + 1;
        let new_cap = old_cap * 2;
        let new_mask = (new_cap - 1) as u32;

        let mut new_entries = vec![JitIndexEntry::default(); new_cap + 1];

        for i in 0..=old_cap {
            // old_cap is the overflow slot
            let old = self.entries[i];
            if old.key == EMPTY_KEY && old.head == SENTINEL {
                continue;
            }
            if old.key == EMPTY_KEY {
                // Copy overflow slot as-is.
                new_entries[new_cap] = old;
            } else {
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
        }

        self.entries = new_entries;
        self.mask = new_mask;
        self.entries_ptr = self.entries.as_ptr();
    }
}

/// A 24-byte C-visible handle pointing into a `JitHashIndex`.
///
/// `#[repr(C)]` layout:
/// - offset  0: `entries` `*const JitIndexEntry`  (8 bytes)
/// - offset  8: `values`  `*const u32`            (8 bytes)
/// - offset 16: `mask`    u32                     (4 bytes)
/// - offset 20: `_pad`    u32                     (4 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JitLookupHandle {
    pub entries: *const JitIndexEntry,
    pub values: *const u32,
    pub mask: u32,
    pub _pad: u32,
}

unsafe impl Send for JitLookupHandle {}
unsafe impl Sync for JitLookupHandle {}

const _: () = {
    assert!(std::mem::size_of::<JitLookupHandle>() == 24);
    assert!(std::mem::offset_of!(JitLookupHandle, entries) == 0);
    assert!(std::mem::offset_of!(JitLookupHandle, values) == 8);
    assert!(std::mem::offset_of!(JitLookupHandle, mask) == 16);
    assert!(std::mem::offset_of!(JitLookupHandle, _pad) == 20);
};

impl JitLookupHandle {
    /// Build a handle from an existing index.
    pub fn from_index(idx: &JitHashIndex) -> Self {
        JitLookupHandle {
            entries: idx.entries_ptr,
            values: idx.values_ptr,
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
            head: SENTINEL,
            _pad0: 0,
            _pad1: 0,
        };
        JitLookupHandle {
            entries: &EMPTY_SENTINEL as *const JitIndexEntry,
            values: std::ptr::null(),
            mask: 0,
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk a linked-list chain starting at `head` in `values`, collecting tuple indices.
    fn walk_chain(values: *const u32, head: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut cur = head;
        while cur != SENTINEL {
            let tuple_idx = unsafe { *values.add(cur as usize * 2) };
            let next = unsafe { *values.add(cur as usize * 2 + 1) };
            result.push(tuple_idx);
            cur = next;
        }
        result
    }

    #[test]
    fn test_build_and_probe() {
        let pairs = vec![(1u32, 0u32), (2u32, 1u32), (1u32, 2u32), (3u32, 3u32)];
        let idx = JitHashIndex::build(&pairs);

        // Probe key=1 manually
        let cap = (idx.mask as usize) + 1;
        let hash = knuth_hash(1);
        let mut slot = hash & (idx.mask as usize);
        loop {
            let entry = unsafe { &*idx.entries_ptr.add(slot) };
            if entry.key == EMPTY_KEY {
                panic!("key 1 not found");
            }
            if entry.key == 1 {
                let mut found = walk_chain(idx.values_ptr, entry.head);
                found.sort();
                assert_eq!(found, [0, 2]);
                break;
            }
            slot = (slot + 1) & (cap - 1);
        }
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

        let cap = (idx.mask as usize) + 1;
        let hash = knuth_hash(1);
        let mut slot = hash & (idx.mask as usize);
        loop {
            let entry = unsafe { &*idx.entries_ptr.add(slot) };
            if entry.key == EMPTY_KEY {
                panic!("key 1 not found after incremental insert");
            }
            if entry.key == 1 {
                let mut found = walk_chain(idx.values_ptr, entry.head);
                found.sort();
                assert_eq!(found, [0, 2]);
                break;
            }
            slot = (slot + 1) & (cap - 1);
        }
    }

    #[test]
    fn test_clear_for_rebuild() {
        let mut idx = JitHashIndex::build(&[(1, 0), (2, 1)]);
        idx.clear_for_rebuild();
        assert_eq!(idx.len, 0);
        assert_eq!(idx.values.len(), 0);
        // All entries should be empty.
        for e in &idx.entries {
            assert_eq!(e.key, EMPTY_KEY);
            assert_eq!(e.head, SENTINEL);
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
        let mut found = walk_chain(idx.values_ptr, ovf.head);
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

        // For each unique key, the sets of tuple indices should match.
        let unique_keys: std::collections::BTreeSet<u32> = pairs.iter().map(|&(k, _)| k).collect();
        for key in unique_keys {
            let find_in = |idx: &JitHashIndex| {
                let cap = (idx.mask as usize) + 1;
                let hash = knuth_hash(key);
                let mut slot = hash & (idx.mask as usize);
                loop {
                    let entry = unsafe { &*idx.entries_ptr.add(slot) };
                    if entry.key == EMPTY_KEY { return vec![]; }
                    if entry.key == key {
                        return walk_chain(idx.values_ptr, entry.head);
                    }
                    slot = (slot + 1) & (cap - 1);
                }
            };
            let mut b = find_in(&built);
            let mut i = find_in(&incremental);
            b.sort();
            i.sort();
            assert_eq!(b, i, "mismatch for key {key}");
        }
    }

    #[test]
    fn test_rehash_on_insert() {
        // Insert enough unique keys to trigger rehash.
        let mut idx = JitHashIndex::empty();
        for i in 0..32u32 {
            idx.insert(i, i * 10);
        }
        // All 32 keys must be findable.
        for i in 0..32u32 {
            let cap = (idx.mask as usize) + 1;
            let hash = knuth_hash(i);
            let mut slot = hash & (idx.mask as usize);
            let found = loop {
                let entry = unsafe { &*idx.entries_ptr.add(slot) };
                if entry.key == EMPTY_KEY { break false; }
                if entry.key == i {
                    let chain = walk_chain(idx.values_ptr, entry.head);
                    break chain.contains(&(i * 10));
                }
                slot = (slot + 1) & (cap - 1);
            };
            assert!(found, "key {i} not found after rehash");
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
