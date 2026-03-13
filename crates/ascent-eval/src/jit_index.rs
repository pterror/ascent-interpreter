//! Open-addressed hash index for inline JIT probing.
//!
//! Provides a C-visible hash table layout (`JitHashIndex`) that JIT-generated
//! Cranelift code can probe inline without calling back into Rust.

/// One slot in the flat open-addressed entries array.
///
/// `#[repr(C)]` layout (16 bytes):
/// - offset  0: `key`    u32  — EMPTY_KEY = 0xFFFFFFFF means slot is empty
/// - offset  4: `offset` u32  — index into the flat values array
/// - offset  8: `len`    u32  — number of tuple indices for this key
/// - offset 12: `_pad`   u32
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct JitIndexEntry {
    pub key: u32,
    pub offset: u32,
    pub len: u32,
    pub _pad: u32,
}

pub const EMPTY_KEY: u32 = 0xFFFF_FFFF;

const _: () = {
    assert!(std::mem::size_of::<JitIndexEntry>() == 16);
    assert!(std::mem::offset_of!(JitIndexEntry, key) == 0);
    assert!(std::mem::offset_of!(JitIndexEntry, offset) == 4);
    assert!(std::mem::offset_of!(JitIndexEntry, len) == 8);
    assert!(std::mem::offset_of!(JitIndexEntry, _pad) == 12);
};

/// Open-addressed hash index for packed u32 data.
///
/// The first 24 bytes are C-visible (accessible from JIT code):
/// - offset  0: `entries_ptr` `*const JitIndexEntry` — heap-allocated entries
/// - offset  8: `values_ptr`  `*const u32`            — flat tuple-index values
/// - offset 16: `mask`        u32                     — capacity − 1 (power of 2)
/// - offset 20: `len`         u32                     — occupied slots
///
/// The `_entries` and `_values` fields hold ownership but are not read by JIT.
#[repr(C)]
pub struct JitHashIndex {
    pub entries_ptr: *const JitIndexEntry,
    pub values_ptr: *const u32,
    pub mask: u32,
    pub len: u32,
    // Ownership: keep the allocations alive.
    _entries: Box<[JitIndexEntry]>,
    _values: Box<[u32]>,
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
        let entries = self._entries.clone();
        let values = self._values.clone();
        JitHashIndex {
            entries_ptr: entries.as_ptr(),
            values_ptr: values.as_ptr(),
            mask: self.mask,
            len: self.len,
            _entries: entries,
            _values: values,
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
    pub fn build(pairs: &[(u32, u32)]) -> Self {
        if pairs.is_empty() {
            return Self::empty();
        }

        // Group by key to build (key, [tuple_idx]) map.
        let mut map: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
        for &(key, idx) in pairs {
            map.entry(key).or_default().push(idx);
        }

        let num_keys = map.len();

        // Capacity: smallest power-of-2 >= 2 * num_keys (load factor ≤ 0.5).
        // +1 extra slot at the end for EMPTY_KEY overflow (may be left sentinel).
        let cap = (num_keys * 2).next_power_of_two().max(4);
        let mask = (cap - 1) as u32;

        // Allocate cap + 1 entries; index cap is the overflow slot for EMPTY_KEY.
        let mut entries: Vec<JitIndexEntry> = vec![
            JitIndexEntry {
                key: EMPTY_KEY,
                offset: 0,
                len: 0,
                _pad: 0,
            };
            cap + 1
        ];

        // Build flat values array.
        let total_vals: usize = map.values().map(|v| v.len()).sum();
        let mut values: Vec<u32> = Vec::with_capacity(total_vals);

        for (&key, idxs) in &map {
            let offset = values.len() as u32;
            let len = idxs.len() as u32;
            values.extend_from_slice(idxs);

            if key == EMPTY_KEY {
                // Store in overflow slot (index `cap`).
                entries[cap] = JitIndexEntry { key, offset, len, _pad: 0 };
            } else {
                // Insert into open-addressed table.
                let hash = knuth_hash(key);
                let mut slot = hash & (cap - 1);
                loop {
                    if entries[slot].key == EMPTY_KEY {
                        entries[slot] = JitIndexEntry { key, offset, len, _pad: 0 };
                        break;
                    }
                    slot = (slot + 1) & (cap - 1);
                }
            }
        }

        let entries_box: Box<[JitIndexEntry]> = entries.into_boxed_slice();
        let values_box: Box<[u32]> = values.into_boxed_slice();

        JitHashIndex {
            entries_ptr: entries_box.as_ptr(),
            values_ptr: values_box.as_ptr(),
            mask,
            len: num_keys as u32,
            _entries: entries_box,
            _values: values_box,
        }
    }

    /// Return an empty index (all probes will find nothing).
    ///
    /// Allocates 2 slots (regular + overflow) both set to EMPTY_KEY.
    pub fn empty() -> Self {
        // Two slots: regular slot at index 0, overflow slot at index 1 (= mask + 1 = 1).
        let entries_box: Box<[JitIndexEntry]> = Box::new([
            JitIndexEntry { key: EMPTY_KEY, offset: 0, len: 0, _pad: 0 },
            JitIndexEntry { key: EMPTY_KEY, offset: 0, len: 0, _pad: 0 },
        ]);
        let values_box: Box<[u32]> = Box::new([]);
        JitHashIndex {
            entries_ptr: entries_box.as_ptr(),
            values_ptr: values_box.as_ptr(),
            mask: 0,
            len: 0,
            _entries: entries_box,
            _values: values_box,
        }
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
            offset: 0,
            len: 0,
            _pad: 0,
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
                assert_eq!(entry.len, 2);
                let v0 = unsafe { *idx.values_ptr.add(entry.offset as usize) };
                let v1 = unsafe { *idx.values_ptr.add(entry.offset as usize + 1) };
                let mut found = [v0, v1];
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
}
