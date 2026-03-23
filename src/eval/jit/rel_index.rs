//! Incrementally-insertable hash-map column index.
//!
//! `RelIndex` maps `u32` keys to growable `u32` value arrays.  Unlike
//! `JitColIndex` (bulk-built, immutable after construction), `RelIndex`
//! supports incremental `insert` with automatic hash-table and per-key
//! value-array growth.
//!
//! Both structs are `#[repr(C)]` with static-asserted offsets so that
//! JIT-generated x86-64 code can address fields directly.

#![allow(dead_code)]

use std::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use std::ptr;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Sentinel key value meaning "empty bucket".
pub const EMPTY_KEY: u32 = u32::MAX;

/// Initial per-key values capacity.
const INITIAL_VALS_CAP: u32 = 4;

// ─── Hash ───────────────────────────────────────────────────────────────────

/// 32-bit Knuth multiplicative hash — same as `JitColIndex` / `col_hash`.
#[inline(always)]
fn rel_hash(key: u32) -> u32 {
    key.wrapping_mul(0x9e3779b9)
}

// ─── RelIndexBucket ─────────────────────────────────────────────────────────

/// A single bucket in the `RelIndex` hash table.
///
/// 32 bytes total — enables `slot << 5` addressing in JIT code.
#[repr(C)]
pub struct RelIndexBucket {
    /// Hash key; `EMPTY_KEY` (0xFFFF_FFFF) = empty bucket.
    pub key: u32,
    /// Number of values stored.
    pub count: u32,
    /// Capacity of the `vals` allocation.
    pub cap: u32,
    /// Padding.
    pub _pad: u32,
    /// Pointer to heap-allocated value array.
    pub vals: *mut u32,
    /// Padding to 32 bytes total.
    pub _pad2: u64,
}

// ─── RelIndex ───────────────────────────────────────────────────────────────

/// Incrementally-insertable hash-map column index.
///
/// 16 bytes, `#[repr(C)]`.
#[repr(C)]
pub struct RelIndex {
    /// Pointer to the bucket array.
    pub buckets: *mut RelIndexBucket,
    /// `capacity - 1` (capacity is always a power of 2).
    pub mask: u32,
    /// Number of occupied buckets (distinct keys).
    pub len: u32,
}

// Safety: RelIndex is only used from a single JIT-callback thread.
unsafe impl Send for RelIndex {}
unsafe impl Sync for RelIndex {}

// ─── Allocation helpers ─────────────────────────────────────────────────────

/// Allocate `count` zeroed `RelIndexBucket` values.
fn alloc_buckets_zeroed(count: usize) -> *mut RelIndexBucket {
    assert!(count > 0);
    let layout = Layout::array::<RelIndexBucket>(count).expect("layout overflow");
    let ptr = unsafe { alloc_zeroed(layout) } as *mut RelIndexBucket;
    assert!(!ptr.is_null(), "allocation failed");
    ptr
}

/// Free a bucket array previously allocated with `alloc_buckets_zeroed`.
unsafe fn free_buckets(ptr: *mut RelIndexBucket, count: usize) {
    if count == 0 {
        return;
    }
    let layout = Layout::array::<RelIndexBucket>(count).expect("layout overflow");
    unsafe { dealloc(ptr as *mut u8, layout) };
}

/// Allocate `count` zeroed `u32` values.
fn alloc_u32_zeroed(count: usize) -> *mut u32 {
    if count == 0 {
        return ptr::NonNull::dangling().as_ptr();
    }
    let layout = Layout::array::<u32>(count).expect("layout overflow");
    let ptr = unsafe { alloc_zeroed(layout) } as *mut u32;
    assert!(!ptr.is_null(), "allocation failed");
    ptr
}

/// Free a `u32` slice previously allocated with `alloc_u32_zeroed`.
unsafe fn free_u32_slice(ptr: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    let layout = Layout::array::<u32>(count).expect("layout overflow");
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

// ─── RelIndex implementation ────────────────────────────────────────────────

impl RelIndex {
    /// Create a new `RelIndex` with `initial_cap` empty buckets.
    ///
    /// `initial_cap` is rounded up to the next power of 2 (minimum 16).
    pub fn new_empty(initial_cap: u32) -> Self {
        let cap = next_pow2_min16(initial_cap as usize);
        let buckets = alloc_buckets_zeroed(cap);
        // Zero-init sets all bytes to 0. We need key = EMPTY_KEY for each bucket.
        for i in 0..cap {
            unsafe {
                (*buckets.add(i)).key = EMPTY_KEY;
            }
        }
        RelIndex {
            buckets,
            mask: (cap - 1) as u32,
            len: 0,
        }
    }

    /// Capacity (number of hash-table slots).
    #[inline]
    pub fn cap(&self) -> usize {
        (self.mask as usize) + 1
    }

    /// Probe for `key`. Returns `(vals_ptr, count)` if found, or `(null, 0)`.
    pub fn probe(&self, key: u32) -> (*const u32, u32) {
        let mask = self.mask;
        let mut slot = (rel_hash(key) & mask) as usize;
        loop {
            let bucket = unsafe { &*self.buckets.add(slot) };
            if bucket.key == key {
                return (bucket.vals as *const u32, bucket.count);
            }
            if bucket.key == EMPTY_KEY {
                return (ptr::null(), 0);
            }
            slot = (slot + 1) & (mask as usize);
        }
    }

    /// Insert `(key, val)`. Grows the hash table if load exceeds 75%.
    /// Grows the per-key vals array if full.
    pub fn insert(&mut self, key: u32, val: u32) {
        debug_assert!(key != EMPTY_KEY, "cannot insert EMPTY_KEY sentinel");

        // Check if we need to grow the hash table first (load > 75%).
        let cap = self.cap();
        if (self.len as usize + 1) * 4 > cap * 3 {
            self.grow_table();
        }

        let mask = self.mask;
        let mut slot = (rel_hash(key) & mask) as usize;
        loop {
            let bucket = unsafe { &mut *self.buckets.add(slot) };
            if bucket.key == key {
                // Key exists — append value.
                if bucket.count >= bucket.cap {
                    self.grow_vals(slot);
                    // Re-borrow after potential realloc.
                    let bucket = unsafe { &mut *self.buckets.add(slot) };
                    unsafe {
                        *bucket.vals.add(bucket.count as usize) = val;
                    }
                    bucket.count += 1;
                } else {
                    unsafe {
                        *bucket.vals.add(bucket.count as usize) = val;
                    }
                    bucket.count += 1;
                }
                return;
            }
            if bucket.key == EMPTY_KEY {
                // New key — allocate vals array.
                let vals = alloc_u32_zeroed(INITIAL_VALS_CAP as usize);
                unsafe {
                    *vals = val;
                }
                bucket.key = key;
                bucket.count = 1;
                bucket.cap = INITIAL_VALS_CAP;
                bucket.vals = vals;
                self.len += 1;
                return;
            }
            slot = (slot + 1) & (mask as usize);
        }
    }

    /// Grow the vals array for the bucket at `slot` (double capacity).
    fn grow_vals(&mut self, slot: usize) {
        let bucket = unsafe { &mut *self.buckets.add(slot) };
        let old_cap = bucket.cap as usize;
        let new_cap = old_cap * 2;
        let old_layout = Layout::array::<u32>(old_cap).expect("layout overflow");
        let new_size = new_cap * std::mem::size_of::<u32>();
        let new_ptr = unsafe { realloc(bucket.vals as *mut u8, old_layout, new_size) } as *mut u32;
        assert!(!new_ptr.is_null(), "realloc failed");
        // Zero the new portion.
        unsafe {
            ptr::write_bytes(new_ptr.add(old_cap), 0, new_cap - old_cap);
        }
        bucket.vals = new_ptr;
        bucket.cap = new_cap as u32;
    }

    /// Double the hash table capacity and rehash all entries.
    fn grow_table(&mut self) {
        let old_cap = self.cap();
        let new_cap = old_cap * 2;
        let new_mask = (new_cap - 1) as u32;
        let new_buckets = alloc_buckets_zeroed(new_cap);
        // Mark all new buckets as empty.
        for i in 0..new_cap {
            unsafe {
                (*new_buckets.add(i)).key = EMPTY_KEY;
            }
        }
        // Rehash existing entries.
        for i in 0..old_cap {
            let old_bucket = unsafe { &*self.buckets.add(i) };
            if old_bucket.key == EMPTY_KEY {
                continue;
            }
            let mut slot = (rel_hash(old_bucket.key) & new_mask) as usize;
            loop {
                let new_bucket = unsafe { &mut *new_buckets.add(slot) };
                if new_bucket.key == EMPTY_KEY {
                    // Move the bucket data (including vals pointer ownership).
                    new_bucket.key = old_bucket.key;
                    new_bucket.count = old_bucket.count;
                    new_bucket.cap = old_bucket.cap;
                    new_bucket.vals = old_bucket.vals;
                    break;
                }
                slot = (slot + 1) & (new_mask as usize);
            }
        }
        // Free old bucket array (but NOT per-key vals — they were moved).
        unsafe {
            free_buckets(self.buckets, old_cap);
        }
        self.buckets = new_buckets;
        self.mask = new_mask;
    }

    /// Clear all buckets, freeing per-key vals allocations.
    /// The hash table itself is kept at its current capacity.
    pub fn clear(&mut self) {
        let cap = self.cap();
        for i in 0..cap {
            let bucket = unsafe { &mut *self.buckets.add(i) };
            if bucket.key != EMPTY_KEY {
                unsafe {
                    free_u32_slice(bucket.vals, bucket.cap as usize);
                }
                bucket.key = EMPTY_KEY;
                bucket.count = 0;
                bucket.cap = 0;
                bucket.vals = ptr::null_mut();
            }
        }
        self.len = 0;
    }

    /// Bulk-construct a `RelIndex` from `(key, val)` pairs.
    pub fn build_from_pairs(pairs: &[(u32, u32)]) -> Self {
        // Estimate capacity: at most pairs.len() distinct keys.
        let estimated_keys = pairs.len();
        let cap = next_pow2_min16((estimated_keys * 10 / 7) + 1);
        let mut idx = RelIndex {
            buckets: alloc_buckets_zeroed(cap),
            mask: (cap - 1) as u32,
            len: 0,
        };
        // Mark all buckets empty.
        for i in 0..cap {
            unsafe {
                (*idx.buckets.add(i)).key = EMPTY_KEY;
            }
        }
        for &(key, val) in pairs {
            idx.insert(key, val);
        }
        idx
    }
}

impl Drop for RelIndex {
    fn drop(&mut self) {
        if self.buckets.is_null() {
            return;
        }
        let cap = self.cap();
        // Free each bucket's vals allocation.
        for i in 0..cap {
            let bucket = unsafe { &*self.buckets.add(i) };
            if bucket.key != EMPTY_KEY && bucket.cap > 0 {
                unsafe {
                    free_u32_slice(bucket.vals, bucket.cap as usize);
                }
            }
        }
        // Free the bucket array itself.
        unsafe {
            free_buckets(self.buckets, cap);
        }
        self.buckets = ptr::null_mut();
    }
}

// ─── Static offset assertions ───────────────────────────────────────────────

const _: () = {
    use std::mem::{offset_of, size_of};

    // ── RelIndexBucket (32 bytes) ────────────────────────────────────────
    assert!(offset_of!(RelIndexBucket, key) == 0);
    assert!(offset_of!(RelIndexBucket, count) == 4);
    assert!(offset_of!(RelIndexBucket, cap) == 8);
    assert!(offset_of!(RelIndexBucket, _pad) == 12);
    assert!(offset_of!(RelIndexBucket, vals) == 16);
    assert!(offset_of!(RelIndexBucket, _pad2) == 24);
    assert!(size_of::<RelIndexBucket>() == 32);

    // ── RelIndex (16 bytes) ─────────────────────────────────────────────
    assert!(offset_of!(RelIndex, buckets) == 0);
    assert!(offset_of!(RelIndex, mask) == 8);
    assert!(offset_of!(RelIndex, len) == 12);
    assert!(size_of::<RelIndex>() == 16);
};

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_probe() {
        let mut idx = RelIndex::new_empty(16);
        idx.insert(10, 100);
        idx.insert(10, 200);
        idx.insert(20, 300);

        let (ptr, count) = idx.probe(10);
        assert!(!ptr.is_null());
        assert_eq!(count, 2);
        let vals: Vec<u32> = (0..count).map(|i| unsafe { *ptr.add(i as usize) }).collect();
        assert_eq!(vals, vec![100, 200]);

        let (ptr, count) = idx.probe(20);
        assert!(!ptr.is_null());
        assert_eq!(count, 1);
        assert_eq!(unsafe { *ptr }, 300);
    }

    #[test]
    fn test_probe_missing() {
        let idx = RelIndex::new_empty(16);
        let (ptr, count) = idx.probe(42);
        assert!(ptr.is_null());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_growth() {
        let mut idx = RelIndex::new_empty(16);
        // Insert enough distinct keys to trigger hash table growth (>75% of 16 = 12).
        for k in 0..20u32 {
            idx.insert(k, k * 10);
        }
        assert_eq!(idx.len, 20);
        // Verify all keys survived.
        for k in 0..20u32 {
            let (ptr, count) = idx.probe(k);
            assert!(!ptr.is_null(), "key {k} missing after growth");
            assert_eq!(count, 1);
            assert_eq!(unsafe { *ptr }, k * 10);
        }
    }

    #[test]
    fn test_per_key_growth() {
        let mut idx = RelIndex::new_empty(16);
        // Insert many values for the same key to trigger per-key vals growth.
        // INITIAL_VALS_CAP = 4, so inserting 20 values should trigger multiple growths.
        for v in 0..20u32 {
            idx.insert(1, v);
        }
        let (ptr, count) = idx.probe(1);
        assert!(!ptr.is_null());
        assert_eq!(count, 20);
        let vals: Vec<u32> = (0..count).map(|i| unsafe { *ptr.add(i as usize) }).collect();
        let expected: Vec<u32> = (0..20).collect();
        assert_eq!(vals, expected);
    }

    #[test]
    fn test_build_from_pairs() {
        let pairs = vec![(5, 50), (5, 51), (10, 100), (15, 150), (10, 101)];
        let idx = RelIndex::build_from_pairs(&pairs);
        assert_eq!(idx.len, 3);

        let (ptr, count) = idx.probe(5);
        assert_eq!(count, 2);
        let vals: Vec<u32> = (0..count).map(|i| unsafe { *ptr.add(i as usize) }).collect();
        assert_eq!(vals, vec![50, 51]);

        let (ptr, count) = idx.probe(10);
        assert_eq!(count, 2);
        let vals: Vec<u32> = (0..count).map(|i| unsafe { *ptr.add(i as usize) }).collect();
        assert_eq!(vals, vec![100, 101]);

        let (ptr, count) = idx.probe(15);
        assert_eq!(count, 1);
        assert_eq!(unsafe { *ptr }, 150);

        let (ptr, count) = idx.probe(99);
        assert!(ptr.is_null());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_clear() {
        let mut idx = RelIndex::new_empty(16);
        idx.insert(1, 10);
        idx.insert(2, 20);
        assert_eq!(idx.len, 2);

        idx.clear();
        assert_eq!(idx.len, 0);
        let (ptr, _) = idx.probe(1);
        assert!(ptr.is_null());

        // Can reuse after clear.
        idx.insert(3, 30);
        assert_eq!(idx.len, 1);
        let (ptr, count) = idx.probe(3);
        assert!(!ptr.is_null());
        assert_eq!(count, 1);
        assert_eq!(unsafe { *ptr }, 30);
    }
}
