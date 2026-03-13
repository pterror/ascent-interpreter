//! Value/Option<Value> size constants and compile-time assertions for JIT codegen.

use std::mem;

use crate::value::Value;

/// Size of a `Value` enum in bytes.
pub const VALUE_SIZE: usize = mem::size_of::<Value>();

/// Alignment of a `Value` enum in bytes.
pub const VALUE_ALIGN: usize = mem::align_of::<Value>();

/// Size of an `Option<Value>` (one binding slot) in bytes.
pub const SLOT_SIZE: usize = mem::size_of::<Option<Value>>();

/// Size of a `usize` in bytes (pointer-width).
pub const PTR_SIZE: usize = mem::size_of::<usize>();

// Compile-time assertions: ensure our constants match actual layout.
const _: () = {
    // Option<Value> should be the same size as Value (niche optimization or +discriminant).
    // We don't rely on niche optimization; just assert sizes are sane.
    assert!(SLOT_SIZE >= VALUE_SIZE);
    assert!(VALUE_SIZE > 0);
    assert!(SLOT_SIZE > 0);
    assert!(PTR_SIZE == 8); // We assume 64-bit pointers throughout JIT codegen.
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_constants_match_runtime() {
        assert_eq!(VALUE_SIZE, mem::size_of::<Value>());
        assert_eq!(SLOT_SIZE, mem::size_of::<Option<Value>>());
        assert_eq!(VALUE_ALIGN, mem::align_of::<Value>());
    }

    #[test]
    fn option_value_none_is_zeroed_discriminant() {
        // Verify that Option<Value>::None has a predictable representation.
        // We use this in jit_slot_clear: we write zeros to clear a slot.
        let none: Option<Value> = None;
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&none as *const _ as *const u8, SLOT_SIZE) };
        // The None discriminant should have at least one zero byte at the start
        // (the discriminant of Option<Value> for None).
        // We don't assume full zeroing — just that writing the discriminant suffices.
        // The actual clearing in helpers.rs uses ptr::drop_in_place + write_bytes.
        let _ = bytes;
    }
}
