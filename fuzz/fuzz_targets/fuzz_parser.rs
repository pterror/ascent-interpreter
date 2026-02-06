//! Fuzz the ascent-syntax parser with arbitrary strings.
//!
//! Goal: find panics, stack overflows, or infinite loops in the parser.
//! The parser should either return Ok or Err — never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ascent_syntax::AscentProgram;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        // The parser must not panic on any input
        let _ = syn::parse_str::<AscentProgram>(input);
    }
});
