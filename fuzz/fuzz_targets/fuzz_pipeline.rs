//! Fuzz the full parse → desugar → IR lowering pipeline.
//!
//! Goal: find panics in desugaring or IR lowering that the parser alone wouldn't catch.
//! If a program parses successfully, the pipeline must not panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ascent_ir::Program;
use ascent_syntax::AscentProgram;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        // If parsing succeeds, lowering must not panic
        if let Ok(ast) = syn::parse_str::<AscentProgram>(input) {
            let _ = Program::from_ast(ast);
        }
    }
});
