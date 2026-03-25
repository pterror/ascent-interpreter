//! Interpreter and JIT for Ascent Datalog programs.
//!
//! # Platform support
//!
//! The `jit-asm` JIT backend generates native x86-64 machine code and is only
//! available on `x86_64` targets. On other architectures (e.g. aarch64 / Apple
//! Silicon), the JIT code is compiled out and all rules use the interpreted
//! evaluator. The `jit-asm` feature can still be enabled in `Cargo.toml` on
//! non-x86-64 targets — it simply has no effect.

pub mod syntax;
pub mod ir;
pub mod eval;
