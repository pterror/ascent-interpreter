//! Parser for Ascent Datalog syntax.
//!
//! This crate provides a standalone parser for Ascent programs, extracted from
//! the ascent_macro crate. It can parse Ascent syntax from strings without
//! requiring a proc-macro context.
//!
//! # Example
//!
//! ```
//! use ascent_syntax::AscentProgram;
//!
//! let input = r#"
//!     relation edge(i32, i32);
//!     relation path(i32, i32);
//!     path(x, y) <-- edge(x, y);
//!     path(x, z) <-- edge(x, y), path(y, z);
//! "#;
//!
//! let program: AscentProgram = syn::parse_str(input).unwrap();
//! assert_eq!(program.relations.len(), 2);
//! assert_eq!(program.rules.len(), 2);
//! ```

pub mod desugar;
mod syn_utils;
mod syntax;
mod utils;

pub use syn_utils::*;
pub use syntax::*;
pub use utils::{expr_to_ident, is_wild_card, pat_to_ident};
