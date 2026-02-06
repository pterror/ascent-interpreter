//! Semi-naive evaluation engine for Ascent programs.
//!
//! This crate provides the runtime for interpreting Ascent Datalog programs.
//!
//! # Example
//!
//! ```
//! use ascent_eval::Engine;
//! use ascent_ir::Program;
//! use ascent_syntax::AscentProgram;
//! use ascent_eval::value::Value;
//!
//! let input = r#"
//!     relation edge(i32, i32);
//!     relation path(i32, i32);
//!     path(x, y) <-- edge(x, y);
//!     path(x, z) <-- edge(x, y), path(y, z);
//! "#;
//!
//! let ast: AscentProgram = syn::parse_str(input).unwrap();
//! let program = Program::from_ast(ast);
//! let mut engine = Engine::new(&program);
//!
//! // Insert initial facts
//! engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
//! engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
//!
//! // Run to fixpoint
//! engine.run(&program);
//!
//! // Query results
//! let path = engine.relation("path").unwrap();
//! assert_eq!(path.len(), 3); // (1,2), (2,3), (1,3)
//! ```

mod eval;
pub mod expr;
mod relation;
pub mod value;

pub use eval::{Bindings, Engine};
pub use relation::RelationStorage;
pub use value::{Tuple, Value};
