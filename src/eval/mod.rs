//! Semi-naive evaluation engine for Ascent programs.
//!
//! This crate provides the runtime for interpreting Ascent Datalog programs.
//!
//! # Example
//!
//! ```
//! use ascent_interpreter::eval::Engine;
//! use ascent_interpreter::ir::Program;
//! use ascent_interpreter::syntax::AscentProgram;
//! use ascent_interpreter::eval::value::Value;
//!
//! let input = r#"
//!     relation edge(i32, i32);
//!     relation path(i32, i32);
//!     path(x, y) <-- edge(x, y);
//!     path(x, z) <-- edge(x, y), path(y, z);
//! "#;
//!
//! let ast: AscentProgram = syn::parse_str(input).unwrap();
//! let program = Program::from_ast(ast).unwrap();
//! let mut engine = Engine::new(program);
//!
//! // Insert initial facts
//! engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
//! engine.insert("edge", vec![Value::I32(2), Value::I32(3)]).unwrap();
//!
//! // Run to fixpoint
//! engine.run().unwrap();
//!
//! // Query results
//! let path = engine.relation("path").unwrap();
//! assert_eq!(path.len(), 3); // (1,2), (2,3), (1,3)
//! ```

pub mod aggregators;
mod bytecode;
mod compiled;
pub mod error;
mod engine;
pub mod expr;
pub mod intern;
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
mod jit;
#[cfg(feature = "specialized")]
mod jit_index;
mod relation;
#[cfg(feature = "serde")]
pub mod serde_bridge;
#[cfg(feature = "specialized")]
mod specialized;
pub mod value;

pub use error::EvalError;
pub use engine::{Engine, TypeRegistry, ValueDestructor};
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub use engine::SharedJitCompiler;
pub use relation::{Relation, SourceId};
pub use value::{DynValue, OrderedFloat, Tuple, Value};
