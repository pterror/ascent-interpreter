//! Error types for Datalog program evaluation.

use std::fmt;

/// Errors that can occur during Datalog program evaluation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EvalError {
    /// A variable was referenced but not bound in the current scope.
    UndefinedVariable(String),
    /// A binary/unary operation received incompatible types.
    TypeMismatch {
        op: String,
        left: String,
        right: Option<String>,
    },
    /// Division or remainder by zero.
    DivisionByZero,
    /// Expression form not supported by the evaluator.
    UnsupportedExpression(String),
    /// Relation not found during insert.
    UnknownRelation(String),
    /// Tuple arity doesn't match relation definition.
    ArityMismatch {
        relation: String,
        expected: usize,
        got: usize,
    },
    /// Aggregator name not recognized.
    UnknownAggregator(String),
    /// Fixpoint iteration limit exceeded.
    IterationLimitExceeded { limit: usize },
    /// Range too large to materialize.
    RangeTooLarge { size: usize, limit: usize },
    /// JIT compilation or execution error.
    Jit(String),
    /// IR lowering error.
    Lowering(String),
    /// Parse error.
    Parse(String),
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::TypeMismatch {
                op,
                left,
                right: Some(right),
            } => write!(f, "type mismatch in {op}: {left} vs {right}"),
            Self::TypeMismatch {
                op,
                left,
                right: None,
            } => write!(f, "type mismatch in {op}: {left}"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::UnsupportedExpression(desc) => write!(f, "unsupported expression: {desc}"),
            Self::UnknownRelation(name) => write!(f, "unknown relation: {name}"),
            Self::ArityMismatch {
                relation,
                expected,
                got,
            } => write!(f, "arity mismatch for {relation}: expected {expected}, got {got}"),
            Self::UnknownAggregator(name) => write!(f, "unknown aggregator: {name}"),
            Self::IterationLimitExceeded { limit } => {
                write!(f, "fixpoint iteration limit exceeded ({limit})")
            }
            Self::RangeTooLarge { size, limit } => {
                write!(f, "range too large: {size} elements (limit: {limit})")
            }
            Self::Jit(msg) => write!(f, "JIT error: {msg}"),
            Self::Lowering(msg) => write!(f, "lowering error: {msg}"),
            Self::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for EvalError {}
