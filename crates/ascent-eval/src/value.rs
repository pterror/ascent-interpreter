//! Runtime values for the interpreter.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// A runtime value in the interpreter.
#[derive(Clone)]
pub enum Value {
    /// Unit value.
    Unit,
    /// Boolean.
    Bool(bool),
    /// Signed integers.
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    Isize(isize),
    /// Unsigned integers.
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(usize),
    /// Floating point (wrapped for Hash/Eq).
    F32(OrderedFloat<f32>),
    F64(OrderedFloat<f64>),
    /// Character.
    Char(char),
    /// String.
    String(Rc<String>),
    /// Tuple of values.
    Tuple(Rc<Vec<Value>>),
    /// Option type.
    Option(Option<Box<Value>>),
    /// A range (for generators).
    Range {
        start: Box<Value>,
        end: Box<Value>,
        inclusive: bool,
    },
}

/// Wrapper for floats that implements Hash and Eq via total ordering.
#[derive(Clone, Copy)]
pub struct OrderedFloat<T>(pub T);

impl PartialEq for OrderedFloat<f32> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrderedFloat<f32> {}

impl Hash for OrderedFloat<f32> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl PartialOrd for OrderedFloat<f32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat<f32> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialEq for OrderedFloat<f64> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == Ordering::Equal
    }
}

impl Eq for OrderedFloat<f64> {}

impl Hash for OrderedFloat<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl PartialOrd for OrderedFloat<f64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat<f64> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Unit, Value::Unit) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::I8(a), Value::I8(b)) => a == b,
            (Value::I16(a), Value::I16(b)) => a == b,
            (Value::I32(a), Value::I32(b)) => a == b,
            (Value::I64(a), Value::I64(b)) => a == b,
            (Value::I128(a), Value::I128(b)) => a == b,
            (Value::Isize(a), Value::Isize(b)) => a == b,
            (Value::U8(a), Value::U8(b)) => a == b,
            (Value::U16(a), Value::U16(b)) => a == b,
            (Value::U32(a), Value::U32(b)) => a == b,
            (Value::U64(a), Value::U64(b)) => a == b,
            (Value::U128(a), Value::U128(b)) => a == b,
            (Value::Usize(a), Value::Usize(b)) => a == b,
            (Value::F32(a), Value::F32(b)) => a == b,
            (Value::F64(a), Value::F64(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Option(a), Value::Option(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Unit => {}
            Value::Bool(v) => v.hash(state),
            Value::I8(v) => v.hash(state),
            Value::I16(v) => v.hash(state),
            Value::I32(v) => v.hash(state),
            Value::I64(v) => v.hash(state),
            Value::I128(v) => v.hash(state),
            Value::Isize(v) => v.hash(state),
            Value::U8(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::U32(v) => v.hash(state),
            Value::U64(v) => v.hash(state),
            Value::U128(v) => v.hash(state),
            Value::Usize(v) => v.hash(state),
            Value::F32(v) => v.hash(state),
            Value::F64(v) => v.hash(state),
            Value::Char(v) => v.hash(state),
            Value::String(v) => v.hash(state),
            Value::Tuple(v) => v.hash(state),
            Value::Option(v) => v.hash(state),
            Value::Range {
                start,
                end,
                inclusive,
            } => {
                start.hash(state);
                end.hash(state);
                inclusive.hash(state);
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}i8"),
            Value::I16(v) => write!(f, "{v}i16"),
            Value::I32(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}i64"),
            Value::I128(v) => write!(f, "{v}i128"),
            Value::Isize(v) => write!(f, "{v}isize"),
            Value::U8(v) => write!(f, "{v}u8"),
            Value::U16(v) => write!(f, "{v}u16"),
            Value::U32(v) => write!(f, "{v}u32"),
            Value::U64(v) => write!(f, "{v}u64"),
            Value::U128(v) => write!(f, "{v}u128"),
            Value::Usize(v) => write!(f, "{v}usize"),
            Value::F32(v) => write!(f, "{:?}f32", v.0),
            Value::F64(v) => write!(f, "{:?}f64", v.0),
            Value::Char(v) => write!(f, "{v:?}"),
            Value::String(v) => write!(f, "{v:?}"),
            Value::Tuple(v) => {
                write!(f, "(")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val:?}")?;
                }
                if v.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Value::Option(None) => write!(f, "None"),
            Value::Option(Some(v)) => write!(f, "Some({v:?})"),
            Value::Range {
                start,
                end,
                inclusive,
            } => {
                if *inclusive {
                    write!(f, "{start:?}..={end:?}")
                } else {
                    write!(f, "{start:?}..{end:?}")
                }
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "{s}"),
            Value::Char(c) => write!(f, "{c}"),
            other => write!(f, "{other:?}"),
        }
    }
}

/// A tuple of values, used as a row in a relation.
pub type Tuple = Vec<Value>;

impl Value {
    /// Create a tuple value.
    pub fn tuple(values: Vec<Value>) -> Self {
        Value::Tuple(Rc::new(values))
    }

    /// Create a string value.
    pub fn string(s: impl Into<String>) -> Self {
        Value::String(Rc::new(s.into()))
    }

    /// Try to get as i32.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Value::I32(v) => Some(*v),
            Value::I8(v) => Some(*v as i32),
            Value::I16(v) => Some(*v as i32),
            Value::U8(v) => Some(*v as i32),
            Value::U16(v) => Some(*v as i32),
            _ => None,
        }
    }

    /// Try to get as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(v) => Some(*v),
            Value::I32(v) => Some(*v as i64),
            Value::I8(v) => Some(*v as i64),
            Value::I16(v) => Some(*v as i64),
            Value::U8(v) => Some(*v as i64),
            Value::U16(v) => Some(*v as i64),
            Value::U32(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to get as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Check if this is a "truthy" value for conditions.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            Value::Option(None) => false,
            Value::Option(Some(_)) => true,
            _ => true,
        }
    }
}

// Convenience From implementations
impl From<()> for Value {
    fn from(_: ()) -> Self {
        Value::Unit
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I32(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I64(v)
    }
}

impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::U32(v)
    }
}

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::U64(v)
    }
}

impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Usize(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::string(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::string(v)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        Value::Option(v.map(|x| Box::new(x.into())))
    }
}
