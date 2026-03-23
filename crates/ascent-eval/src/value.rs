//! Runtime values for the interpreter.
//!
//! # Examples
//!
//! Creating values:
//!
//! ```
//! use ascent_eval::value::Value;
//!
//! let i = Value::I32(42);
//! let b = Value::Bool(true);
//! let c = Value::Char('x');
//! ```
//!
//! Arithmetic (wrapping):
//!
//! ```
//! use ascent_eval::value::Value;
//!
//! let result = Value::I32(10).add(&Value::I32(20));
//! assert_eq!(result, Some(Value::I32(30)));
//! ```
//!
//! Comparison:
//!
//! ```
//! use ascent_eval::value::Value;
//! use std::cmp::Ordering;
//!
//! let ord = Value::I32(1).try_cmp(&Value::I32(2));
//! assert_eq!(ord, Some(Ordering::Less));
//! ```

use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Intern table for packing/unpacking values to/from u32 identifiers.
///
/// Implemented by [`crate::intern::StringTable`] for strings and by
/// [`crate::specialized::HashInternTable`] for arbitrary `Hash + Eq` types.
/// Values that carry their own intern id (i.e. `Value::Interned`) use the
/// table for display, comparison, and packed-storage round-trips.
pub trait InternTable {
    /// Pack a `Value` to its u32 id. Returns `None` on type mismatch.
    fn pack(&self, val: &Value) -> Option<u32>;
    /// Write the display representation of the value with the given id.
    fn fmt_display(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Write the debug representation of the value with the given id.
    fn fmt_debug(&self, id: u32, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Compare two values by their ids (within the same table).
    fn cmp_ids(&self, a: u32, b: u32) -> Ordering;
    /// Resolve an id to its string representation, if this table stores strings.
    /// Returns `None` for non-string intern tables (e.g. `HashInternTable`).
    fn resolve_str(&self, _id: u32) -> Option<&str> {
        None
    }

    /// Return the human-readable type name for values in this table
    /// (e.g. `"String"` for `StringTable`).
    fn type_name(&self) -> &str {
        "interned"
    }
}

/// Object-safe trait for user-defined value types (custom types registered via `Engine::register_type`).
pub trait DynValue: Any + Send + Sync {
    /// Clone into a new `Box<dyn DynValue>`.
    fn clone_box(&self) -> Box<dyn DynValue>;
    /// Test equality with another `DynValue`.
    fn eq_box(&self, other: &dyn DynValue) -> bool;
    /// Feed this value into a `Hasher` (object-safe hash).
    fn hash_box(&self, state: &mut dyn Hasher);
    /// Compare with another `DynValue` for ordering. Returns `None` on type mismatch.
    fn cmp_box(&self, other: &dyn DynValue) -> Option<Ordering>;
    /// Write `Debug` representation to a formatter.
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Write `Display` representation to a formatter.
    fn display_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    /// Downcast to `Any` for concrete type recovery.
    fn as_any(&self) -> &dyn Any;
    /// Return the concrete type name of this value.
    fn type_name(&self) -> &'static str;
}

impl<T> DynValue for T
where
    T: Clone + Eq + Hash + Ord + fmt::Debug + fmt::Display + Any + Send + Sync + 'static,
{
    fn clone_box(&self) -> Box<dyn DynValue> {
        Box::new(self.clone())
    }
    fn eq_box(&self, other: &dyn DynValue) -> bool {
        other
            .as_any()
            .downcast_ref::<T>()
            .is_some_and(|o| self == o)
    }
    fn hash_box(&self, state: &mut dyn Hasher) {
        self.hash(&mut HasherWrapper(state));
    }
    fn cmp_box(&self, other: &dyn DynValue) -> Option<Ordering> {
        other.as_any().downcast_ref::<T>().map(|o| self.cmp(o))
    }
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
    fn display_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

/// Adapter to bridge `&mut dyn Hasher` (not object-safe) with the `Hash` trait.
struct HasherWrapper<'a>(&'a mut dyn Hasher);

impl Hasher for HasherWrapper<'_> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
}

/// A runtime value in the interpreter.
///
/// Use accessor methods ([`Value::as_i32`], [`Value::as_bool`], [`Value::as_str`],
/// [`Value::type_name`]) rather than matching on variants directly — some variants
/// like [`Value::Interned`] are internal implementation details.
#[non_exhaustive]
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
    /// An interned value — an internal representation for efficient storage.
    ///
    /// The intern table owns the backing storage and provides display, debug,
    /// comparison, and pack/unpack. Strings, and any other `Hash + Eq` type
    /// that opts in, are represented here.
    ///
    /// **Users should not match on this variant directly.** Use accessor
    /// methods like [`Value::as_str`] or [`Value::type_name`] instead.
    Interned(Rc<dyn InternTable>, u32),
    /// Tuple of values.
    Tuple(Rc<Vec<Value>>),
    /// Option type.
    Option(Option<Box<Value>>),
    /// Dual lattice wrapper (reverses ordering for lattice join).
    Dual(Box<Value>),
    /// A range (for generators).
    Range {
        start: Box<Value>,
        end: Box<Value>,
        inclusive: bool,
    },
    /// A user-defined custom type.
    Custom(Box<dyn DynValue>),
}

impl Clone for Value {
    fn clone(&self) -> Self {
        match self {
            Value::Unit => Value::Unit,
            Value::Bool(v) => Value::Bool(*v),
            Value::I8(v) => Value::I8(*v),
            Value::I16(v) => Value::I16(*v),
            Value::I32(v) => Value::I32(*v),
            Value::I64(v) => Value::I64(*v),
            Value::I128(v) => Value::I128(*v),
            Value::Isize(v) => Value::Isize(*v),
            Value::U8(v) => Value::U8(*v),
            Value::U16(v) => Value::U16(*v),
            Value::U32(v) => Value::U32(*v),
            Value::U64(v) => Value::U64(*v),
            Value::U128(v) => Value::U128(*v),
            Value::Usize(v) => Value::Usize(*v),
            Value::F32(v) => Value::F32(*v),
            Value::F64(v) => Value::F64(*v),
            Value::Char(v) => Value::Char(*v),
            Value::Interned(table, id) => Value::Interned(table.clone(), *id),
            Value::Tuple(v) => Value::Tuple(v.clone()),
            Value::Option(v) => Value::Option(v.clone()),
            Value::Dual(v) => Value::Dual(v.clone()),
            Value::Range {
                start,
                end,
                inclusive,
            } => Value::Range {
                start: start.clone(),
                end: end.clone(),
                inclusive: *inclusive,
            },
            Value::Custom(v) => Value::Custom(v.clone_box()),
        }
    }
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
            (Value::Interned(t1, id1), Value::Interned(t2, id2)) => {
                Rc::ptr_eq(t1, t2) && id1 == id2
            }
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Option(a), Value::Option(b)) => a == b,
            (Value::Dual(a), Value::Dual(b)) => a == b,
            (Value::Custom(a), Value::Custom(b)) => a.eq_box(b.as_ref()),
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
            Value::Interned(table, id) => {
                // Mix in the table's identity (data pointer) so values from
                // different tables with the same id don't collide.
                (Rc::as_ptr(table) as *const () as usize).hash(state);
                id.hash(state);
            }
            Value::Tuple(v) => v.hash(state),
            Value::Option(v) => v.hash(state),
            Value::Dual(v) => v.hash(state),
            Value::Range {
                start,
                end,
                inclusive,
            } => {
                start.hash(state);
                end.hash(state);
                inclusive.hash(state);
            }
            Value::Custom(v) => v.hash_box(state),
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
            Value::Interned(table, id) => table.fmt_debug(*id, f),
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
            Value::Dual(v) => write!(f, "Dual({v:?})"),
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
            Value::Custom(v) => v.debug_fmt(f),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Interned(table, id) => table.fmt_display(*id, f),
            Value::Char(c) => write!(f, "{c}"),
            Value::Custom(v) => v.display_fmt(f),
            other => write!(f, "{other:?}"),
        }
    }
}

/// A tuple of values. Alias for `Vec<Value>`.
///
/// Tuples are the basic unit of data in relations. Each tuple has a fixed
/// arity (number of columns) determined by the relation it belongs to.
pub type Tuple = Vec<Value>;

impl Value {
    /// Create a tuple value.
    pub fn tuple(values: Vec<Value>) -> Self {
        Value::Tuple(Rc::new(values))
    }

    /// Create a custom value from any type implementing the required traits.
    pub fn custom<T: DynValue + 'static>(v: T) -> Self {
        Value::Custom(Box::new(v))
    }

    /// Downcast a `Value::Custom` to a concrete type.
    ///
    /// Returns `None` if the value is not a `Custom` variant or if
    /// the inner type doesn't match `T`.
    pub fn downcast_custom<T: 'static>(&self) -> Option<&T> {
        if let Value::Custom(v) = self {
            v.as_any().downcast_ref::<T>()
        } else {
            None
        }
    }

    /// Create a string value (interns the string in the thread-local string table).
    pub fn string(s: impl AsRef<str>) -> Self {
        crate::intern::string_value(s.as_ref())
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
            Value::Isize(v) => Some(*v as i64),
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

    /// Try to get as a string slice, resolving interned strings transparently.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Interned(table, id) => table.resolve_str(*id),
            _ => None,
        }
    }

    /// Get a display-friendly type name for this value.
    pub fn type_name(&self) -> &str {
        match self {
            Value::Unit => "()",
            Value::Bool(_) => "bool",
            Value::I8(_) => "i8",
            Value::I16(_) => "i16",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::I128(_) => "i128",
            Value::Isize(_) => "isize",
            Value::U8(_) => "u8",
            Value::U16(_) => "u16",
            Value::U32(_) => "u32",
            Value::U64(_) => "u64",
            Value::U128(_) => "u128",
            Value::Usize(_) => "usize",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Char(_) => "char",
            Value::Interned(table, _) => table.type_name(),
            Value::Tuple(_) => "tuple",
            Value::Option(_) => "Option",
            Value::Dual(_) => "Dual",
            Value::Range { .. } => "Range",
            Value::Custom(v) => v.type_name(),
        }
    }

    /// Check if this is a "truthy" value for conditions.
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            Value::Option(None) => false,
            _ => true,
        }
    }
}

// Arithmetic operations — wrapping for integer add/sub/mul (matches release-mode semantics),
// raw operator for floats (no overflow concern).
macro_rules! impl_numeric_wrapping_binop {
    ($method:ident, $wrapping_method:ident, $op:tt) => {
        pub fn $method(&self, other: &Value) -> Option<Value> {
            match (self, other) {
                (Value::I8(a), Value::I8(b)) => Some(Value::I8(a.$wrapping_method(*b))),
                (Value::I16(a), Value::I16(b)) => Some(Value::I16(a.$wrapping_method(*b))),
                (Value::I32(a), Value::I32(b)) => Some(Value::I32(a.$wrapping_method(*b))),
                (Value::I64(a), Value::I64(b)) => Some(Value::I64(a.$wrapping_method(*b))),
                (Value::I128(a), Value::I128(b)) => Some(Value::I128(a.$wrapping_method(*b))),
                (Value::Isize(a), Value::Isize(b)) => Some(Value::Isize(a.$wrapping_method(*b))),
                (Value::U8(a), Value::U8(b)) => Some(Value::U8(a.$wrapping_method(*b))),
                (Value::U16(a), Value::U16(b)) => Some(Value::U16(a.$wrapping_method(*b))),
                (Value::U32(a), Value::U32(b)) => Some(Value::U32(a.$wrapping_method(*b))),
                (Value::U64(a), Value::U64(b)) => Some(Value::U64(a.$wrapping_method(*b))),
                (Value::U128(a), Value::U128(b)) => Some(Value::U128(a.$wrapping_method(*b))),
                (Value::Usize(a), Value::Usize(b)) => Some(Value::Usize(a.$wrapping_method(*b))),
                (Value::F32(OrderedFloat(a)), Value::F32(OrderedFloat(b))) => Some(Value::F32(OrderedFloat(a $op b))),
                (Value::F64(OrderedFloat(a)), Value::F64(OrderedFloat(b))) => Some(Value::F64(OrderedFloat(a $op b))),
                _ => None,
            }
        }
    };
}

// Checked div/rem for integer types (returns None on zero divisor); floats use raw operator
macro_rules! impl_checked_binop {
    ($method:ident, $checked_method:ident, $op:tt) => {
        pub fn $method(&self, other: &Value) -> Option<Value> {
            match (self, other) {
                (Value::I8(a), Value::I8(b)) => a.$checked_method(*b).map(Value::I8),
                (Value::I16(a), Value::I16(b)) => a.$checked_method(*b).map(Value::I16),
                (Value::I32(a), Value::I32(b)) => a.$checked_method(*b).map(Value::I32),
                (Value::I64(a), Value::I64(b)) => a.$checked_method(*b).map(Value::I64),
                (Value::I128(a), Value::I128(b)) => a.$checked_method(*b).map(Value::I128),
                (Value::Isize(a), Value::Isize(b)) => a.$checked_method(*b).map(Value::Isize),
                (Value::U8(a), Value::U8(b)) => a.$checked_method(*b).map(Value::U8),
                (Value::U16(a), Value::U16(b)) => a.$checked_method(*b).map(Value::U16),
                (Value::U32(a), Value::U32(b)) => a.$checked_method(*b).map(Value::U32),
                (Value::U64(a), Value::U64(b)) => a.$checked_method(*b).map(Value::U64),
                (Value::U128(a), Value::U128(b)) => a.$checked_method(*b).map(Value::U128),
                (Value::Usize(a), Value::Usize(b)) => a.$checked_method(*b).map(Value::Usize),
                (Value::F32(OrderedFloat(a)), Value::F32(OrderedFloat(b))) => Some(Value::F32(OrderedFloat(a $op b))),
                (Value::F64(OrderedFloat(a)), Value::F64(OrderedFloat(b))) => Some(Value::F64(OrderedFloat(a $op b))),
                _ => None,
            }
        }
    };
}

macro_rules! impl_integer_binop {
    ($method:ident, $op:tt) => {
        pub fn $method(&self, other: &Value) -> Option<Value> {
            match (self, other) {
                (Value::I8(a), Value::I8(b)) => Some(Value::I8(a $op b)),
                (Value::I16(a), Value::I16(b)) => Some(Value::I16(a $op b)),
                (Value::I32(a), Value::I32(b)) => Some(Value::I32(a $op b)),
                (Value::I64(a), Value::I64(b)) => Some(Value::I64(a $op b)),
                (Value::I128(a), Value::I128(b)) => Some(Value::I128(a $op b)),
                (Value::Isize(a), Value::Isize(b)) => Some(Value::Isize(a $op b)),
                (Value::U8(a), Value::U8(b)) => Some(Value::U8(a $op b)),
                (Value::U16(a), Value::U16(b)) => Some(Value::U16(a $op b)),
                (Value::U32(a), Value::U32(b)) => Some(Value::U32(a $op b)),
                (Value::U64(a), Value::U64(b)) => Some(Value::U64(a $op b)),
                (Value::U128(a), Value::U128(b)) => Some(Value::U128(a $op b)),
                (Value::Usize(a), Value::Usize(b)) => Some(Value::Usize(a $op b)),
                _ => None,
            }
        }
    };
}

// Checked shift: returns None when shift amount >= type width
macro_rules! impl_checked_shift {
    ($method:ident, $checked_method:ident) => {
        pub fn $method(&self, other: &Value) -> Option<Value> {
            match (self, other) {
                (Value::I8(a), Value::I8(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::I8),
                (Value::I16(a), Value::I16(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::I16),
                (Value::I32(a), Value::I32(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::I32),
                (Value::I64(a), Value::I64(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::I64),
                (Value::I128(a), Value::I128(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::I128),
                (Value::Isize(a), Value::Isize(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::Isize),
                (Value::U8(a), Value::U8(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::U8),
                (Value::U16(a), Value::U16(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::U16),
                (Value::U32(a), Value::U32(b)) => a.$checked_method(*b).map(Value::U32),
                (Value::U64(a), Value::U64(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::U64),
                (Value::U128(a), Value::U128(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::U128),
                (Value::Usize(a), Value::Usize(b)) => u32::try_from(*b).ok().and_then(|s| a.$checked_method(s)).map(Value::Usize),
                _ => None,
            }
        }
    };
}

impl Value {
    impl_numeric_wrapping_binop!(add, wrapping_add, +);
    impl_numeric_wrapping_binop!(sub, wrapping_sub, -);
    impl_numeric_wrapping_binop!(mul, wrapping_mul, *);
    impl_checked_binop!(div, checked_div, /);
    impl_checked_binop!(rem, checked_rem, %);

    impl_integer_binop!(bitand, &);
    impl_integer_binop!(bitor, |);
    impl_integer_binop!(bitxor, ^);
    impl_checked_shift!(shl, checked_shl);
    impl_checked_shift!(shr, checked_shr);

    pub fn neg(&self) -> Option<Value> {
        match self {
            Value::I8(v) => Some(Value::I8(v.wrapping_neg())),
            Value::I16(v) => Some(Value::I16(v.wrapping_neg())),
            Value::I32(v) => Some(Value::I32(v.wrapping_neg())),
            Value::I64(v) => Some(Value::I64(v.wrapping_neg())),
            Value::I128(v) => Some(Value::I128(v.wrapping_neg())),
            Value::Isize(v) => Some(Value::Isize(v.wrapping_neg())),
            Value::F32(OrderedFloat(v)) => Some(Value::F32(OrderedFloat(-v))),
            Value::F64(OrderedFloat(v)) => Some(Value::F64(OrderedFloat(-v))),
            _ => None,
        }
    }

    pub fn not(&self) -> Option<Value> {
        match self {
            Value::Bool(v) => Some(Value::Bool(!v)),
            Value::I8(v) => Some(Value::I8(!v)),
            Value::I16(v) => Some(Value::I16(!v)),
            Value::I32(v) => Some(Value::I32(!v)),
            Value::I64(v) => Some(Value::I64(!v)),
            Value::I128(v) => Some(Value::I128(!v)),
            Value::Isize(v) => Some(Value::Isize(!v)),
            Value::U8(v) => Some(Value::U8(!v)),
            Value::U16(v) => Some(Value::U16(!v)),
            Value::U32(v) => Some(Value::U32(!v)),
            Value::U64(v) => Some(Value::U64(!v)),
            Value::U128(v) => Some(Value::U128(!v)),
            Value::Usize(v) => Some(Value::Usize(!v)),
            _ => None,
        }
    }

    pub fn abs(&self) -> Option<Value> {
        match self {
            Value::I8(v) => Some(Value::I8(v.wrapping_abs())),
            Value::I16(v) => Some(Value::I16(v.wrapping_abs())),
            Value::I32(v) => Some(Value::I32(v.wrapping_abs())),
            Value::I64(v) => Some(Value::I64(v.wrapping_abs())),
            Value::I128(v) => Some(Value::I128(v.wrapping_abs())),
            Value::Isize(v) => Some(Value::Isize(v.wrapping_abs())),
            Value::F32(OrderedFloat(v)) => Some(Value::F32(OrderedFloat(v.abs()))),
            Value::F64(OrderedFloat(v)) => Some(Value::F64(OrderedFloat(v.abs()))),
            _ => None,
        }
    }

    pub fn try_cmp(&self, other: &Value) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::I8(a), Value::I8(b)) => Some(a.cmp(b)),
            (Value::I16(a), Value::I16(b)) => Some(a.cmp(b)),
            (Value::I32(a), Value::I32(b)) => Some(a.cmp(b)),
            (Value::I64(a), Value::I64(b)) => Some(a.cmp(b)),
            (Value::I128(a), Value::I128(b)) => Some(a.cmp(b)),
            (Value::Isize(a), Value::Isize(b)) => Some(a.cmp(b)),
            (Value::U8(a), Value::U8(b)) => Some(a.cmp(b)),
            (Value::U16(a), Value::U16(b)) => Some(a.cmp(b)),
            (Value::U32(a), Value::U32(b)) => Some(a.cmp(b)),
            (Value::U64(a), Value::U64(b)) => Some(a.cmp(b)),
            (Value::U128(a), Value::U128(b)) => Some(a.cmp(b)),
            (Value::Usize(a), Value::Usize(b)) => Some(a.cmp(b)),
            (Value::F32(a), Value::F32(b)) => Some(a.cmp(b)),
            (Value::F64(a), Value::F64(b)) => Some(a.cmp(b)),
            (Value::Char(a), Value::Char(b)) => Some(a.cmp(b)),
            (Value::Interned(t1, a), Value::Interned(t2, b)) => {
                if Rc::ptr_eq(t1, t2) {
                    Some(t1.cmp_ids(*a, *b))
                } else {
                    // Cross-table: stable but arbitrary ordering by table address.
                    let p1 = Rc::as_ptr(t1) as *const () as usize;
                    let p2 = Rc::as_ptr(t2) as *const () as usize;
                    Some(p1.cmp(&p2).then(a.cmp(b)))
                }
            }
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            // Dual reverses the ordering
            (Value::Dual(a), Value::Dual(b)) => b.try_cmp(a),
            (Value::Custom(a), Value::Custom(b)) => a.cmp_box(b.as_ref()),
            _ => None,
        }
    }

    /// Lattice join: combine two values using lattice semantics.
    ///
    /// For numeric types, join = max (least upper bound).
    /// For Dual, join = Dual(min(inner)) (reversed).
    /// Returns `Some(joined_value)` if a merge happened, `None` if types don't match.
    pub fn lattice_join(&self, other: &Value) -> Option<Value> {
        match (self, other) {
            (Value::Dual(a), Value::Dual(b)) => {
                // Dual join = Dual(meet of inner) = Dual(min)
                a.try_cmp(b).map(|ord| {
                    Value::Dual(Box::new(if ord.is_le() {
                        (**a).clone()
                    } else {
                        (**b).clone()
                    }))
                })
            }
            _ => {
                // Regular join = max
                self.try_cmp(other).map(|ord| {
                    if ord.is_ge() {
                        self.clone()
                    } else {
                        other.clone()
                    }
                })
            }
        }
    }

    /// Try to get as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(OrderedFloat(v)) => Some(*v),
            Value::F32(OrderedFloat(v)) => Some(*v as f64),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    /// Cast to a target type by name.
    ///
    /// Supports integer-to-integer, float-to-integer, and integer-to-float casts.
    pub fn cast_to(&self, target: &str) -> Option<Value> {
        // Try integer path first, then float-to-int fallback.
        if let Some(v) = self.as_i64() {
            return match target {
                "i8" => Some(Value::I8(v as i8)),
                "i16" => Some(Value::I16(v as i16)),
                "i32" => Some(Value::I32(v as i32)),
                "i64" => Some(Value::I64(v)),
                "i128" => Some(Value::I128(v as i128)),
                "isize" => Some(Value::Isize(v as isize)),
                "u8" => Some(Value::U8(v as u8)),
                "u16" => Some(Value::U16(v as u16)),
                "u32" => Some(Value::U32(v as u32)),
                "u64" => Some(Value::U64(v as u64)),
                "u128" => Some(Value::U128(v as u128)),
                "usize" => Some(Value::Usize(v as usize)),
                "f32" => Some(Value::F32(OrderedFloat(v as f32))),
                "f64" => Some(Value::F64(OrderedFloat(v as f64))),
                _ => None,
            };
        }
        // Float-to-integer fallback.
        let f = self.as_f64()?;
        match target {
            "i8" => Some(Value::I8(f as i8)),
            "i16" => Some(Value::I16(f as i16)),
            "i32" => Some(Value::I32(f as i32)),
            "i64" => Some(Value::I64(f as i64)),
            "i128" => Some(Value::I128(f as i128)),
            "isize" => Some(Value::Isize(f as isize)),
            "u8" => Some(Value::U8(f as u8)),
            "u16" => Some(Value::U16(f as u16)),
            "u32" => Some(Value::U32(f as u32)),
            "u64" => Some(Value::U64(f as u64)),
            "u128" => Some(Value::U128(f as u128)),
            "usize" => Some(Value::Usize(f as usize)),
            "f32" => Some(Value::F32(OrderedFloat(f as f32))),
            "f64" => Some(Value::F64(OrderedFloat(f))),
            _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct Point {
        x: i32,
        y: i32,
    }

    impl fmt::Display for Point {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "({}, {})", self.x, self.y)
        }
    }

    #[test]
    fn test_custom_clone() {
        let v = Value::custom(Point { x: 1, y: 2 });
        let v2 = v.clone();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_custom_eq() {
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(Point { x: 1, y: 2 });
        let c = Value::custom(Point { x: 3, y: 4 });
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_custom_eq_different_types() {
        // Different concrete types are never equal
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(42i64);
        assert_ne!(a, b);
    }

    #[test]
    fn test_custom_hash() {
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(Point { x: 1, y: 2 });
        let hash_a = {
            let mut h = DefaultHasher::new();
            a.hash(&mut h);
            h.finish()
        };
        let hash_b = {
            let mut h = DefaultHasher::new();
            b.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn test_custom_debug() {
        let v = Value::custom(Point { x: 1, y: 2 });
        let dbg = format!("{v:?}");
        assert!(dbg.contains("Point"));
        assert!(dbg.contains("1"));
        assert!(dbg.contains("2"));
    }

    #[test]
    fn test_custom_display() {
        let v = Value::custom(Point { x: 1, y: 2 });
        let disp = format!("{v}");
        assert_eq!(disp, "(1, 2)");
    }

    #[test]
    fn test_custom_cmp() {
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(Point { x: 3, y: 4 });
        assert_eq!(a.try_cmp(&b), Some(Ordering::Less));
    }

    #[test]
    fn test_custom_no_arithmetic() {
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(Point { x: 3, y: 4 });
        assert!(a.add(&b).is_none());
        assert!(a.neg().is_none());
        assert!(a.not().is_none());
        assert!(a.abs().is_none());
    }

    #[test]
    fn test_custom_truthy() {
        let v = Value::custom(Point { x: 0, y: 0 });
        assert!(v.is_truthy());
    }

    #[test]
    fn test_custom_cast_to_none() {
        let v = Value::custom(Point { x: 1, y: 2 });
        assert!(v.cast_to("i32").is_none());
    }

    #[test]
    fn test_custom_lattice_join() {
        let a = Value::custom(Point { x: 1, y: 2 });
        let b = Value::custom(Point { x: 3, y: 4 });
        // lattice_join uses try_cmp → max
        let joined = a.lattice_join(&b).unwrap();
        assert_eq!(joined, Value::custom(Point { x: 3, y: 4 }));
    }

    #[test]
    fn test_custom_downcast() {
        let v = Value::custom(Point { x: 1, y: 2 });
        if let Value::Custom(inner) = &v {
            let point = inner.as_any().downcast_ref::<Point>().unwrap();
            assert_eq!(point.x, 1);
            assert_eq!(point.y, 2);
        } else {
            panic!("expected Custom");
        }
    }
}
