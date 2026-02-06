//! Runtime values for the interpreter.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Object-safe trait for user-defined types in the interpreter.
///
/// Library consumers implement this (typically via the blanket impl) to register
/// custom Rust types that work with Ascent programs.
pub trait DynValue: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn DynValue>;
    fn eq_box(&self, other: &dyn DynValue) -> bool;
    fn hash_box(&self, state: &mut dyn Hasher);
    fn cmp_box(&self, other: &dyn DynValue) -> Option<Ordering>;
    fn debug_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn display_fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn as_any(&self) -> &dyn Any;
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
            Value::String(v) => Value::String(v.clone()),
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
            (Value::String(a), Value::String(b)) => a == b,
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
            Value::String(v) => v.hash(state),
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
            Value::String(s) => write!(f, "{s}"),
            Value::Char(c) => write!(f, "{c}"),
            Value::Custom(v) => v.display_fmt(f),
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

    /// Create a custom value from any type implementing the required traits.
    pub fn custom<T: DynValue + 'static>(v: T) -> Self {
        Value::Custom(Box::new(v))
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
            _ => true,
        }
    }
}

// Arithmetic operations
macro_rules! impl_numeric_binop {
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

impl Value {
    impl_numeric_binop!(add, +);
    impl_numeric_binop!(sub, -);
    impl_numeric_binop!(mul, *);
    impl_numeric_binop!(div, /);
    impl_numeric_binop!(rem, %);

    impl_integer_binop!(bitand, &);
    impl_integer_binop!(bitor, |);
    impl_integer_binop!(bitxor, ^);
    impl_integer_binop!(shl, <<);
    impl_integer_binop!(shr, >>);

    pub fn neg(&self) -> Option<Value> {
        match self {
            Value::I8(v) => Some(Value::I8(-v)),
            Value::I16(v) => Some(Value::I16(-v)),
            Value::I32(v) => Some(Value::I32(-v)),
            Value::I64(v) => Some(Value::I64(-v)),
            Value::I128(v) => Some(Value::I128(-v)),
            Value::Isize(v) => Some(Value::Isize(-v)),
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
            Value::I8(v) => Some(Value::I8(v.abs())),
            Value::I16(v) => Some(Value::I16(v.abs())),
            Value::I32(v) => Some(Value::I32(v.abs())),
            Value::I64(v) => Some(Value::I64(v.abs())),
            Value::I128(v) => Some(Value::I128(v.abs())),
            Value::Isize(v) => Some(Value::Isize(v.abs())),
            Value::F32(OrderedFloat(v)) => Some(Value::F32(OrderedFloat(v.abs()))),
            Value::F64(OrderedFloat(v)) => Some(Value::F64(OrderedFloat(v.abs()))),
            _ => None,
        }
    }

    pub fn partial_cmp_val(&self, other: &Value) -> Option<std::cmp::Ordering> {
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
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            // Dual reverses the ordering
            (Value::Dual(a), Value::Dual(b)) => b.partial_cmp_val(a),
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
                a.partial_cmp_val(b).map(|ord| {
                    Value::Dual(Box::new(if ord.is_le() {
                        (**a).clone()
                    } else {
                        (**b).clone()
                    }))
                })
            }
            _ => {
                // Regular join = max
                self.partial_cmp_val(other).map(|ord| {
                    if ord.is_ge() {
                        self.clone()
                    } else {
                        other.clone()
                    }
                })
            }
        }
    }

    /// Cast to a target type by name.
    pub fn cast_to(&self, target: &str) -> Option<Value> {
        let v = self.as_i64()?;
        match target {
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
        assert_eq!(a.partial_cmp_val(&b), Some(Ordering::Less));
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
        // lattice_join uses partial_cmp_val → max
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
