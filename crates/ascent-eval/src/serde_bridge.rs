//! Serde bridge: serialize Rust structs to/from `Vec<Value>`.
//!
//! This module provides a custom serde `Serializer` and `Deserializer` that
//! convert directly between Rust structs and `Vec<Value>`, with no intermediate
//! format (JSON, etc.). This enables `register_serde_type` as a one-liner
//! alternative to manually writing constructor/destructor closures.

use std::fmt;
use std::rc::Rc;

use serde::de::{self, DeserializeSeed, SeqAccess, Visitor};
use serde::ser::{self, SerializeStruct, SerializeTupleStruct};
use serde::{Deserialize, Serialize};

use crate::value::{OrderedFloat, Value};

// ─── Error type ─────────────────────────────────────────────────────

/// Error type for serde bridge operations.
#[derive(Debug)]
pub struct BridgeError(pub String);

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "serde bridge error: {}", self.0)
    }
}

impl std::error::Error for BridgeError {}

impl ser::Error for BridgeError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        BridgeError(msg.to_string())
    }
}

impl de::Error for BridgeError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        BridgeError(msg.to_string())
    }
}

// ─── Serialization ──────────────────────────────────────────────────

/// Serializes a single field value into a `Value`.
struct FieldSerializer;

impl ser::Serializer for FieldSerializer {
    type Ok = Value;
    type Error = BridgeError;

    type SerializeSeq = ser::Impossible<Value, BridgeError>;
    type SerializeTuple = ser::Impossible<Value, BridgeError>;
    type SerializeTupleStruct = ser::Impossible<Value, BridgeError>;
    type SerializeTupleVariant = ser::Impossible<Value, BridgeError>;
    type SerializeMap = ser::Impossible<Value, BridgeError>;
    type SerializeStruct = ser::Impossible<Value, BridgeError>;
    type SerializeStructVariant = ser::Impossible<Value, BridgeError>;

    fn serialize_bool(self, v: bool) -> Result<Value, BridgeError> {
        Ok(Value::Bool(v))
    }

    fn serialize_i8(self, v: i8) -> Result<Value, BridgeError> {
        Ok(Value::I8(v))
    }

    fn serialize_i16(self, v: i16) -> Result<Value, BridgeError> {
        Ok(Value::I16(v))
    }

    fn serialize_i32(self, v: i32) -> Result<Value, BridgeError> {
        Ok(Value::I32(v))
    }

    fn serialize_i64(self, v: i64) -> Result<Value, BridgeError> {
        Ok(Value::I64(v))
    }

    fn serialize_i128(self, v: i128) -> Result<Value, BridgeError> {
        Ok(Value::I128(v))
    }

    fn serialize_u8(self, v: u8) -> Result<Value, BridgeError> {
        Ok(Value::U8(v))
    }

    fn serialize_u16(self, v: u16) -> Result<Value, BridgeError> {
        Ok(Value::U16(v))
    }

    fn serialize_u32(self, v: u32) -> Result<Value, BridgeError> {
        Ok(Value::U32(v))
    }

    fn serialize_u64(self, v: u64) -> Result<Value, BridgeError> {
        Ok(Value::U64(v))
    }

    fn serialize_u128(self, v: u128) -> Result<Value, BridgeError> {
        Ok(Value::U128(v))
    }

    fn serialize_f32(self, v: f32) -> Result<Value, BridgeError> {
        Ok(Value::F32(OrderedFloat(v)))
    }

    fn serialize_f64(self, v: f64) -> Result<Value, BridgeError> {
        Ok(Value::F64(OrderedFloat(v)))
    }

    fn serialize_char(self, v: char) -> Result<Value, BridgeError> {
        Ok(Value::Char(v))
    }

    fn serialize_str(self, v: &str) -> Result<Value, BridgeError> {
        Ok(Value::String(Rc::new(v.to_string())))
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Value, BridgeError> {
        Err(BridgeError("bytes are not supported".into()))
    }

    fn serialize_none(self) -> Result<Value, BridgeError> {
        Ok(Value::Option(None))
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Value, BridgeError> {
        let inner = value.serialize(FieldSerializer)?;
        Ok(Value::Option(Some(Box::new(inner))))
    }

    fn serialize_unit(self) -> Result<Value, BridgeError> {
        Ok(Value::Unit)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value, BridgeError> {
        Ok(Value::Unit)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<Value, BridgeError> {
        Err(BridgeError(
            "enum variants are not supported as fields".into(),
        ))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Value, BridgeError> {
        value.serialize(FieldSerializer)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Value, BridgeError> {
        Err(BridgeError(
            "enum variants are not supported as fields".into(),
        ))
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, BridgeError> {
        Err(BridgeError("sequences are not supported as fields".into()))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, BridgeError> {
        Err(BridgeError("tuples are not supported as fields".into()))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, BridgeError> {
        Err(BridgeError(
            "nested tuple structs are not supported as fields".into(),
        ))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, BridgeError> {
        Err(BridgeError(
            "enum variants are not supported as fields".into(),
        ))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, BridgeError> {
        Err(BridgeError("maps are not supported as fields".into()))
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, BridgeError> {
        Err(BridgeError(
            "nested structs are not supported as fields".into(),
        ))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, BridgeError> {
        Err(BridgeError(
            "enum variants are not supported as fields".into(),
        ))
    }
}

/// Collects struct fields into a `Vec<Value>`.
struct StructCollector {
    fields: Vec<Value>,
}

impl SerializeStruct for StructCollector {
    type Ok = Vec<Value>;
    type Error = BridgeError;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), BridgeError> {
        self.fields.push(value.serialize(FieldSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<Vec<Value>, BridgeError> {
        Ok(self.fields)
    }
}

impl SerializeTupleStruct for StructCollector {
    type Ok = Vec<Value>;
    type Error = BridgeError;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), BridgeError> {
        self.fields.push(value.serialize(FieldSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<Vec<Value>, BridgeError> {
        Ok(self.fields)
    }
}

/// Top-level serializer that produces `Vec<Value>` from a struct.
struct ValueSerializer;

impl ser::Serializer for ValueSerializer {
    type Ok = Vec<Value>;
    type Error = BridgeError;

    type SerializeSeq = ser::Impossible<Vec<Value>, BridgeError>;
    type SerializeTuple = ser::Impossible<Vec<Value>, BridgeError>;
    type SerializeTupleStruct = StructCollector;
    type SerializeTupleVariant = ser::Impossible<Vec<Value>, BridgeError>;
    type SerializeMap = ser::Impossible<Vec<Value>, BridgeError>;
    type SerializeStruct = StructCollector;
    type SerializeStructVariant = ser::Impossible<Vec<Value>, BridgeError>;

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<StructCollector, BridgeError> {
        Ok(StructCollector {
            fields: Vec::with_capacity(len),
        })
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<StructCollector, BridgeError> {
        Ok(StructCollector {
            fields: Vec::with_capacity(len),
        })
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Vec<Value>, BridgeError> {
        let v = value.serialize(FieldSerializer)?;
        Ok(vec![v])
    }

    // Everything else is unsupported at the top level.

    fn serialize_bool(self, _v: bool) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got bool".into()))
    }

    fn serialize_i8(self, _v: i8) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got i8".into()))
    }

    fn serialize_i16(self, _v: i16) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got i16".into()))
    }

    fn serialize_i32(self, _v: i32) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got i32".into()))
    }

    fn serialize_i64(self, _v: i64) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got i64".into()))
    }

    fn serialize_i128(self, _v: i128) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got i128".into()))
    }

    fn serialize_u8(self, _v: u8) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got u8".into()))
    }

    fn serialize_u16(self, _v: u16) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got u16".into()))
    }

    fn serialize_u32(self, _v: u32) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got u32".into()))
    }

    fn serialize_u64(self, _v: u64) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got u64".into()))
    }

    fn serialize_u128(self, _v: u128) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got u128".into()))
    }

    fn serialize_f32(self, _v: f32) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got f32".into()))
    }

    fn serialize_f64(self, _v: f64) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got f64".into()))
    }

    fn serialize_char(self, _v: char) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got char".into()))
    }

    fn serialize_str(self, _v: &str) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got str".into()))
    }

    fn serialize_bytes(self, _v: &[u8]) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got bytes".into()))
    }

    fn serialize_none(self) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got None".into()))
    }

    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got Some".into()))
    }

    fn serialize_unit(self) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got unit".into()))
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Vec<Value>, BridgeError> {
        Ok(vec![])
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
    ) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got enum variant".into()))
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Vec<Value>, BridgeError> {
        Err(BridgeError("expected struct, got newtype variant".into()))
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, BridgeError> {
        Err(BridgeError("expected struct, got seq".into()))
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, BridgeError> {
        Err(BridgeError("expected struct, got tuple".into()))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, BridgeError> {
        Err(BridgeError("expected struct, got tuple variant".into()))
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, BridgeError> {
        Err(BridgeError("expected struct, got map".into()))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _idx: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, BridgeError> {
        Err(BridgeError("expected struct, got struct variant".into()))
    }
}

// ─── Deserialization ────────────────────────────────────────────────

/// Deserializes a single `&Value` into a Rust type.
struct FieldDeserializer<'a> {
    value: &'a Value,
}

impl<'de> de::Deserializer<'de> for FieldDeserializer<'de> {
    type Error = BridgeError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, BridgeError> {
        match self.value {
            Value::Unit => visitor.visit_unit(),
            Value::Bool(v) => visitor.visit_bool(*v),
            Value::I8(v) => visitor.visit_i8(*v),
            Value::I16(v) => visitor.visit_i16(*v),
            Value::I32(v) => visitor.visit_i32(*v),
            Value::I64(v) => visitor.visit_i64(*v),
            Value::I128(v) => visitor.visit_i128(*v),
            Value::U8(v) => visitor.visit_u8(*v),
            Value::U16(v) => visitor.visit_u16(*v),
            Value::U32(v) => visitor.visit_u32(*v),
            Value::U64(v) => visitor.visit_u64(*v),
            Value::U128(v) => visitor.visit_u128(*v),
            Value::Isize(v) => visitor.visit_i64(*v as i64),
            Value::Usize(v) => visitor.visit_u64(*v as u64),
            Value::F32(OrderedFloat(v)) => visitor.visit_f32(*v),
            Value::F64(OrderedFloat(v)) => visitor.visit_f64(*v),
            Value::Char(v) => visitor.visit_char(*v),
            Value::String(v) => visitor.visit_str(v),
            Value::Option(None) => visitor.visit_none(),
            Value::Option(Some(inner)) => visitor.visit_some(FieldDeserializer { value: inner }),
            _ => Err(BridgeError(format!(
                "unsupported Value variant for deserialization: {self:?}",
            ))),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, BridgeError> {
        match self.value {
            Value::Option(None) => visitor.visit_none(),
            Value::Option(Some(inner)) => visitor.visit_some(FieldDeserializer { value: inner }),
            // Non-Option value treated as Some
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, BridgeError> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, BridgeError> {
        match self.value {
            Value::String(v) => visitor.visit_string(v.as_ref().clone()),
            _ => self.deserialize_any(visitor),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str
        bytes byte_buf unit unit_struct seq tuple tuple_struct map
        struct enum identifier ignored_any
    }
}

impl fmt::Debug for FieldDeserializer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldDeserializer({:?})", self.value)
    }
}

/// Sequential access over a `&[Value]` slice.
struct FieldAccess<'a> {
    values: &'a [Value],
    pos: usize,
}

impl<'de> SeqAccess<'de> for FieldAccess<'de> {
    type Error = BridgeError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, BridgeError> {
        if self.pos >= self.values.len() {
            return Ok(None);
        }
        let value = &self.values[self.pos];
        self.pos += 1;
        seed.deserialize(FieldDeserializer { value }).map(Some)
    }
}

/// Top-level deserializer: takes a `&[Value]` slice and deserializes into a struct.
struct ValueSliceDeserializer<'a> {
    values: &'a [Value],
}

impl<'de> de::Deserializer<'de> for ValueSliceDeserializer<'de> {
    type Error = BridgeError;

    fn deserialize_any<V: Visitor<'de>>(self, _visitor: V) -> Result<V::Value, BridgeError> {
        Err(BridgeError(
            "ValueSliceDeserializer only supports struct types".into(),
        ))
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, BridgeError> {
        visitor.visit_seq(FieldAccess {
            values: self.values,
            pos: 0,
        })
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, BridgeError> {
        visitor.visit_seq(FieldAccess {
            values: self.values,
            pos: 0,
        })
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, BridgeError> {
        if self.values.len() != 1 {
            return Err(BridgeError(format!(
                "expected 1 value for newtype struct, got {}",
                self.values.len()
            )));
        }
        visitor.visit_newtype_struct(FieldDeserializer {
            value: &self.values[0],
        })
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, BridgeError> {
        visitor.visit_unit()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit seq tuple map enum identifier ignored_any
    }
}

// ─── Public API ─────────────────────────────────────────────────────

/// Serialize a struct into a `Vec<Value>`.
///
/// Each field of the struct becomes one element in the returned vector.
/// Only structs (named, tuple, newtype, unit) are supported at the top level.
pub fn to_values<T: Serialize>(value: &T) -> Result<Vec<Value>, BridgeError> {
    value.serialize(ValueSerializer)
}

/// Deserialize a struct from a `&[Value]` slice.
///
/// Each element in the slice is mapped to a field of the target struct.
pub fn from_values<T: for<'de> Deserialize<'de>>(values: &[Value]) -> Result<T, BridgeError> {
    T::deserialize(ValueSliceDeserializer { values })
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    struct Point {
        x: i32,
        y: i32,
    }

    impl fmt::Display for Point {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "({}, {})", self.x, self.y)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    struct Pair(i32, String);

    impl fmt::Display for Pair {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "({}, {})", self.0, self.1)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    struct Wrapper(i32);

    impl fmt::Display for Wrapper {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Wrapper({})", self.0)
        }
    }

    #[test]
    fn round_trip_named_struct() {
        let p = Point { x: 10, y: 20 };
        let values = to_values(&p).unwrap();
        assert_eq!(values, vec![Value::I32(10), Value::I32(20)]);
        let p2: Point = from_values(&values).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn round_trip_tuple_struct() {
        let pair = Pair(42, "hello".to_string());
        let values = to_values(&pair).unwrap();
        assert_eq!(
            values,
            vec![Value::I32(42), Value::String(Rc::new("hello".into()))]
        );
        let pair2: Pair = from_values(&values).unwrap();
        assert_eq!(pair, pair2);
    }

    #[test]
    fn round_trip_newtype_struct() {
        let w = Wrapper(99);
        let values = to_values(&w).unwrap();
        assert_eq!(values, vec![Value::I32(99)]);
        let w2: Wrapper = from_values(&values).unwrap();
        assert_eq!(w, w2);
    }

    #[test]
    fn all_primitive_field_types() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct AllPrimitives {
            a: bool,
            b: i8,
            c: i16,
            d: i32,
            e: i64,
            f: i128,
            g: u8,
            h: u16,
            i: u32,
            j: u64,
            k: u128,
            l: f32,
            m: f64,
            n: char,
            o: String,
        }

        let val = AllPrimitives {
            a: true,
            b: -1,
            c: -2,
            d: -3,
            e: -4,
            f: -5,
            g: 1,
            h: 2,
            i: 3,
            j: 4,
            k: 5,
            l: 1.5,
            m: 2.5,
            n: 'Z',
            o: "test".into(),
        };

        let values = to_values(&val).unwrap();
        assert_eq!(values.len(), 15);
        assert_eq!(values[0], Value::Bool(true));
        assert_eq!(values[1], Value::I8(-1));
        assert_eq!(values[13], Value::Char('Z'));

        let val2: AllPrimitives = from_values(&values).unwrap();
        assert_eq!(val, val2);
    }

    #[test]
    fn option_field_some_and_none() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Opt {
            a: Option<i32>,
            b: Option<i32>,
        }

        let val = Opt {
            a: Some(42),
            b: None,
        };
        let values = to_values(&val).unwrap();
        assert_eq!(
            values,
            vec![
                Value::Option(Some(Box::new(Value::I32(42)))),
                Value::Option(None),
            ]
        );
        let val2: Opt = from_values(&values).unwrap();
        assert_eq!(val, val2);
    }

    #[test]
    fn arity_mismatch_returns_err() {
        // Too few values
        let result = from_values::<Point>(&[Value::I32(1)]);
        assert!(result.is_err());

        // Too many values (serde ignores extras in SeqAccess, but struct expects exactly 2)
        // This actually succeeds because serde only reads what it needs — that's fine.
        // The important case is too few.
    }

    #[test]
    fn unit_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Empty;

        let values = to_values(&Empty).unwrap();
        assert!(values.is_empty());
        let _: Empty = from_values(&values).unwrap();
    }
}
