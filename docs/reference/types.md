# Types & Aggregators

## Built-in Value Types

| Type | Rust Type | Literal Example |
|------|-----------|-----------------|
| Unit | `()` | — |
| Bool | `bool` | `true`, `false` |
| I8 | `i8` | `1i8` |
| I16 | `i16` | `1i16` |
| I32 | `i32` | `42` (default) |
| I64 | `i64` | `1i64` |
| I128 | `i128` | `1i128` |
| Isize | `isize` | `1isize` |
| U8 | `u8` | `1u8` |
| U16 | `u16` | `1u16` |
| U32 | `u32` | `1u32` |
| U64 | `u64` | `1u64` |
| U128 | `u128` | `1u128` |
| Usize | `usize` | `1usize` |
| F32 | `f32` | `1.0f32` |
| F64 | `f64` | `1.0f64` |
| Char | `char` | `'a'` |
| String | `String` | `"hello"` |
| Tuple | `(T, ...)` | `(1, 2)` |
| Option | `Option<T>` | `Some(1)`, `None` |
| Dual | `Dual<T>` | `Dual(42)` |
| Range | `T..T` | `1..10`, `1..=10` |
| Custom | user-defined | — |

Unsuffixed integer literals default to `i32`.

## Dual Type and Lattice Semantics

`Dual<T>` reverses the ordering of its inner value, enabling minimum-accumulation in lattice relations.

**Regular values:** lattice join = max (least upper bound)
**Dual values:** lattice join = min of the inner values

### Example: Shortest Path

```
relation edge(i32, i32, i32);
lattice shortest(i32, i32, Dual<i32>);

edge(1, 2, 10);
edge(2, 3, 20);
edge(1, 3, 50);

shortest(x, y, Dual(*w)) <-- edge(x, y, w);
shortest(x, z, Dual(w + l)) <-- edge(x, y, w), shortest(y, z, ?Dual(l));
```

Because `shortest` is a lattice and the value column is `Dual<i32>`, inserting a duplicate key merges by taking the **minimum** distance:

- `shortest(1, 3, Dual(30))` — via 1 &rarr; 2 &rarr; 3 (10 + 20)
- `shortest(1, 3, Dual(50))` — direct edge

The lattice join picks `Dual(30)` since `30 < 50`.

## Custom Types

The interpreter supports user-defined types via the Rust embedding API. Custom types appear as `Value::Custom(...)` and can be used in facts, rules, and pattern matching.

### Manual Registration

Register a type with explicit constructor and destructor:

```rust
use ascent_eval::Engine;
use ascent_eval::value::Value;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Point { x: i32, y: i32 }

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point({}, {})", self.x, self.y)
    }
}

let mut engine = Engine::new();
engine.register_type(
    "Point",
    |args| {
        let x = args.get(0)?.as_i32()?;
        let y = args.get(1)?.as_i32()?;
        Some(Value::custom(Point { x, y }))
    },
    |val| {
        let p = Engine::downcast_custom::<Point>(val)?;
        Some(vec![Value::I32(p.x), Value::I32(p.y)])
    },
);
```

The **constructor** takes a `&[Value]` and returns `Option<Value>`. It's called when the type name appears in facts or rule heads (e.g., `Point(1, 2)`).

The **destructor** takes a `&Value` and returns `Option<Vec<Value>>` with the fields. It's called for pattern matching in clauses (e.g., `?Point(x, y)`).

### Serde-Based Registration

With the `serde` feature, types implementing `Serialize + Deserialize` can be registered automatically:

```rust
use ascent_eval::Engine;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord,
         serde::Serialize, serde::Deserialize)]
struct Point { x: i32, y: i32 }

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point({}, {})", self.x, self.y)
    }
}

let mut engine = Engine::new();
engine.register_serde_type::<Point>("Point");
```

Serde registration automatically derives the constructor and destructor from the type's serialization format — struct fields become positional arguments in order.

### Using Custom Types in Ascent

Once registered, custom types work like built-in types:

```
relation points(Point);
points(Point(1, 2));
points(Point(3, 4));

relation x_coords(i32);
x_coords(x) <-- points(?Point(x, _));
```

## Built-in Aggregators

| Aggregator | Syntax | Description |
|------------|--------|-------------|
| `min` | `agg m = min(x) in rel(x)` | Minimum value. Empty if no tuples match. |
| `max` | `agg m = max(x) in rel(x)` | Maximum value. Empty if no tuples match. |
| `sum` | `agg s = sum(x) in rel(x)` | Sum of values. Empty if no tuples match. |
| `count` | `agg c = count() in rel(_)` | Number of matching tuples. Returns 0 for empty. |
| `mean` | `agg a = mean(x) in rel(x)` | Arithmetic mean (f64). Empty if no tuples match. |
| `not` | `!rel(x)` | Succeeds when no tuples match. Used via negation syntax. |

### Aggregation Examples

**Count tuples:**

```
relation edge(i32, i32);
relation total(i32);
edge(1, 2); edge(2, 3); edge(3, 4);
total(c) <-- agg c = count() in edge(_, _);
// total: {(3)}
```

**Grouped aggregation:**

```
relation score(i32, i32);
relation best(i32, i32);
score(1, 10); score(1, 20); score(2, 30); score(2, 15);
best(player, m) <-- score(player, _), agg m = max(s) in score(player, s);
// best: {(1, 20), (2, 30)}
```

Variables not bound inside the aggregation (like `player` above) act as group-by keys.

**Negation:**

```
relation a(i32);
relation b(i32);
relation only_a(i32);
a(1); a(2); a(3); b(2);
only_a(x) <-- a(x), !b(x);
// only_a: {(1), (3)}
```
