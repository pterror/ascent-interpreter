# Embedding ascent-interpreter as a Rust Library

This guide shows how to use `ascent-interpreter` as a library to parse, evaluate, and query Datalog programs from Rust code.

## 1. Setup

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
ascent-interpreter = { git = "https://github.com/pterror/ascent-interpreter" }
syn = "2"
```

For local development, use a path dependency instead:

```toml
[dependencies]
ascent-interpreter = { path = "../ascent-interpreter" }
syn = "2"
```

The `jit-asm` feature is on by default. To disable it (interpreter-only, no platform restriction):

```toml
ascent-interpreter = { path = "...", default-features = false }
```

## 2. Basic Usage

The full flow is: parse the source into an AST, lower to IR, create an engine, insert facts, run to fixpoint, and query results.

```rust
use ascent_interpreter::eval::{Engine, value::Value};
use ascent_interpreter::ir::Program;
use ascent_interpreter::syntax::AscentProgram;

fn main() {
    // 1. Parse the Datalog source
    let source = r#"
        relation edge(i32, i32);
        relation path(i32, i32);

        path(x, y) <-- edge(x, y);
        path(x, z) <-- edge(x, y), path(y, z);
    "#;
    let ast: AscentProgram = syn::parse_str(source).unwrap();

    // 2. Lower AST to IR
    let program = Program::from_ast(ast).unwrap();

    // 3. Create the evaluation engine
    let mut engine = Engine::new(program);

    // 4. Insert ground facts
    engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
    engine.insert("edge", vec![Value::I32(2), Value::I32(3)]).unwrap();
    engine.insert("edge", vec![Value::I32(3), Value::I32(4)]).unwrap();

    // 5. Run to fixpoint (semi-naive evaluation)
    engine.run().unwrap();

    // 6. Query results
    let path = engine.relation("path").unwrap();
    println!("path has {} tuples:", path.len());
    for tuple in path.iter() {
        println!("  ({}, {})", tuple[0], tuple[1]);
    }
}
```

Output:

```
path has 6 tuples:
  (1, 2)
  (1, 3)
  (1, 4)
  (2, 3)
  (2, 4)
  (3, 4)
```

## 3. Working with Values

The `Value` enum represents all runtime values. Create values with constructors, read them back with accessor methods.

### Creating values

```rust
use ascent_interpreter::eval::value::Value;

// Primitive types
let i = Value::I32(42);
let b = Value::Bool(true);
let c = Value::Char('x');
let u = Value::U64(1000);

// Strings (interned automatically)
let s = Value::string("hello");

// Unit
let unit = Value::Unit;
```

### Reading values from results

```rust
let path = engine.relation("path").unwrap();

for tuple in path.iter() {
    // Each tuple is a &[Value] slice
    let from = tuple[0].as_i32().unwrap();
    let to = tuple[1].as_i32().unwrap();
    println!("{from} -> {to}");
}
```

### Checking relation contents

```rust
// Number of tuples
let count = engine.relation("path").unwrap().len();

// Check if a specific tuple exists
let exists = engine.relation("path").unwrap().contains(
    &[Value::I32(1), Value::I32(3)]
);
```

### Display

`Value` implements `Display`, so you can print values directly:

```rust
for tuple in engine.relation("path").unwrap().iter() {
    println!("({}, {})", tuple[0], tuple[1]);
}
```

## 4. Incremental Evaluation

After an initial `run()`, you can add new facts and re-evaluate only the affected strata using `run_incremental()`.

```rust
use rustc_hash::FxHashSet;

let source = r#"
    relation edge(i32, i32);
    relation path(i32, i32);
    path(x, y) <-- edge(x, y);
    path(x, z) <-- edge(x, y), path(y, z);
"#;
let ast: AscentProgram = syn::parse_str(source).unwrap();
let program = Program::from_ast(ast).unwrap();
let mut engine = Engine::new(program);

// Initial facts and evaluation
engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
engine.insert("edge", vec![Value::I32(2), Value::I32(3)]).unwrap();
engine.run().unwrap();
assert_eq!(engine.relation("path").unwrap().len(), 3);

// Add a new edge and re-evaluate incrementally
engine.insert("edge", vec![Value::I32(3), Value::I32(4)]).unwrap();

let rederived = engine.run_incremental(&["edge"], &[]).unwrap();
// rederived contains the names of relations that were updated
assert_eq!(engine.relation("path").unwrap().len(), 6);
```

### Source tagging and retraction

You can tag facts with a source ID so you can retract them later:

```rust
// Create a source tag
let src = engine.intern_source("file_a.rs");

// Insert facts tagged with a source
engine.insert_with_source("edge", vec![Value::I32(10), Value::I32(20)], src).unwrap();
engine.insert_with_source("edge", vec![Value::I32(20), Value::I32(30)], src).unwrap();
engine.run().unwrap();

// Later: retract all facts from that source
let removed = engine.retract_source(src);
println!("removed {removed} tuples");

// Re-derive without those facts
engine.run_incremental(&["edge"], &["edge"]).unwrap();
```

## 5. Custom Types

Register custom Rust types with the engine so they can appear as values in relations.

Your type must implement `Clone + Eq + Hash + Ord + Debug + Display + Send + Sync`.

```rust
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Point {
    x: i32,
    y: i32,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Point({}, {})", self.x, self.y)
    }
}

// Register the type with a constructor and destructor
engine.register_type(
    "Point",
    // Constructor: build a Point from Value arguments
    |args| {
        let x = args.get(0)?.as_i32()?;
        let y = args.get(1)?.as_i32()?;
        Some(Value::custom(Point { x, y }))
    },
    // Destructor: extract fields from a Point value
    |val| {
        let p = Engine::downcast_custom::<Point>(val)?;
        Some(vec![Value::I32(p.x), Value::I32(p.y)])
    },
);
```

Once registered, the type can be used in Datalog source:

```datalog
relation points(Point);
points(Point(1, 2));
points(Point(3, 4));
```

## 6. JIT Compilation

The JIT compiler generates native x86-64 machine code for eligible stratum functions, providing significant speedups for compute-heavy programs.

::: warning Platform support
The JIT backend is gated at compile time to `target_arch = "x86_64"`. On other architectures (e.g., Apple Silicon / aarch64), the `jit-asm` feature compiles but JIT code is excluded — `enable_jit()` is not available and all rules run through the interpreter. No code changes are needed.
:::

### Feature flags

`jit-asm` is the default feature. To opt out:

```toml
ascent-interpreter = { path = "...", default-features = false }
```

To explicitly enable (no-op if already default, useful in transitive dependencies):

```toml
ascent-interpreter = { path = "...", features = ["jit-asm"] }
```

`jit-asm` implies `jit` and `specialized`.

### Enabling JIT at runtime

```rust
let mut engine = Engine::new(program);

// Enable JIT — returns Err if initialization fails
engine.enable_jit()?;

// Insert facts and run as normal
engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
engine.run();
```

Rules that the JIT cannot compile fall back to the interpreter automatically. No code changes are needed beyond calling `enable_jit()`.

### Sharing JIT state across engines

If you create multiple engines for the same program (e.g., for different datasets), you can share the compiled JIT code to avoid recompilation:

```rust
let mut engine1 = Engine::new(program.clone());
engine1.enable_jit()?;
engine1.run();

// Extract the shared JIT handle
let jit_handle = engine1.share_jit_compiler().unwrap();

// Inject into a second engine — no recompilation needed
let mut engine2 = Engine::new(program);
engine2.set_jit_compiler(jit_handle);
engine2.insert("edge", vec![Value::I32(10), Value::I32(20)]);
engine2.run();
```

## 7. Error Handling

### Parsing errors

`syn::parse_str` returns a `syn::Error` on malformed input:

```rust
let result = syn::parse_str::<AscentProgram>("not valid datalog !!!");
match result {
    Ok(ast) => { /* proceed */ }
    Err(e) => eprintln!("parse error: {e}"),
}
```

### Lowering errors

`Program::from_ast` returns `Result<Program, String>`. Errors include undefined relations, arity mismatches, and stratification failures:

```rust
let ast: AscentProgram = syn::parse_str(source).unwrap();
match Program::from_ast(ast) {
    Ok(program) => { /* proceed */ }
    Err(msg) => eprintln!("lowering error: {msg}"),
}
```

### JIT initialization errors

`enable_jit()` returns `Result<(), String>`:

```rust
if let Err(e) = engine.enable_jit() {
    eprintln!("JIT unavailable: {e}, falling back to interpreter");
    // engine still works — just runs interpreted
}
```

### Insert errors

`insert()` returns `Result<bool, EvalError>`. It returns `Ok(false)` if the tuple was a duplicate (already present). It returns `Err` in two cases:

- **Arity mismatch**: the tuple length does not match the relation's declared arity.
- **Unknown relation**: the relation name was not declared in the program.

```rust
use ascent_interpreter::eval::error::EvalError;

// Err — relation "foo" was never declared
let result = engine.insert("foo", vec![Value::I32(1)]);
assert!(matches!(result, Err(EvalError { .. })));

// Err — edge is arity 2 but we gave 3 values
let result = engine.insert("edge", vec![Value::I32(1), Value::I32(2), Value::I32(3)]);
assert!(matches!(result, Err(EvalError { .. })));
```

### Iteration limit

By default, evaluation stops after 10,000 fixpoint iterations (to guard against non-terminating programs). You can adjust this:

```rust
engine.set_max_iterations(100_000);
```

If the limit is hit, a warning is printed to stderr and partial results are returned.
