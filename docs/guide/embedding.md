# Embedding ascent-interpreter as a Rust Library

This guide shows how to use `ascent-eval` as a library to parse, evaluate, and query Datalog programs from Rust code.

## 1. Setup

Add the three crates to your `Cargo.toml`:

```toml
[dependencies]
ascent-eval = { git = "https://github.com/user/ascent-interpreter" }
ascent-ir = { git = "https://github.com/user/ascent-interpreter" }
ascent-syntax = { git = "https://github.com/user/ascent-interpreter" }
syn = "2"
```

For local development, use path dependencies instead:

```toml
[dependencies]
ascent-eval = { path = "../ascent-interpreter/crates/ascent-eval" }
ascent-ir = { path = "../ascent-interpreter/crates/ascent-ir" }
ascent-syntax = { path = "../ascent-interpreter/crates/ascent-syntax" }
syn = "2"
```

To enable the JIT compiler, add the `jit-asm` feature:

```toml
ascent-eval = { path = "...", features = ["jit-asm"] }
```

## 2. Basic Usage

The full flow is: parse the source into an AST, lower to IR, create an engine, insert facts, run to fixpoint, and query results.

```rust
use ascent_eval::{Engine, value::Value};
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

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
    let mut engine = Engine::new(&program);

    // 4. Insert ground facts
    engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
    engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
    engine.insert("edge", vec![Value::I32(3), Value::I32(4)]);

    // 5. Run to fixpoint (semi-naive evaluation)
    engine.run(&program);

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
use ascent_eval::value::Value;

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
let mut engine = Engine::new(&program);

// Initial facts and evaluation
engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
engine.insert("edge", vec![Value::I32(2), Value::I32(3)]);
engine.run(&program);
assert_eq!(engine.relation("path").unwrap().len(), 3);

// Add a new edge and re-evaluate incrementally
engine.insert("edge", vec![Value::I32(3), Value::I32(4)]);

let mut dirty = FxHashSet::default();
dirty.insert("edge".to_string());
let retracted = FxHashSet::default(); // no retractions

let rederived = engine.run_incremental(&program, &dirty, &retracted);
// rederived contains the names of relations that were updated
assert_eq!(engine.relation("path").unwrap().len(), 6);
```

### Source tagging and retraction

You can tag facts with a source ID so you can retract them later:

```rust
// Create a source tag
let src = engine.intern_source("file_a.rs");

// Insert facts tagged with a source
engine.insert_with_source("edge", vec![Value::I32(10), Value::I32(20)], src);
engine.insert_with_source("edge", vec![Value::I32(20), Value::I32(30)], src);
engine.run(&program);

// Later: retract all facts from that source
let removed = engine.retract_source(src);
println!("removed {removed} tuples");

// Re-derive without those facts
let mut dirty = FxHashSet::default();
dirty.insert("edge".to_string());
let mut retracted_rels = FxHashSet::default();
retracted_rels.insert("edge".to_string());
engine.run_incremental(&program, &dirty, &retracted_rels);
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

The JIT compiler generates native x86-64 machine code for eligible rules, providing significant speedups for compute-heavy programs.

### Feature flags

Enable the `jit-asm` feature in your `Cargo.toml`:

```toml
ascent-eval = { path = "...", features = ["jit-asm"] }
```

The `jit-asm` feature implies `jit` and `specialized`.

### Enabling JIT at runtime

```rust
let mut engine = Engine::new(&program);

// Enable JIT — returns Err if initialization fails
engine.enable_jit()?;

// Insert facts and run as normal
engine.insert("edge", vec![Value::I32(1), Value::I32(2)]);
engine.run(&program);
```

Rules that the JIT cannot compile fall back to the interpreter automatically. No code changes are needed beyond calling `enable_jit()`.

### Sharing JIT state across engines

If you create multiple engines for the same program (e.g., for different datasets), you can share the compiled JIT code to avoid recompilation:

```rust
let mut engine1 = Engine::new(&program);
engine1.enable_jit()?;
engine1.run(&program);

// Extract the shared JIT handle
let jit_handle = engine1.share_jit_compiler().unwrap();

// Inject into a second engine — no recompilation needed
let mut engine2 = Engine::new(&program);
engine2.set_jit_compiler(jit_handle);
engine2.insert("edge", vec![Value::I32(10), Value::I32(20)]);
engine2.run(&program);
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

### Insert warnings

`insert()` returns `false` and prints a warning to stderr in two cases:

- **Arity mismatch**: the tuple length does not match the relation's declared arity.
- **Unknown relation**: the relation name was not declared in the program.

```rust
// Returns false — relation "foo" was never declared
let ok = engine.insert("foo", vec![Value::I32(1)]);
assert!(!ok);

// Returns false — edge is arity 2 but we gave 3 values
let ok = engine.insert("edge", vec![Value::I32(1), Value::I32(2), Value::I32(3)]);
assert!(!ok);
```

### Iteration limit

By default, evaluation stops after 10,000 fixpoint iterations (to guard against non-terminating programs). You can adjust this:

```rust
engine.set_max_iterations(100_000);
```

If the limit is hit, a warning is printed to stderr and partial results are returned.
