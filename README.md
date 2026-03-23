# ascent-interpreter

Interpreter and JIT compiler for [Ascent](https://github.com/s-arash/ascent) Datalog programs.

<!-- badges -->

Ascent is a Datalog DSL embedded in Rust that compiles rules at `rustc` time via proc macros. This project provides an alternative execution model: parse and evaluate Ascent programs at runtime, without requiring Rust compilation. Useful for REPLs, LSP tooling, dynamic analysis, and embedding Datalog in applications.

## Features

- **Interactive REPL** with incremental evaluation, undo, retraction, and pattern-filtered queries
- **File execution** -- run `.dl` programs from the command line
- **JIT compilation** via a custom x86-64 assembler backend (near-parity with `ascent_macro` for eval-only time)
- **Semi-naive evaluation** with stratification
- **Aggregation** -- `count`, `sum`, `min`, `max`
- **Negation** (stratified)
- **Lattice relations** with user-defined join semantics
- **Custom types** via serde deserialization
- **String interning** for zero-cost string comparisons at runtime
- **Embeddable as a library** -- parse, insert facts, run, and query from Rust

## Quick Start

```bash
# Build (with JIT enabled)
cargo build --release --features jit

# Run a Datalog program from a file
cargo run --features jit -- program.dl

# Start the interactive REPL
cargo run --features jit
```

## Example

A complete triangle detection program (`triangles.dl`):

```datalog
relation edge(i32, i32);
relation triangle(i32, i32, i32);

edge(1, 2);
edge(2, 3);
edge(3, 4);
edge(1, 3);
edge(1, 4);
edge(2, 4);

triangle(a, b, c) <-- edge(a, b), edge(b, c), edge(a, c), if a < b, if b < c;
```

```
$ cargo run --features jit -- triangles.dl
triangle (4 tuples):
  (1, 2, 3)
  (1, 2, 4)
  (1, 3, 4)
  (2, 3, 4)
```

## Embedding as a Library

```rust
use ascent_eval::{Engine, value::Value};
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

let input = r#"
    relation edge(i32, i32);
    relation path(i32, i32);
    path(x, y) <-- edge(x, y);
    path(x, z) <-- edge(x, y), path(y, z);
"#;

let ast: AscentProgram = syn::parse_str(input).unwrap();
let program = Program::from_ast(ast).unwrap();
let mut engine = Engine::new(program);

// Insert initial facts
engine.insert("edge", vec![Value::I32(1), Value::I32(2)]).unwrap();
engine.insert("edge", vec![Value::I32(2), Value::I32(3)]).unwrap();

// Run to fixpoint
engine.run().unwrap();

// Query results
let path = engine.relation("path").unwrap();
assert_eq!(path.len(), 3); // (1,2), (2,3), (1,3)
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `:help` | Show available commands |
| `:relations` | List all relations and their sizes |
| `:query <rel>` | Show all tuples in a relation |
| `:query rel(1, _)` | Filter tuples by pattern (`_`, int, `"str"`, bool) |
| `:count <rel>` | Show number of tuples in a relation |
| `:dump` | Show all non-empty relations |
| `:undo` | Remove the last statement |
| `:retract fact(...)` | Remove a specific fact from the program |
| `:source` | Show accumulated program source |
| `:clear` | Clear program and start over |
| `:quit` | Exit the REPL |

Enter Ascent statements ending with `;`. Multi-line input continues until `;`. An empty line cancels the current input.

## Performance

The JIT backend targets wall-clock parity with `ascent_macro` (LLVM-compiled Datalog). At small working-set sizes (n=20), eval-only time is within 1.2--1.3x of `ascent_macro`. The JIT uses a custom x86-64 assembler (via `dynasm-rs`) with specialized data structures including Swiss tables and sorted merge intersection.

Run benchmarks with:

```bash
cargo bench --features jit
```

## Architecture

The project is split into four crates:

| Crate | Role |
|-------|------|
| `ascent-syntax` | Parser (syn-based) and desugaring of Ascent Datalog syntax |
| `ascent-ir` | Intermediate representation lowered from the AST |
| `ascent-eval` | Semi-naive evaluation engine, expression evaluator, and JIT compiler |
| `ascent-interpreter` | CLI binary (REPL + file execution) |

## Development

```bash
nix develop          # Enter dev shell (provides Rust toolchain)
cargo test           # Run tests
cargo clippy         # Lint
cargo bench          # Run benchmarks
cd docs && bun dev   # Local documentation site
```

## License

MIT OR Apache-2.0
