---
layout: home
hero:
  name: Ascent Interpreter
  text: Interactive Datalog
  tagline: An interpreter and REPL for Ascent Datalog programs
  actions:
    - theme: brand
      text: Getting Started
      link: /guide/getting-started
    - theme: alt
      text: Syntax Reference
      link: /reference/syntax

features:
  - title: Interactive REPL
    details: Add facts and rules incrementally, query relations, undo and retract — all from an interactive session.
  - title: Lattice Support
    details: First-class lattice relations with Dual type and join semantics for shortest-path and optimization problems.
  - title: Custom Types
    details: Register your own types with manual constructors or automatic serde-based registration.
  - title: Full Ascent Syntax
    details: Supports the complete Ascent language — aggregation, negation, disjunction, pattern matching, and generators.
---

## Quick Example

Compute transitive closure over a graph:

```
relation edge(i32, i32);
relation path(i32, i32);

edge(1, 2);
edge(2, 3);
edge(3, 4);

path(x, y) <-- edge(x, y);
path(x, z) <-- edge(x, y), path(y, z);
```

Run it:

```bash
$ cargo run -- graph.dl
path: {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
```

Or explore interactively in the REPL:

```
$ cargo run
>> edge(1, 2); edge(2, 3); edge(3, 4);
>> relation path(i32, i32);
>> path(x, y) <-- edge(x, y);
>> path(x, z) <-- edge(x, y), path(y, z);
>> :query path(1, _)
path(1, 2)
path(1, 3)
path(1, 4)
```
