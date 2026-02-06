# Getting Started

## Running Programs

**REPL mode** — start an interactive session:

```bash
cargo run
```

**File mode** — run a `.dl` program and print all non-empty relations:

```bash
cargo run -- path/to/program.dl
```

**Help:**

```bash
cargo run -- -h
```

## REPL Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `:help` | `:h` | Show help |
| `:quit` | `:exit` | Exit the REPL |
| `:dump` | `:d` | Print all non-empty relations |
| `:query <rel>(pattern)` | `:q` | Query a relation with optional pattern filters |
| `:count <rel>` | | Show number of tuples in a relation |
| `:relations` | `:rels` | List all relations and their tuple counts |
| `:source` | `:src` | Print the accumulated program source |
| `:undo` | | Remove the last statement |
| `:retract fact(args);` | | Remove a specific fact |
| `:clear` | | Reset the program and all relations |

## Adding Facts and Rules

Enter Ascent statements directly at the `>>` prompt. Statements are accumulated and re-evaluated after each input.

```
>> relation edge(i32, i32);
>> edge(1, 2); edge(2, 3);
>> relation path(i32, i32);
>> path(x, y) <-- edge(x, y);
>> path(x, z) <-- edge(x, y), path(y, z);
```

Multi-line input is supported — if a line doesn't end with `;`, the REPL waits for more input.

## Querying Relations

Use `:query` (or `:q`) to filter relation contents:

```
>> :query path(1, _)
path(1, 2)
path(1, 3)
```

### Query Pattern Syntax

| Pattern | Matches |
|---------|---------|
| `_` | Any value (wildcard) |
| `42` | Exact integer |
| `"hello"` | Exact string |
| `true` / `false` | Exact boolean |

Bare identifiers (without quotes) are treated as string matches.

```
>> :query edge(_, 3)       # second column = 3
>> :query node("alice")    # string match
```

## Incremental Updates

The REPL supports incremental evaluation. You can modify the program and re-evaluate without starting over.

**Undo** the last statement:

```
>> edge(99, 100);
>> :undo
```

**Retract** a specific fact:

```
>> :retract edge(1, 2);
```

After undo or retract, derived relations are recomputed from the remaining facts and rules.

## Output Conventions

- Prompts and status messages go to **stderr**
- Relation data goes to **stdout**

This means you can pipe output to files or other programs:

```bash
cargo run -- program.dl > output.txt
cargo run -- program.dl | grep "path"
```
