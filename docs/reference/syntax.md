# Syntax Reference

This is the complete syntax reference for the Ascent Datalog language as supported by the interpreter.

## Declarations

### Relations

```
relation name(Type1, Type2, ...);
```

Declares a relation with typed columns. Column types follow Rust syntax.

```
relation edge(i32, i32);
relation node(String);
relation triple(i32, i32, i32);
```

### Lattices

```
lattice name(KeyType1, ..., ValueType);
```

Lattice relations use join semantics on their value columns instead of set-union. When a duplicate key is inserted, the value is merged using `lattice_join` (max for regular values, min for `Dual` values).

```
lattice shortest(i32, i32, Dual<i32>);
lattice best_score(i32, i32);
```

### Attributes

Relations and lattices can have attributes:

```
#[ds(btree)]
relation indexed(i32, i32);
```

## Facts

Facts are rules with no body — they insert tuples unconditionally.

```
edge(1, 2);
edge(2, 3);
node("alice");
```

Multiple facts can appear on one line:

```
edge(1, 2); edge(2, 3); edge(3, 4);
```

## Rules

Rules have the form:

```
head <-- body;
```

The head specifies what to insert; the body specifies conditions.

```
path(x, y) <-- edge(x, y);
path(x, z) <-- edge(x, y), path(y, z);
```

### Expressions in Heads

Head arguments can be arbitrary expressions:

```
double(x * 2) <-- number(x);
fac(n + 1, (n + 1) * f) <-- fac(n, f), if n < 20;
```

### Multi-Head Rules

A single rule can insert into multiple relations:

```
b(x), c(x) <-- a(x);
```

## Body Items

Body items are separated by commas. All items must be satisfied for the rule to fire.

### Clauses

Match against a relation:

```
edge(x, y)
```

Variables bind to column values. A variable appearing for the first time binds; subsequent occurrences require equality.

**Repeated variables** become equality checks:

```
self_loop(x) <-- edge(x, x);
// Desugars to: edge(x, __x0), if x == __x0
```

**Wildcards** match any value:

```
has_edge(x) <-- edge(x, _);
```

**Constants** filter to exact values:

```
starts_at_one(y) <-- edge(1, y);
```

### Pattern Arguments

Use `?` to destructure values with Rust patterns:

```
shortest(y, z, ?Dual(l))
option_rel(?Some(x))
```

Pattern arguments desugar to a fresh variable plus `if let`:

```
// shortest(y, z, ?Dual(l))  becomes:
// shortest(y, z, __pat), if let Dual(l) = __pat
```

### Conditions

**If expressions** — filter with a boolean condition:

```
if x > 0
if x % 2 == 0
```

**Let bindings** — bind a computed value:

```
let s = x + y
```

**If-let** — pattern match on a value:

```
if let Some(val) = compute(x)
```

### Generators

Produce values from an iterable:

```
for x in 0..10
for x in 1..=20
for x in [1, 2, 3]
```

Generators can include conditions:

```
for x in 1..20, if x % 2 == 0
```

### Aggregation

```
agg result = aggregator(bound_vars) in relation(args)
```

Computes an aggregate over matching tuples. Variables not bound inside the aggregation are grouped over.

```
agg m = min(x) in scores(x);
agg m = max(s) in score(player, s);
agg c = count() in edge(_, _);
agg s = sum(x) in numbers(x);
agg a = mean(x) in values(x);
```

See [Types & Aggregators](./types.md#built-in-aggregators) for the full aggregator list.

### Negation

```
!relation(args)
```

Succeeds when no matching tuples exist. Internally desugars to the `not` aggregator.

```
fizz(x) <-- divisible(x, 3), !divisible(x, 5);
```

### Disjunction

```
(alternative1 | alternative2)
```

Matches if any alternative matches. Desugars to multiple rules.

```
reachable(x) <-- (edge(x, _) | edge(_, x));

// Desugars to:
// reachable(x) <-- edge(x, _);
// reachable(x) <-- edge(_, x);
```

`||` is also accepted:

```
(a(x) || b(x))
```

## Expressions

Expressions can appear in head arguments, conditions, let bindings, and generator ranges.

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Remainder |

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `<=` | Less or equal |
| `>` | Greater than |
| `>=` | Greater or equal |

### Logical

| Operator | Description |
|----------|-------------|
| `&&` | Short-circuit AND |
| `\|\|` | Short-circuit OR |
| `!` | Logical NOT (unary) |

### Bitwise

| Operator | Description |
|----------|-------------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `<<` | Left shift |
| `>>` | Right shift |

### Other

| Expression | Description |
|------------|-------------|
| `-x` | Unary negation |
| `*x` | Dereference (identity in Datalog context) |
| `x as i64` | Type cast |
| `x..y` | Exclusive range |
| `x..=y` | Inclusive range |
| `(a, b, c)` | Tuple |
| `Some(x)` | Option constructor |
| `Dual(x)` | Lattice dual wrapper |
| `if c { a } else { b }` | Conditional expression |
| `x.clone()` | Method call |

## Stratification

Rules involving aggregation or negation require stratification. The interpreter automatically computes strata using dependency analysis:

1. Rules are grouped into strata based on their dependencies.
2. Within a stratum, semi-naive evaluation runs to fixpoint.
3. Strata execute in dependency order — a stratum depending on an aggregation over relation `R` runs after the stratum producing `R` reaches fixpoint.

This means aggregation and negation see a complete snapshot of the relations they depend on.

## Complete Example

FizzBuzz in Datalog:

```
relation number(i32);
relation divisible(i32, i32);
relation fizz(i32);
relation buzz(i32);
relation fizz_buzz(i32);

number(x) <-- for x in 1..16;
divisible(x, y) <-- number(x), for y in [3, 5], if x % y == 0;
fizz(x) <-- divisible(x, 3), !divisible(x, 5);
buzz(x) <-- divisible(x, 5), !divisible(x, 3);
fizz_buzz(x) <-- divisible(x, 3), divisible(x, 5);
```
