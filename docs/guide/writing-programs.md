# Writing Programs

This guide walks through common Datalog patterns using the Ascent language, building from simple facts and rules up to aggregation, negation, and lattice relations.

## Facts and Rules

A Datalog program is built from **relations** (typed tables), **facts** (ground tuples), and **rules** (implications).

```
relation edge(i32, i32);

edge(1, 2);
edge(2, 3);
edge(3, 4);
```

A rule derives new tuples from existing ones. The head is on the left of `<--`, the body on the right:

```
relation path(i32, i32);

path(x, y) <-- edge(x, y);
path(x, z) <-- edge(x, y), path(y, z);
```

The interpreter runs rules to a **fixpoint**: it keeps applying rules until no new tuples are produced. This gives transitive closure automatically.

## Conditions and Guards

Use `if` to filter tuples:

```
relation number(i32);
relation even(i32);

number(x) <-- for x in 0..20;
even(x)   <-- number(x), if x % 2 == 0;
```

Use `let` to bind intermediate values:

```
relation word(String);
relation long_word(String, i32);

long_word(w, len) <-- word(w), let len = w.len() as i32, if len > 5;
```

Use `if let` for pattern matching:

```
relation maybe_value(Option<i32>);
relation present_value(i32);

present_value(x) <-- maybe_value(opt), if let Some(x) = opt;
```

## Generators

Generate tuples from a Rust iterator using `for`:

```
relation square(i32, i32);

square(x, x * x) <-- for x in 1..=10;
```

Generators can be combined with other body items:

```
relation pair(i32, i32);
relation close_pair(i32, i32);

pair(a, b) <-- for a in 0..5, for b in 0..5;
close_pair(a, b) <-- pair(a, b), if (a - b).abs() <= 1;
```

## Pattern Arguments

Use `?` to match against a Rust pattern instead of binding a variable:

```
relation tagged(String, i32);
relation positive_tagged(String, i32);

positive_tagged(tag, n) <-- tagged(tag, ?n) if n > 0;
```

This is especially useful with enums:

```
relation item(String, Option<i32>);
relation present_item(String, i32);

present_item(name, val) <-- item(name, ?Some(val));
```

## Negation

Prefix a relation name with `!` to match tuples **not** in the relation:

```
relation node(i32);
relation edge(i32, i32);
relation isolated(i32);

isolated(x) <-- node(x), !edge(x, _), !edge(_, x);
```

Negation uses **stratified semantics**: the negated relation must be fully computed before the rule that uses it. The interpreter enforces this automatically.

## Aggregation

Compute aggregate values with `agg`:

```
relation score(String, i32);
relation top_score(i32);

top_score(m) <-- agg m = max(s) in score(_, s);
```

Group aggregates by binding keys before the `in`:

```
relation best_per_player(String, i32);

best_per_player(player, m) <-- agg m = max(s) in score(player, s);
```

Available aggregators: `min`, `max`, `sum`, `count`, `mean`, `not`.

Count tuples:

```
relation word(String);
relation word_count(i32);

word_count(n) <-- agg n = count() in word(_);
```

Check for absence (equivalent to negation):

```
relation flag(bool);
relation no_flags(bool);

no_flags(b) <-- agg b = not() in flag(_);
```

## Multiple Heads

A rule can produce tuples in multiple relations at once:

```
relation parent(String, String);
relation ancestor(String, String);
relation sibling(String, String);

ancestor(x, y), sibling(x, z) <-- parent(x, p), parent(z, p), parent(p, y);
```

## Lattice Relations

A `lattice` relation applies **join semantics** on its last column instead of set-union. When a duplicate key arrives, the value is merged (maximised for regular types, minimised for `Dual`).

Use `Dual<T>` for minimisation — useful for shortest paths:

```
relation edge(i32, i32, u32);   // from, to, weight
lattice shortest(i32, i32, Dual<u32>);  // from, to, distance

shortest(s, s, Dual(0)) <-- for s in 0..5;
shortest(x, z, Dual(d + w)) <-- edge(x, y, w), shortest(y, z, ?Dual(d));
```

The `?Dual(d)` pattern in the body extracts the inner value from the lattice element.

Query the result:

```
>> :query shortest(0, _)
shortest(0, 0, Dual(0))
shortest(0, 1, Dual(2))
shortest(0, 2, Dual(5))
```

## FizzBuzz Example

A self-contained example combining generators, conditions, and multiple relations:

```
relation number(i32);
relation fizz(i32);
relation buzz(i32);
relation fizzbuzz(i32);
relation plain(i32);

number(x) <-- for x in 1..=20;

fizzbuzz(x) <-- number(x), if x % 15 == 0;
fizz(x)     <-- number(x), if x % 3 == 0, if x % 5 != 0;
buzz(x)     <-- number(x), if x % 5 == 0, if x % 3 != 0;
plain(x)    <-- number(x), if x % 3 != 0, if x % 5 != 0;
```

## Graph Analysis Example

Compute in-degree for each node:

```
relation edge(i32, i32);
relation in_degree(i32, i32);

in_degree(node, n) <-- agg n = count() in edge(_, node);
```

Find nodes with no outgoing edges (sinks):

```
relation node(i32);
relation sink(i32);

sink(x) <-- node(x), !edge(x, _);
```

Find strongly connected components (reachability in both directions):

```
relation path(i32, i32);
relation scc(i32, i32);

path(x, y) <-- edge(x, y);
path(x, z) <-- edge(x, y), path(y, z);

scc(x, y) <-- path(x, y), path(y, x);
```
