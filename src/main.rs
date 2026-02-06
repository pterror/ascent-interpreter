//! CLI for the Ascent Datalog interpreter.
//!
//! Usage:
//!   ascent-interpreter              Start interactive REPL
//!   ascent-interpreter <file>       Run a program from a file

use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::{env, fs};

use ascent_eval::value::Value;
use ascent_eval::{Engine, RelationStorage};
use ascent_ir::Program;
use ascent_syntax::AscentProgram;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        None => repl(),
        Some("-h" | "--help") => {
            println!("Ascent Datalog Interpreter\n");
            println!("Usage: ascent-interpreter [file]\n");
            println!("  file    Run an Ascent program from a file");
            println!("  (none)  Start interactive REPL");
        }
        Some(path) => run_file(path),
    }
}

fn run_file(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        }
    };

    match eval_source(&source) {
        Ok((engine, program)) => dump_all(&engine, &program),
        Err(e) => {
            eprintln!("parse error: {e}");
            std::process::exit(1);
        }
    }
}

fn eval_source(source: &str) -> Result<(Engine, Program), syn::Error> {
    let ast: AscentProgram = syn::parse_str(source)?;
    let program = Program::from_ast(ast);
    let mut engine = Engine::new(&program);
    engine.run(&program);
    Ok((engine, program))
}

fn repl() {
    println!("Ascent Datalog Interpreter");
    println!("Type :help for commands, :quit to exit.\n");

    let stdin = io::stdin();
    let mut source = String::new();
    let mut line_buf = String::new();
    let mut prev_counts: HashMap<String, usize> = HashMap::new();

    loop {
        if line_buf.is_empty() {
            eprint!(">> ");
        } else {
            eprint!(".. ");
        }
        io::stderr().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            eprintln!();
            break;
        }

        let trimmed = line.trim();

        // Empty line in multi-line mode cancels input
        if trimmed.is_empty() {
            if !line_buf.is_empty() {
                line_buf.clear();
                eprintln!("(cancelled)");
            }
            continue;
        }

        // REPL commands (only at start of new input)
        if line_buf.is_empty() && trimmed.starts_with(':') {
            let (cmd, arg) = trimmed
                .split_once(' ')
                .map_or((trimmed, ""), |(a, b)| (a, b.trim()));

            match cmd {
                ":help" | ":h" => print_help(),
                ":quit" | ":exit" => break,
                ":clear" => {
                    source.clear();
                    prev_counts.clear();
                    eprintln!("(cleared)");
                }
                ":source" | ":src" => {
                    if source.is_empty() {
                        eprintln!("(empty)");
                    } else {
                        println!("{source}");
                    }
                }
                ":dump" | ":d" => {
                    if source.is_empty() {
                        eprintln!("(no program)");
                    } else if let Ok((engine, program)) = eval_source(&source) {
                        dump_all(&engine, &program);
                    }
                }
                ":query" | ":q" => {
                    if arg.is_empty() {
                        eprintln!("usage: :query <relation> or :query rel(pattern, ...)");
                    } else if source.is_empty() {
                        eprintln!("(no program)");
                    } else if let Ok((engine, _)) = eval_source(&source) {
                        query_relation(&engine, arg);
                    }
                }
                ":count" => {
                    if arg.is_empty() {
                        eprintln!("usage: :count <relation>");
                    } else if source.is_empty() {
                        eprintln!("(no program)");
                    } else if let Ok((engine, _)) = eval_source(&source) {
                        match engine.relation(arg) {
                            Some(rel) => eprintln!("  {arg}: {}", rel.len()),
                            None => eprintln!("  unknown relation: {arg}"),
                        }
                    }
                }
                ":relations" | ":rels" => {
                    if source.is_empty() {
                        eprintln!("(no relations)");
                    } else if let Ok((engine, program)) = eval_source(&source) {
                        list_relations(&engine, &program);
                    }
                }
                _ => eprintln!("unknown command: {cmd} (type :help for commands)"),
            }
            continue;
        }

        // Accumulate multi-line input
        line_buf.push_str(&line);

        // Wait for semicolon to indicate statement end
        if !trimmed.ends_with(';') {
            continue;
        }

        // Try adding to source
        let candidate = if source.is_empty() {
            line_buf.trim().to_string()
        } else {
            format!("{}\n{}", source, line_buf.trim())
        };
        line_buf.clear();

        match eval_source(&candidate) {
            Ok((engine, program)) => {
                source = candidate;
                show_changes(&engine, &program, &mut prev_counts);
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }
}

fn print_help() {
    eprintln!("Commands:");
    eprintln!("  :help             Show this help");
    eprintln!("  :relations        List all relations and their sizes");
    eprintln!("  :query <rel>      Show all tuples in a relation");
    eprintln!("  :query rel(1, _)  Filter tuples by pattern (_, int, \"str\", bool)");
    eprintln!("  :count <rel>      Show number of tuples in a relation");
    eprintln!("  :dump             Show all non-empty relations");
    eprintln!("  :source           Show accumulated program source");
    eprintln!("  :clear            Clear program and start over");
    eprintln!("  :quit             Exit the REPL");
    eprintln!();
    eprintln!("Enter Ascent statements (relations, rules, facts) ending with ';'.");
    eprintln!("Multi-line input continues until ';'. Empty line cancels.");
}

fn show_changes(engine: &Engine, program: &Program, prev_counts: &mut HashMap<String, usize>) {
    let mut any_change = false;
    let mut names: Vec<&str> = program.relations.keys().map(String::as_str).collect();
    names.sort();

    for name in names {
        let count = engine.relation(name).map_or(0, RelationStorage::len);
        let prev = prev_counts.get(name).copied().unwrap_or(0);
        prev_counts.insert(name.to_string(), count);

        if count != prev {
            let delta = count as isize - prev as isize;
            let sign = if delta > 0 { "+" } else { "" };
            eprintln!(
                "  {name}: {count} tuple{} ({sign}{delta})",
                if count == 1 { "" } else { "s" }
            );
            any_change = true;
        }
    }

    if !any_change {
        eprintln!("  (ok)");
    }
}

fn list_relations(engine: &Engine, program: &Program) {
    let mut names: Vec<&str> = program.relations.keys().map(String::as_str).collect();
    names.sort();

    for name in names {
        let count = engine.relation(name).map_or(0, RelationStorage::len);
        println!(
            "  {name}: {count} tuple{}",
            if count == 1 { "" } else { "s" }
        );
    }
}

fn query_relation(engine: &Engine, input: &str) {
    let (name, pattern) = parse_query(input);

    match engine.relation(name) {
        Some(rel) if rel.is_empty() => eprintln!("  (empty)"),
        Some(rel) => {
            if let Some(ref pats) = pattern {
                print_filtered(name, rel, pats);
            } else {
                print_tuples(name, rel);
            }
        }
        None => eprintln!("  unknown relation: {name}"),
    }
}

/// Parse a query like `path` or `path(1, _)` into (name, optional patterns).
fn parse_query(input: &str) -> (&str, Option<Vec<QueryPat>>) {
    if let Some(paren) = input.find('(') {
        let name = input[..paren].trim();
        let rest = input[paren + 1..].trim();
        let rest = rest.strip_suffix(')').unwrap_or(rest);
        let pats = rest.split(',').map(|s| parse_query_pat(s.trim())).collect();
        (name, Some(pats))
    } else {
        (input.trim(), None)
    }
}

#[derive(Debug)]
enum QueryPat {
    Wild,
    Int(i32),
    Str(String),
    Bool(bool),
}

fn parse_query_pat(s: &str) -> QueryPat {
    if s == "_" {
        return QueryPat::Wild;
    }
    if s == "true" {
        return QueryPat::Bool(true);
    }
    if s == "false" {
        return QueryPat::Bool(false);
    }
    if let Ok(n) = s.parse::<i32>() {
        return QueryPat::Int(n);
    }
    // Strip quotes for string literals
    if let Some(inner) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return QueryPat::Str(inner.to_string());
    }
    QueryPat::Str(s.to_string())
}

fn matches_pattern(value: &Value, pat: &QueryPat) -> bool {
    match pat {
        QueryPat::Wild => true,
        QueryPat::Int(n) => value.as_i64().is_some_and(|v| v == *n as i64),
        QueryPat::Str(s) => matches!(value, Value::String(vs) if vs.as_ref() == s),
        QueryPat::Bool(b) => matches!(value, Value::Bool(vb) if vb == b),
    }
}

fn print_filtered(name: &str, rel: &RelationStorage, pats: &[QueryPat]) {
    let mut tuples: Vec<&Vec<Value>> = rel
        .iter()
        .filter(|tuple| {
            tuple.len() >= pats.len()
                && tuple
                    .iter()
                    .zip(pats.iter())
                    .all(|(v, p)| matches_pattern(v, p))
        })
        .collect();
    tuples.sort_by(|a, b| cmp_tuples(a, b));

    if tuples.is_empty() {
        eprintln!("  (no matches)");
        return;
    }

    println!(
        "{name} ({} match{}):",
        tuples.len(),
        if tuples.len() == 1 { "" } else { "es" }
    );
    for tuple in tuples {
        print!("  (");
        for (i, val) in tuple.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{val:?}");
        }
        println!(")");
    }
}

fn dump_all(engine: &Engine, program: &Program) {
    let mut names: Vec<&str> = program.relations.keys().map(String::as_str).collect();
    names.sort();

    let mut first = true;
    for name in names {
        if let Some(rel) = engine.relation(name)
            && !rel.is_empty()
        {
            if !first {
                println!();
            }
            print_tuples(name, rel);
            first = false;
        }
    }
}

fn print_tuples(name: &str, rel: &RelationStorage) {
    println!(
        "{name} ({} tuple{}):",
        rel.len(),
        if rel.len() == 1 { "" } else { "s" }
    );

    let mut tuples: Vec<&Vec<Value>> = rel.iter().collect();
    tuples.sort_by(|a, b| cmp_tuples(a, b));

    for tuple in tuples {
        print!("  (");
        for (i, val) in tuple.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{val:?}");
        }
        println!(")");
    }
}

fn cmp_tuples(a: &[Value], b: &[Value]) -> Ordering {
    for (va, vb) in a.iter().zip(b.iter()) {
        if let Some(ord) = va.partial_cmp_val(vb)
            && ord != Ordering::Equal
        {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}
