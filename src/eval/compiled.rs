//! Pre-compiled rule representation for fast evaluation.
//!
//! Transforms IR rules into an evaluation-friendly format:
//! - Variable names pre-interned to VarId (no HashMap lookup per tuple)
//! - Literals pre-evaluated to Values (no repeated parsing)
//! - Expressions flattened from syn::Expr to CExpr (simpler dispatch)

use crate::ir::{
    Aggregation, BodyItem, Clause, ClauseArg, Condition, Generator, IrBinOp, IrExpr, IrLit,
    IrPattern, IrUnOp, Rule,
};
use rustc_hash::FxHashSet;

use crate::eval::bytecode::{BytecodeProgram, eval_bytecode, try_compile_to_bytecode};
use crate::eval::engine::{Bindings, TypeRegistry, VarId, VarInterner};
use crate::eval::expr::{eval_expr, eval_expr_with_registry};
use crate::eval::value::Value;

// ─── Compiled expression types ──────────────────────────────────────

/// Pre-compiled expression for fast evaluation.
#[derive(Debug, Clone)]
pub(crate) enum CExpr {
    /// Pre-evaluated literal value (no repeated parsing).
    Literal(Value),
    /// Variable reference (pre-interned, no HashMap lookup to resolve name).
    Var(VarId),
    /// Binary operation on two variables (no recursion needed).
    VarBinVar(CBinOp, VarId, VarId),
    /// Binary operation on variable and literal (no recursion needed).
    VarBinLit(CBinOp, VarId, Value),
    /// Binary operation on literal and variable (no recursion needed).
    LitBinVar(CBinOp, Value, VarId),
    /// Deref of a variable (`*x` — identity in Datalog context).
    DerefVar(VarId),
    /// Binary operation (general case, recursive).
    Binary(CBinOp, Box<CExpr>, Box<CExpr>),
    /// Unary operation.
    Unary(CUnOp, Box<CExpr>),
    /// Range expression.
    Range {
        start: Box<CExpr>,
        end: Box<CExpr>,
        inclusive: bool,
    },
    /// Tuple expression.
    Tuple(Vec<CExpr>),
    /// Function/constructor call (name pre-resolved).
    Call(String, Vec<CExpr>),
    /// Method call on a receiver.
    MethodCall(Box<CExpr>, String, Vec<CExpr>),
    /// Type cast.
    Cast(Box<CExpr>, String),
    /// Array expression (evaluated as tuple).
    Array(Vec<CExpr>),
    /// Flat bytecode for complex expressions (eliminates recursive tree-walk).
    Bytecode(BytecodeProgram),
    /// Fallback: expressions not yet compiled (blocks, if-let, etc).
    Dynamic(syn::Expr),
}

/// Pre-compiled binary operator.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

/// Pre-compiled unary operator.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CUnOp {
    Neg,
    Not,
    Deref,
}

// ─── Compiled rule types ────────────────────────────────────────────

/// Pre-compiled clause argument.
#[derive(Debug, Clone)]
pub(crate) enum CClauseArg {
    Var(VarId),
    Expr(CExpr),
}

/// Pre-compiled body clause.
#[derive(Debug, Clone)]
pub(crate) struct CClause {
    pub relation: String,
    pub args: Vec<CClauseArg>,
    pub conditions: Vec<CCondition>,
    /// True if all args will be bound by preceding body items at evaluation time.
    /// Enables fast-path contains check without `find_bound_columns` allocation.
    pub all_args_bound: bool,
    /// Pre-computed: column indices where the arg is bound by preceding body items.
    /// Used for index lookup without runtime `find_bound_columns` allocation.
    pub bound_cols: Vec<usize>,
    /// Pre-computed: column indices + VarId where the arg is a fresh variable.
    /// Used for direct binding without `match_clause` overhead.
    pub fresh_cols: Vec<(usize, VarId)>,
    /// Subset of fresh_cols: fresh variables referenced in any subsequent body item or head.
    /// A fresh var NOT in this set is a "wildcard" — bound but never used downstream.
    /// Used to dedup the outer full-scan: same meaningful_fresh_cols values → same result.
    pub meaningful_fresh_cols: Vec<(usize, VarId)>,
}

/// Pre-compiled condition.
#[derive(Debug, Clone)]
pub(crate) enum CCondition {
    If(CExpr),
    IfLet { pattern: IrPattern, expr: CExpr },
    Let { pattern: IrPattern, expr: CExpr },
}

/// Pre-compiled generator.
#[derive(Debug, Clone)]
pub(crate) struct CGenerator {
    pub vars: Vec<VarId>,
    pub expr: CExpr,
}

/// Pre-compiled aggregation argument (variable or expression).
#[derive(Debug, Clone)]
pub(crate) enum CAggArg {
    Var(VarId),
    Expr(CExpr),
}

/// Pre-compiled aggregation.
#[derive(Debug, Clone)]
pub(crate) struct CAggregation {
    pub result_vars: Vec<VarId>,
    pub aggregator_name: String,
    pub bound_vars: Vec<VarId>,
    pub relation: String,
    pub args: Vec<CAggArg>,
}

/// Pre-compiled body item.
#[derive(Debug, Clone)]
pub(crate) enum CBodyItem {
    Clause(CClause),
    Generator(CGenerator),
    Condition(CCondition),
    Aggregation(CAggregation),
}

/// Pre-compiled head clause.
#[derive(Debug, Clone)]
pub(crate) struct CHeadClause {
    pub relation: String,
    pub args: Vec<CExpr>,
}

/// Pre-compiled rule.
#[derive(Debug, Clone)]
pub(crate) struct CRule {
    pub heads: Vec<CHeadClause>,
    pub body: Vec<CBodyItem>,
}

// ─── Compilation ────────────────────────────────────────────────────

/// Compile an IR rule into the evaluation-friendly representation.
pub(crate) fn compile_rule(rule: &Rule, interner: &VarInterner) -> CRule {
    let body: Vec<CBodyItem> = rule
        .body
        .iter()
        .map(|item| compile_body_item(item, interner))
        .collect();

    let heads: Vec<CHeadClause> = rule
        .heads
        .iter()
        .map(|h| CHeadClause {
            relation: h.relation.clone(),
            args: h.args.iter().map(|a| compile_expr(a, interner)).collect(),
        })
        .collect();
    let body = optimize_body(body, &heads);
    CRule { heads, body }
}

fn compile_body_item(item: &BodyItem, interner: &VarInterner) -> CBodyItem {
    match item {
        BodyItem::Clause(c) => CBodyItem::Clause(compile_clause(c, interner)),
        BodyItem::Generator(g) => CBodyItem::Generator(compile_generator(g, interner)),
        BodyItem::Condition(c) => CBodyItem::Condition(compile_condition(c, interner)),
        BodyItem::Aggregation(a) => CBodyItem::Aggregation(compile_aggregation(a, interner)),
    }
}

fn compile_clause(clause: &Clause, interner: &VarInterner) -> CClause {
    CClause {
        relation: clause.relation.clone(),
        args: clause
            .args
            .iter()
            .map(|a| compile_clause_arg(a, interner))
            .collect(),
        conditions: clause
            .conditions
            .iter()
            .map(|c| compile_condition(c, interner))
            .collect(),
        all_args_bound: false,         // computed by optimize_body
        bound_cols: Vec::new(),        // computed by optimize_body
        fresh_cols: Vec::new(),        // computed by optimize_body
        meaningful_fresh_cols: Vec::new(), // computed by optimize_body
    }
}

fn compile_clause_arg(arg: &ClauseArg, interner: &VarInterner) -> CClauseArg {
    match arg {
        ClauseArg::Var(name) => CClauseArg::Var(interner.intern(name)),
        ClauseArg::Expr(expr) => CClauseArg::Expr(compile_expr(expr, interner)),
    }
}

fn compile_condition(cond: &Condition, interner: &VarInterner) -> CCondition {
    match cond {
        Condition::If(expr) => CCondition::If(compile_expr(expr, interner)),
        Condition::IfLet { pattern, expr } => CCondition::IfLet {
            pattern: pattern.clone(),
            expr: compile_expr(expr, interner),
        },
        Condition::Let { pattern, expr } => CCondition::Let {
            pattern: pattern.clone(),
            expr: compile_expr(expr, interner),
        },
    }
}

fn compile_generator(generator: &Generator, interner: &VarInterner) -> CGenerator {
    CGenerator {
        vars: generator.vars.iter().map(|v| interner.intern(v)).collect(),
        expr: compile_expr(&generator.expr, interner),
    }
}

fn compile_aggregation(agg: &Aggregation, interner: &VarInterner) -> CAggregation {
    CAggregation {
        result_vars: agg.result_vars.iter().map(|v| interner.intern(v)).collect(),
        aggregator_name: resolve_aggregator_name(&agg.aggregator),
        bound_vars: agg.bound_vars.iter().map(|v| interner.intern(v)).collect(),
        relation: agg.relation.clone(),
        args: agg
            .args
            .iter()
            .map(|a| compile_agg_arg(a, interner))
            .collect(),
    }
}

fn compile_agg_arg(expr: &IrExpr, interner: &VarInterner) -> CAggArg {
    // Check if the arg is a simple variable (common case in aggregations)
    if let IrExpr::Var(name) = expr {
        return CAggArg::Var(interner.intern(name));
    }
    CAggArg::Expr(compile_expr(expr, interner))
}

// ─── Body optimization ──────────────────────────────────────────────

/// Check if all variable references in a CExpr are in the defined set.
fn cexpr_vars_defined(expr: &CExpr, defined: &FxHashSet<VarId>) -> bool {
    match expr {
        CExpr::Literal(_) => true,
        CExpr::Var(id) | CExpr::DerefVar(id) => defined.contains(id),
        CExpr::VarBinVar(_, a, b) => defined.contains(a) && defined.contains(b),
        CExpr::VarBinLit(_, a, _) => defined.contains(a),
        CExpr::LitBinVar(_, _, b) => defined.contains(b),
        CExpr::Binary(_, l, r) => cexpr_vars_defined(l, defined) && cexpr_vars_defined(r, defined),
        CExpr::Unary(_, e) | CExpr::Cast(e, _) => cexpr_vars_defined(e, defined),
        CExpr::Range { start, end, .. } => {
            cexpr_vars_defined(start, defined) && cexpr_vars_defined(end, defined)
        }
        CExpr::Tuple(es) | CExpr::Array(es) => es.iter().all(|e| cexpr_vars_defined(e, defined)),
        CExpr::Call(_, args) => args.iter().all(|a| cexpr_vars_defined(a, defined)),
        CExpr::MethodCall(recv, _, args) => {
            cexpr_vars_defined(recv, defined) && args.iter().all(|a| cexpr_vars_defined(a, defined))
        }
        CExpr::Bytecode(bc) => bc.referenced_vars.iter().all(|id| defined.contains(id)),
        CExpr::Dynamic(_) => false,
    }
}

/// Collect all variable references from a CExpr.
fn cexpr_referenced_vars(expr: &CExpr, vars: &mut FxHashSet<VarId>) {
    match expr {
        CExpr::Literal(_) => {}
        CExpr::Var(id) | CExpr::DerefVar(id) => {
            vars.insert(*id);
        }
        CExpr::VarBinVar(_, a, b) => {
            vars.insert(*a);
            vars.insert(*b);
        }
        CExpr::VarBinLit(_, a, _) => {
            vars.insert(*a);
        }
        CExpr::LitBinVar(_, _, b) => {
            vars.insert(*b);
        }
        CExpr::Binary(_, l, r) => {
            cexpr_referenced_vars(l, vars);
            cexpr_referenced_vars(r, vars);
        }
        CExpr::Unary(_, e) | CExpr::Cast(e, _) => cexpr_referenced_vars(e, vars),
        CExpr::Range { start, end, .. } => {
            cexpr_referenced_vars(start, vars);
            cexpr_referenced_vars(end, vars);
        }
        CExpr::Tuple(es) | CExpr::Array(es) => {
            for e in es {
                cexpr_referenced_vars(e, vars);
            }
        }
        CExpr::Call(_, args) => {
            for a in args {
                cexpr_referenced_vars(a, vars);
            }
        }
        CExpr::MethodCall(recv, _, args) => {
            cexpr_referenced_vars(recv, vars);
            for a in args {
                cexpr_referenced_vars(a, vars);
            }
        }
        CExpr::Bytecode(bc) => {
            for id in &bc.referenced_vars {
                vars.insert(*id);
            }
        }
        CExpr::Dynamic(_) => {} // conservative: unknown vars
    }
}

/// Check if a CExpr contains any Dynamic (uncompiled) subexpressions.
fn cexpr_has_dynamic(expr: &CExpr) -> bool {
    match expr {
        CExpr::Dynamic(_) => true,
        CExpr::Literal(_)
        | CExpr::Var(_)
        | CExpr::DerefVar(_)
        | CExpr::VarBinVar(..)
        | CExpr::VarBinLit(..)
        | CExpr::LitBinVar(..) => false,
        CExpr::Binary(_, l, r) => cexpr_has_dynamic(l) || cexpr_has_dynamic(r),
        CExpr::Unary(_, e) | CExpr::Cast(e, _) => cexpr_has_dynamic(e),
        CExpr::Range { start, end, .. } => cexpr_has_dynamic(start) || cexpr_has_dynamic(end),
        CExpr::Tuple(es) | CExpr::Array(es) => es.iter().any(cexpr_has_dynamic),
        CExpr::Call(_, args) => args.iter().any(cexpr_has_dynamic),
        CExpr::MethodCall(recv, _, args) => {
            cexpr_has_dynamic(recv) || args.iter().any(cexpr_has_dynamic)
        }
        CExpr::Bytecode(_) => false,
    }
}

/// Track variables defined by a body item.
fn add_defined_vars(item: &CBodyItem, defined: &mut FxHashSet<VarId>) {
    match item {
        CBodyItem::Clause(c) => {
            for arg in &c.args {
                if let CClauseArg::Var(id) = arg {
                    defined.insert(*id);
                }
            }
        }
        CBodyItem::Generator(g) => {
            for &var_id in &g.vars {
                defined.insert(var_id);
            }
        }
        CBodyItem::Aggregation(a) => {
            for &var_id in &a.result_vars {
                defined.insert(var_id);
            }
        }
        CBodyItem::Condition(_) => {}
    }
}

/// Collect all VarIds referenced (read) in a body item — both bound inputs and
/// freshly-defined outputs. Used by Phase 4 of `optimize_body`.
fn body_item_all_vars(item: &CBodyItem, vars: &mut FxHashSet<VarId>) {
    match item {
        CBodyItem::Clause(c) => {
            for arg in &c.args {
                match arg {
                    CClauseArg::Var(id) => {
                        vars.insert(*id);
                    }
                    CClauseArg::Expr(expr) => cexpr_referenced_vars(expr, vars),
                }
            }
            for cond in &c.conditions {
                match cond {
                    CCondition::If(e) | CCondition::IfLet { expr: e, .. } | CCondition::Let { expr: e, .. } => {
                        cexpr_referenced_vars(e, vars);
                    }
                }
            }
        }
        CBodyItem::Generator(g) => {
            for &id in &g.vars {
                vars.insert(id);
            }
            cexpr_referenced_vars(&g.expr, vars);
        }
        CBodyItem::Condition(cond) => {
            match cond {
                CCondition::If(e) | CCondition::IfLet { expr: e, .. } | CCondition::Let { expr: e, .. } => {
                    cexpr_referenced_vars(e, vars);
                }
            }
        }
        CBodyItem::Aggregation(a) => {
            for &id in &a.bound_vars {
                vars.insert(id);
            }
            for arg in &a.args {
                match arg {
                    CAggArg::Var(id) => {
                        vars.insert(*id);
                    }
                    CAggArg::Expr(expr) => cexpr_referenced_vars(expr, vars),
                }
            }
        }
    }
}

/// Optimize compiled rule body: reorder conditions and compute bound flags.
///
/// Pure `if expr` conditions are moved to the earliest position where all their
/// referenced variables are defined, filtering tuples before expensive joins.
/// Additionally, computes `all_args_bound` flags for clauses to enable the
/// fast-path contains check without runtime `find_bound_columns` allocation.
fn optimize_body(body: Vec<CBodyItem>, heads: &[CHeadClause]) -> Vec<CBodyItem> {
    // Phase 1: Condition reordering.
    // Only reorder pure `if expr` conditions without Dynamic subexpressions,
    // since Dynamic expressions may reference variables we can't detect.
    let mut result = Vec::with_capacity(body.len());
    let mut pending: Vec<(FxHashSet<VarId>, CBodyItem)> = Vec::new();
    let mut defined = FxHashSet::default();

    for item in body {
        // Check if this is a reorderable condition
        if let CBodyItem::Condition(CCondition::If(ref expr)) = item
            && !cexpr_has_dynamic(expr)
        {
            let mut required = FxHashSet::default();
            cexpr_referenced_vars(expr, &mut required);
            if required.is_subset(&defined) {
                result.push(item);
            } else {
                pending.push((required, item));
            }
            continue;
        }

        // Non-reorderable item: track defined vars, push, flush pending
        add_defined_vars(&item, &mut defined);
        result.push(item);

        // Place any pending conditions whose vars are now satisfied
        let mut i = 0;
        while i < pending.len() {
            if pending[i].0.is_subset(&defined) {
                result.push(pending.remove(i).1);
            } else {
                i += 1;
            }
        }
    }

    // Append any remaining conditions at the end
    for (_, item) in pending {
        result.push(item);
    }

    // Phase 2: Merge standalone `if` conditions into the preceding clause.
    // After reordering, conditions are placed right after the clause that defines
    // their last variable. Merging them into clause.conditions eliminates a
    // recursion level in process_body_recursive (one less function call per match).
    // Only merge pure `if expr` conditions (not if-let/let which define new vars).
    let mut merged = Vec::with_capacity(result.len());
    for item in result {
        if matches!(&item, CBodyItem::Condition(CCondition::If(_)))
            && matches!(merged.last(), Some(CBodyItem::Clause(_)))
        {
            let CBodyItem::Condition(cond) = item else {
                unreachable!()
            };
            let Some(CBodyItem::Clause(clause)) = merged.last_mut() else {
                unreachable!()
            };
            clause.conditions.push(cond);
            continue;
        }
        merged.push(item);
    }
    let mut result = merged;

    // Phase 3: Compute all_args_bound, bound_cols, and fresh_cols for each clause.
    let mut defined = FxHashSet::default();
    for item in &mut result {
        if let CBodyItem::Clause(c) = item {
            c.all_args_bound = c.args.iter().all(|arg| match arg {
                CClauseArg::Var(id) => defined.contains(id),
                CClauseArg::Expr(expr) => cexpr_vars_defined(expr, &defined),
            });

            // Pre-compute bound/fresh column classification.
            // Track vars seen within this clause to handle repeated vars correctly:
            // first occurrence is fresh, subsequent occurrences are implicitly bound.
            let mut seen_in_clause = FxHashSet::default();
            let mut has_repeated = false;
            for (col, arg) in c.args.iter().enumerate() {
                match arg {
                    CClauseArg::Var(id) => {
                        if defined.contains(id) {
                            c.bound_cols.push(col);
                        } else if seen_in_clause.insert(*id) {
                            c.fresh_cols.push((col, *id));
                        } else {
                            // Same var appears twice in this clause — fall back to match_clause
                            has_repeated = true;
                            break;
                        }
                    }
                    CClauseArg::Expr(expr) => {
                        if cexpr_vars_defined(expr, &defined) {
                            c.bound_cols.push(col);
                        }
                    }
                }
            }
            if has_repeated {
                c.bound_cols.clear();
                c.fresh_cols.clear();
            }
        }
        add_defined_vars(item, &mut defined);
    }

    // Phase 4: Compute meaningful_fresh_cols.
    // Backwards pass: for each clause at index i, determine which of its fresh_cols vars
    // are actually referenced in body[i+1..] or heads. Fresh vars never used downstream
    // are wildcards — dedup on meaningful cols only.
    {
        let mut downstream: FxHashSet<VarId> = FxHashSet::default();
        for head in heads {
            for arg in &head.args {
                cexpr_referenced_vars(arg, &mut downstream);
            }
        }
        for i in (0..result.len()).rev() {
            if let CBodyItem::Clause(c) = &mut result[i] {
                c.meaningful_fresh_cols = c
                    .fresh_cols
                    .iter()
                    .filter(|(_, var_id)| downstream.contains(var_id))
                    .copied()
                    .collect();
            }
            let item = &result[i];
            let mut item_vars: FxHashSet<VarId> = FxHashSet::default();
            body_item_all_vars(item, &mut item_vars);
            downstream.extend(item_vars);
        }
    }

    result
}

/// Compile an IR expression into the flat CExpr representation.
///
/// Complex expressions (nested Binary/Unary trees) are additionally compiled
/// to bytecode for flat evaluation without recursive function calls.
pub(crate) fn compile_expr(expr: &IrExpr, interner: &VarInterner) -> CExpr {
    let cexpr = compile_expr_inner(expr, interner);
    // Try to compile complex expressions to bytecode
    if let Some(bytecode) = try_compile_to_bytecode(&cexpr) {
        return CExpr::Bytecode(bytecode);
    }
    cexpr
}

fn compile_expr_inner(expr: &IrExpr, interner: &VarInterner) -> CExpr {
    match expr {
        IrExpr::Lit(lit) => CExpr::Literal(ir_lit_to_value(lit)),
        IrExpr::Var(name) => CExpr::Var(interner.intern(name)),
        IrExpr::Binary(ir_op, left, right) => {
            let op = ir_binop_to_cbinop(*ir_op);
            let left = compile_expr_inner(left, interner);
            let right = compile_expr_inner(right, interner);
            // Specialize common patterns to avoid recursive eval_cexpr calls
            match (&left, &right) {
                (CExpr::Var(a), CExpr::Var(b)) => CExpr::VarBinVar(op, *a, *b),
                (CExpr::Var(a), CExpr::Literal(v)) => CExpr::VarBinLit(op, *a, v.clone()),
                (CExpr::Literal(v), CExpr::Var(b)) => CExpr::LitBinVar(op, v.clone(), *b),
                // Also catch DerefVar patterns: `*x op y`, `x op *y`
                (CExpr::DerefVar(a), CExpr::Var(b)) | (CExpr::Var(a), CExpr::DerefVar(b)) => {
                    CExpr::VarBinVar(op, *a, *b)
                }
                (CExpr::DerefVar(a), CExpr::Literal(v)) => CExpr::VarBinLit(op, *a, v.clone()),
                (CExpr::Literal(v), CExpr::DerefVar(b)) => CExpr::LitBinVar(op, v.clone(), *b),
                (CExpr::DerefVar(a), CExpr::DerefVar(b)) => CExpr::VarBinVar(op, *a, *b),
                _ => CExpr::Binary(op, Box::new(left), Box::new(right)),
            }
        }
        IrExpr::Unary(ir_op, inner) => {
            let op = ir_unop_to_cunop(*ir_op);
            let inner = compile_expr_inner(inner, interner);
            // Specialize `*var` (deref of variable — identity in Datalog)
            if matches!(op, CUnOp::Deref)
                && let CExpr::Var(id) = inner
            {
                return CExpr::DerefVar(id);
            }
            CExpr::Unary(op, Box::new(inner))
        }
        IrExpr::Range {
            start,
            end,
            inclusive,
        } => CExpr::Range {
            start: Box::new(compile_expr_inner(start, interner)),
            end: Box::new(compile_expr_inner(end, interner)),
            inclusive: *inclusive,
        },
        IrExpr::Tuple(elems) => {
            CExpr::Tuple(elems.iter().map(|e| compile_expr_inner(e, interner)).collect())
        }
        IrExpr::Call(name, args) => {
            let args: Vec<CExpr> = args.iter().map(|a| compile_expr_inner(a, interner)).collect();
            CExpr::Call(name.clone(), args)
        }
        IrExpr::MethodCall(receiver, method, args) => {
            let receiver = compile_expr_inner(receiver, interner);
            let args: Vec<CExpr> = args.iter().map(|a| compile_expr_inner(a, interner)).collect();
            CExpr::MethodCall(Box::new(receiver), method.clone(), args)
        }
        IrExpr::Cast(inner, target) => {
            CExpr::Cast(Box::new(compile_expr_inner(inner, interner)), target.clone())
        }
        IrExpr::Array(elems) => {
            CExpr::Array(elems.iter().map(|e| compile_expr_inner(e, interner)).collect())
        }
        IrExpr::Raw(raw) => {
            // Re-parse to syn::Expr for Dynamic fallback
            match syn::parse_str::<syn::Expr>(raw) {
                Ok(syn_expr) => CExpr::Dynamic(syn_expr),
                Err(_) => CExpr::Literal(Value::Unit), // should not happen
            }
        }
    }
}

/// Convert an IR literal to a Value.
pub(crate) fn ir_lit_to_value(lit: &IrLit) -> Value {
    match lit {
        IrLit::Int(n, suffix) => match suffix.as_deref() {
            Some("i8") => Value::I8(*n as i8),
            Some("i16") => Value::I16(*n as i16),
            Some("i32") => Value::I32(*n as i32),
            Some("i64") => Value::I64(*n as i64),
            Some("i128") => Value::I128(*n),
            Some("isize") => Value::Isize(*n as isize),
            Some("u8") => Value::U8(*n as u8),
            Some("u16") => Value::U16(*n as u16),
            Some("u32") => Value::U32(*n as u32),
            Some("u64") => Value::U64(*n as u64),
            Some("u128") => Value::U128(*n as u128),
            Some("usize") => Value::Usize(*n as usize),
            // No suffix or unrecognized: default to i32, widening as needed
            _ => {
                if let Ok(v) = i32::try_from(*n) {
                    Value::I32(v)
                } else if let Ok(v) = i64::try_from(*n) {
                    Value::I64(v)
                } else {
                    Value::I128(*n)
                }
            }
        }
        IrLit::Float(f) => Value::F64(crate::eval::value::OrderedFloat(*f)),
        IrLit::Bool(b) => Value::Bool(*b),
        IrLit::Char(c) => Value::Char(*c),
        IrLit::String(s) => Value::string(s.clone()),
    }
}

fn ir_binop_to_cbinop(op: IrBinOp) -> CBinOp {
    match op {
        IrBinOp::Add => CBinOp::Add,
        IrBinOp::Sub => CBinOp::Sub,
        IrBinOp::Mul => CBinOp::Mul,
        IrBinOp::Div => CBinOp::Div,
        IrBinOp::Rem => CBinOp::Rem,
        IrBinOp::BitAnd => CBinOp::BitAnd,
        IrBinOp::BitOr => CBinOp::BitOr,
        IrBinOp::BitXor => CBinOp::BitXor,
        IrBinOp::Shl => CBinOp::Shl,
        IrBinOp::Shr => CBinOp::Shr,
        IrBinOp::Eq => CBinOp::Eq,
        IrBinOp::Ne => CBinOp::Ne,
        IrBinOp::Lt => CBinOp::Lt,
        IrBinOp::Le => CBinOp::Le,
        IrBinOp::Gt => CBinOp::Gt,
        IrBinOp::Ge => CBinOp::Ge,
        IrBinOp::And => CBinOp::And,
        IrBinOp::Or => CBinOp::Or,
    }
}

fn ir_unop_to_cunop(op: IrUnOp) -> CUnOp {
    match op {
        IrUnOp::Neg => CUnOp::Neg,
        IrUnOp::Not => CUnOp::Not,
        IrUnOp::Deref => CUnOp::Deref,
    }
}

/// Resolve the aggregator name from the IR expression (pre-computed at compile time).
fn resolve_aggregator_name(expr: &IrExpr) -> String {
    match expr {
        IrExpr::Var(name) => name.clone(),
        IrExpr::Call(name, _) => name.clone(),
        IrExpr::Raw(raw) => {
            // Try to extract path from raw expression
            if let Ok(syn::Expr::Path(p)) = syn::parse_str::<syn::Expr>(raw) {
                p.path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

// ─── Evaluation ─────────────────────────────────────────────────────

/// Evaluate a pre-compiled expression.
pub(crate) fn eval_cexpr(
    expr: &CExpr,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    match expr {
        CExpr::Literal(val) => Some(val.clone()),
        CExpr::Var(var_id) | CExpr::DerefVar(var_id) => bindings.get(var_id).cloned(),
        // Specialized fast paths: binary ops on vars/literals without recursion
        CExpr::VarBinVar(op, a, b) => {
            let l = bindings.get(a)?;
            let r = bindings.get(b)?;
            eval_binary_op(*op, l, r)
        }
        CExpr::VarBinLit(op, a, v) => {
            let l = bindings.get(a)?;
            eval_binary_op(*op, l, v)
        }
        CExpr::LitBinVar(op, v, b) => {
            let r = bindings.get(b)?;
            eval_binary_op(*op, v, r)
        }
        CExpr::Binary(op, left, right) => {
            // Short-circuit for logical operators
            if matches!(op, CBinOp::And | CBinOp::Or) {
                let l = eval_cexpr(left, bindings, registry, interner)?;
                return match op {
                    CBinOp::And => {
                        if !l.as_bool()? {
                            Some(Value::Bool(false))
                        } else {
                            let r = eval_cexpr(right, bindings, registry, interner)?;
                            Some(Value::Bool(r.as_bool()?))
                        }
                    }
                    CBinOp::Or => {
                        if l.as_bool()? {
                            Some(Value::Bool(true))
                        } else {
                            let r = eval_cexpr(right, bindings, registry, interner)?;
                            Some(Value::Bool(r.as_bool()?))
                        }
                    }
                    _ => unreachable!(),
                };
            }
            let l = eval_cexpr(left, bindings, registry, interner)?;
            let r = eval_cexpr(right, bindings, registry, interner)?;
            eval_binary_op(*op, &l, &r)
        }
        CExpr::Unary(op, inner) => {
            let v = eval_cexpr(inner, bindings, registry, interner)?;
            match op {
                CUnOp::Neg => v.neg(),
                CUnOp::Not => v.not(),
                CUnOp::Deref => Some(v), // identity in Datalog context
            }
        }
        CExpr::Range {
            start,
            end,
            inclusive,
        } => {
            let s = eval_cexpr(start, bindings, registry, interner)?;
            let e = eval_cexpr(end, bindings, registry, interner)?;
            Some(Value::Range {
                start: Box::new(s),
                end: Box::new(e),
                inclusive: *inclusive,
            })
        }
        CExpr::Tuple(exprs) => {
            let vals: Option<Vec<Value>> = exprs
                .iter()
                .map(|e| eval_cexpr(e, bindings, registry, interner))
                .collect();
            vals.map(Value::tuple)
        }
        CExpr::Call(name, args) => {
            let vals: Option<Vec<Value>> = args
                .iter()
                .map(|a| eval_cexpr(a, bindings, registry, interner))
                .collect();
            let vals = vals?;
            match name.as_str() {
                "Some" => vals
                    .into_iter()
                    .next()
                    .map(|v| Value::Option(Some(Box::new(v)))),
                "None" => Some(Value::Option(None)),
                "Dual" => vals.into_iter().next().map(|v| Value::Dual(Box::new(v))),
                _ => registry.and_then(|r| r.constructor(name).and_then(|ctor| ctor(&vals))),
            }
        }
        CExpr::MethodCall(receiver, method, args) => {
            let recv = eval_cexpr(receiver, bindings, registry, interner)?;
            let _arg_vals: Option<Vec<Value>> = args
                .iter()
                .map(|a| eval_cexpr(a, bindings, registry, interner))
                .collect();
            match method.as_str() {
                "abs" => recv.abs(),
                "clone" => Some(recv),
                _ => None,
            }
        }
        CExpr::Cast(inner, target) => {
            let v = eval_cexpr(inner, bindings, registry, interner)?;
            v.cast_to(target)
        }
        CExpr::Array(exprs) => {
            let vals: Option<Vec<Value>> = exprs
                .iter()
                .map(|e| eval_cexpr(e, bindings, registry, interner))
                .collect();
            vals.map(Value::tuple)
        }
        CExpr::Bytecode(bc) => eval_bytecode(bc, bindings),
        CExpr::Dynamic(expr) => match registry {
            Some(reg) => eval_expr_with_registry(expr, bindings, reg, interner),
            None => eval_expr(expr, bindings, interner),
        },
    }
}

pub(crate) fn eval_binary_op(op: CBinOp, left: &Value, right: &Value) -> Option<Value> {
    // Fast path for i32 operands (the most common case in Datalog programs).
    // Avoids double dispatch through Value method + 14-way type match.
    if let (&Value::I32(l), &Value::I32(r)) = (left, right) {
        return eval_i32_binary_op(op, l, r);
    }
    match op {
        CBinOp::Add => left.add(right),
        CBinOp::Sub => left.sub(right),
        CBinOp::Mul => left.mul(right),
        CBinOp::Div => left.div(right),
        CBinOp::Rem => left.rem(right),
        CBinOp::BitAnd => left.bitand(right),
        CBinOp::BitOr => left.bitor(right),
        CBinOp::BitXor => left.bitxor(right),
        CBinOp::Shl => left.shl(right),
        CBinOp::Shr => left.shr(right),
        CBinOp::Eq => Some(Value::Bool(left == right)),
        CBinOp::Ne => Some(Value::Bool(left != right)),
        CBinOp::Lt => left.try_cmp(right).map(|o| Value::Bool(o.is_lt())),
        CBinOp::Le => left.try_cmp(right).map(|o| Value::Bool(o.is_le())),
        CBinOp::Gt => left.try_cmp(right).map(|o| Value::Bool(o.is_gt())),
        CBinOp::Ge => left.try_cmp(right).map(|o| Value::Bool(o.is_ge())),
        CBinOp::And | CBinOp::Or => unreachable!("handled by short-circuit path"),
    }
}

/// Fast path for i32 binary operations, avoiding Value method dispatch.
#[inline]
fn eval_i32_binary_op(op: CBinOp, l: i32, r: i32) -> Option<Value> {
    Some(match op {
        CBinOp::Add => Value::I32(l.wrapping_add(r)),
        CBinOp::Sub => Value::I32(l.wrapping_sub(r)),
        CBinOp::Mul => Value::I32(l.wrapping_mul(r)),
        CBinOp::Div => return l.checked_div(r).map(Value::I32),
        CBinOp::Rem => return l.checked_rem(r).map(Value::I32),
        CBinOp::BitAnd => Value::I32(l & r),
        CBinOp::BitOr => Value::I32(l | r),
        CBinOp::BitXor => Value::I32(l ^ r),
        CBinOp::Shl => return l.checked_shl(r as u32).map(Value::I32),
        CBinOp::Shr => return l.checked_shr(r as u32).map(Value::I32),
        CBinOp::Eq => Value::Bool(l == r),
        CBinOp::Ne => Value::Bool(l != r),
        CBinOp::Lt => Value::Bool(l < r),
        CBinOp::Le => Value::Bool(l <= r),
        CBinOp::Gt => Value::Bool(l > r),
        CBinOp::Ge => Value::Bool(l >= r),
        CBinOp::And | CBinOp::Or => unreachable!("handled by short-circuit path"),
    })
}
