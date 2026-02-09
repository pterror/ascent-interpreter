//! Pre-compiled rule representation for fast evaluation.
//!
//! Transforms IR rules into an evaluation-friendly format:
//! - Variable names pre-interned to VarId (no HashMap lookup per tuple)
//! - Literals pre-evaluated to Values (no repeated parsing)
//! - Expressions flattened from syn::Expr to CExpr (simpler dispatch)

use ascent_ir::{Aggregation, BodyItem, Clause, ClauseArg, Condition, Generator, Rule};
use rustc_hash::FxHashSet;

use crate::eval::{Bindings, TypeRegistry, VarId, VarInterner};
use crate::expr::{eval_expr, eval_expr_with_registry, eval_lit};
use crate::value::Value;

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
}

/// Pre-compiled condition.
#[derive(Debug, Clone)]
pub(crate) enum CCondition {
    If(CExpr),
    IfLet { pattern: syn::Pat, expr: CExpr },
    Let { pattern: syn::Pat, expr: CExpr },
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

    CRule {
        heads: rule
            .heads
            .iter()
            .map(|h| CHeadClause {
                relation: h.relation.clone(),
                args: h.args.iter().map(|a| compile_expr(a, interner)).collect(),
            })
            .collect(),
        body: optimize_body(body),
    }
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
        all_args_bound: false,  // computed by optimize_body
        bound_cols: Vec::new(), // computed by optimize_body
        fresh_cols: Vec::new(), // computed by optimize_body
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

fn compile_agg_arg(expr: &syn::Expr, interner: &VarInterner) -> CAggArg {
    // Check if the arg is a simple variable (common case in aggregations)
    if let syn::Expr::Path(p) = expr
        && let Some(ident) = p.path.get_ident()
    {
        let name = ident.to_string();
        if name != "true" && name != "false" {
            return CAggArg::Var(interner.intern(&name));
        }
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

/// Optimize compiled rule body: reorder conditions and compute bound flags.
///
/// Pure `if expr` conditions are moved to the earliest position where all their
/// referenced variables are defined, filtering tuples before expensive joins.
/// Additionally, computes `all_args_bound` flags for clauses to enable the
/// fast-path contains check without runtime `find_bound_columns` allocation.
fn optimize_body(body: Vec<CBodyItem>) -> Vec<CBodyItem> {
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

    result
}

/// Compile a syn expression into the flat CExpr representation.
pub(crate) fn compile_expr(expr: &syn::Expr, interner: &VarInterner) -> CExpr {
    match expr {
        syn::Expr::Lit(lit) => match eval_lit(&lit.lit) {
            Some(val) => CExpr::Literal(val),
            None => CExpr::Dynamic(expr.clone()),
        },
        syn::Expr::Path(p) => {
            if let Some(ident) = p.path.get_ident() {
                let name = ident.to_string();
                match name.as_str() {
                    "true" => CExpr::Literal(Value::Bool(true)),
                    "false" => CExpr::Literal(Value::Bool(false)),
                    _ => CExpr::Var(interner.intern(&name)),
                }
            } else {
                CExpr::Dynamic(expr.clone())
            }
        }
        syn::Expr::Binary(bin) => match compile_binop(&bin.op) {
            Some(op) => {
                let left = compile_expr(&bin.left, interner);
                let right = compile_expr(&bin.right, interner);
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
            None => CExpr::Dynamic(expr.clone()),
        },
        syn::Expr::Unary(u) => match compile_unop(&u.op) {
            Some(op) => {
                let inner = compile_expr(&u.expr, interner);
                // Specialize `*var` (deref of variable — identity in Datalog)
                if matches!(op, CUnOp::Deref)
                    && let CExpr::Var(id) = inner
                {
                    return CExpr::DerefVar(id);
                }
                CExpr::Unary(op, Box::new(inner))
            }
            None => CExpr::Dynamic(expr.clone()),
        },
        syn::Expr::Paren(p) => compile_expr(&p.expr, interner),
        syn::Expr::Reference(r) => compile_expr(&r.expr, interner),
        syn::Expr::Range(r) => {
            let start = r
                .start
                .as_ref()
                .map(|e| Box::new(compile_expr(e, interner)));
            let end = r.end.as_ref().map(|e| Box::new(compile_expr(e, interner)));
            match (start, end) {
                (Some(s), Some(e)) => CExpr::Range {
                    start: s,
                    end: e,
                    inclusive: matches!(r.limits, syn::RangeLimits::Closed(_)),
                },
                _ => CExpr::Dynamic(expr.clone()),
            }
        }
        syn::Expr::Tuple(t) => {
            CExpr::Tuple(t.elems.iter().map(|e| compile_expr(e, interner)).collect())
        }
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = &*call.func {
                let name = p
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let args: Vec<CExpr> = call
                    .args
                    .iter()
                    .map(|a| compile_expr(a, interner))
                    .collect();
                CExpr::Call(name, args)
            } else {
                CExpr::Dynamic(expr.clone())
            }
        }
        syn::Expr::MethodCall(mc) => {
            let receiver = compile_expr(&mc.receiver, interner);
            let method = mc.method.to_string();
            let args: Vec<CExpr> = mc.args.iter().map(|a| compile_expr(a, interner)).collect();
            CExpr::MethodCall(Box::new(receiver), method, args)
        }
        syn::Expr::Cast(cast) => {
            if let syn::Type::Path(tp) = &*cast.ty {
                let target = tp
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                CExpr::Cast(Box::new(compile_expr(&cast.expr, interner)), target)
            } else {
                CExpr::Dynamic(expr.clone())
            }
        }
        syn::Expr::Array(arr) => CExpr::Array(
            arr.elems
                .iter()
                .map(|e| compile_expr(e, interner))
                .collect(),
        ),
        // Block, If, IfLet, etc. fall back to dynamic evaluation
        _ => CExpr::Dynamic(expr.clone()),
    }
}

fn compile_binop(op: &syn::BinOp) -> Option<CBinOp> {
    Some(match op {
        syn::BinOp::Add(_) => CBinOp::Add,
        syn::BinOp::Sub(_) => CBinOp::Sub,
        syn::BinOp::Mul(_) => CBinOp::Mul,
        syn::BinOp::Div(_) => CBinOp::Div,
        syn::BinOp::Rem(_) => CBinOp::Rem,
        syn::BinOp::BitAnd(_) => CBinOp::BitAnd,
        syn::BinOp::BitOr(_) => CBinOp::BitOr,
        syn::BinOp::BitXor(_) => CBinOp::BitXor,
        syn::BinOp::Shl(_) => CBinOp::Shl,
        syn::BinOp::Shr(_) => CBinOp::Shr,
        syn::BinOp::Eq(_) => CBinOp::Eq,
        syn::BinOp::Ne(_) => CBinOp::Ne,
        syn::BinOp::Lt(_) => CBinOp::Lt,
        syn::BinOp::Le(_) => CBinOp::Le,
        syn::BinOp::Gt(_) => CBinOp::Gt,
        syn::BinOp::Ge(_) => CBinOp::Ge,
        syn::BinOp::And(_) => CBinOp::And,
        syn::BinOp::Or(_) => CBinOp::Or,
        _ => return None,
    })
}

fn compile_unop(op: &syn::UnOp) -> Option<CUnOp> {
    Some(match op {
        syn::UnOp::Neg(_) => CUnOp::Neg,
        syn::UnOp::Not(_) => CUnOp::Not,
        syn::UnOp::Deref(_) => CUnOp::Deref,
        _ => return None,
    })
}

/// Resolve the aggregator name from the expression (pre-computed at compile time).
fn resolve_aggregator_name(expr: &syn::Expr) -> String {
    if let syn::Expr::Path(p) = expr {
        if let Some(ident) = p.path.get_ident() {
            return ident.to_string();
        }
        // Handle ::ascent::aggregators::not → "not"
        p.path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default()
    } else {
        String::new()
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
                _ => registry.and_then(|r| r.get(name).and_then(|ctor| ctor(&vals))),
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
        CExpr::Dynamic(expr) => match registry {
            Some(reg) => eval_expr_with_registry(expr, bindings, reg, interner),
            None => eval_expr(expr, bindings, interner),
        },
    }
}

fn eval_binary_op(op: CBinOp, left: &Value, right: &Value) -> Option<Value> {
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
        CBinOp::Lt => left.partial_cmp_val(right).map(|o| Value::Bool(o.is_lt())),
        CBinOp::Le => left.partial_cmp_val(right).map(|o| Value::Bool(o.is_le())),
        CBinOp::Gt => left.partial_cmp_val(right).map(|o| Value::Bool(o.is_gt())),
        CBinOp::Ge => left.partial_cmp_val(right).map(|o| Value::Bool(o.is_ge())),
        CBinOp::And | CBinOp::Or => unreachable!("handled by short-circuit path"),
    }
}

/// Fast path for i32 binary operations, avoiding Value method dispatch.
#[inline]
fn eval_i32_binary_op(op: CBinOp, l: i32, r: i32) -> Option<Value> {
    Some(match op {
        CBinOp::Add => Value::I32(l + r),
        CBinOp::Sub => Value::I32(l - r),
        CBinOp::Mul => Value::I32(l * r),
        CBinOp::Div => {
            if r != 0 {
                Value::I32(l / r)
            } else {
                return None;
            }
        }
        CBinOp::Rem => {
            if r != 0 {
                Value::I32(l % r)
            } else {
                return None;
            }
        }
        CBinOp::BitAnd => Value::I32(l & r),
        CBinOp::BitOr => Value::I32(l | r),
        CBinOp::BitXor => Value::I32(l ^ r),
        CBinOp::Shl => Value::I32(l << (r as u32)),
        CBinOp::Shr => Value::I32(l >> (r as u32)),
        CBinOp::Eq => Value::Bool(l == r),
        CBinOp::Ne => Value::Bool(l != r),
        CBinOp::Lt => Value::Bool(l < r),
        CBinOp::Le => Value::Bool(l <= r),
        CBinOp::Gt => Value::Bool(l > r),
        CBinOp::Ge => Value::Bool(l >= r),
        CBinOp::And | CBinOp::Or => unreachable!("handled by short-circuit path"),
    })
}
