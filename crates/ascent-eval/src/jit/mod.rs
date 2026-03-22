//! JIT compiler for rule bodies (asm backend only).
//!
//! Cranelift has been removed. Stage 4 (asm backend) is the sole JIT execution path.

pub(crate) mod layout;
pub mod rel_index;
pub mod storage;
#[cfg(feature = "specialized")]
pub(crate) mod packed_helpers;
#[cfg(feature = "jit-asm")]
mod asm_codegen;
#[cfg(test)]
mod tests;

use rustc_hash::FxHashMap;

#[cfg(feature = "specialized")]
use crate::compiled::{CBinOp, CClauseArg, CExpr, CUnOp};
use crate::compiled::{CAggArg, CAggregation, CBodyItem, CCondition, CRule};

impl std::fmt::Debug for JitCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiler")
            .field("asm_buffers", &self.asm_buffers.len())
            .field("stratum_stage4_fn_cache", &self.stratum_stage4_fn_cache.len())
            .finish()
    }
}

/// The JIT compiler (asm backend only).
#[cfg(feature = "jit-asm")]
pub struct JitCompiler {
    /// Cache of Stage 4 stratum functions (non-native asm).
    pub(crate) stratum_stage4_fn_cache: FxHashMap<usize, Option<packed_helpers::StratumStage4Fn>>,
    /// Cache of Stage 4 native stratum functions (JitRelData direct read path).
    pub(crate) stratum_stage4_native_fn_cache:
        FxHashMap<usize, Option<packed_helpers::StratumStage4Fn>>,
    /// Asm-backend stratum buffers: kept alive so the fn_ptr remains valid.
    pub(crate) asm_buffers: Vec<asm_codegen::AsmStratum>,
    /// Total number of interned variables (set by Engine before each compilation batch).
    pub(crate) var_count: usize,
    /// Peak dedup-table capacity observed per relation (relation name → cap).
    pub(crate) dedup_cap_hints: rustc_hash::FxHashMap<String, u32>,
    /// Peak tuple count observed per IDB head relation (relation name → count).
    pub(crate) tuple_count_hints: rustc_hash::FxHashMap<String, u32>,
    /// Pool of reusable Stage 4 runtimes (one per stratum key, non-native path only).
    /// Avoids the ~9 Box allocations in build_stratum_stage4_runtime on every fresh-engine
    /// hot-bench iteration; repopulate_runtime writes new engine pointers without allocating.
    #[cfg(feature = "specialized")]
    pub(crate) stratum_runtime_pool: rustc_hash::FxHashMap<usize, crate::eval::StratumStage4Runtime>,
    /// Cache of pre-sorted EDB `total` JitRelData projections across Engine instances.
    ///
    /// Keyed by (relation_name). Value is (tuple_count, data_hash, Box<JitRelData>) where
    /// data_hash is a FxHasher hash of the raw packed_data buffer. On a cache hit
    /// (count+hash match), the cached JitRelData is O(n) memcpy-cloned instead of
    /// rebuilding via O(n log n) sort + JitColIndex construction.
    ///
    /// Shared across Engine instances via Arc<Mutex<JitCompiler>>, so the cache survives
    /// Engine::new() calls in hot-bench iterations that reuse a compiled JitCompiler.
    #[cfg(all(feature = "jit-asm", feature = "specialized"))]
    pub(crate) edb_native_total_cache:
        FxHashMap<String, (usize, u64, Box<storage::JitRelData>)>,
}

// Safety: JitCompiler is only accessed from one thread at a time (guarded by Mutex).
#[cfg(feature = "jit-asm")]
unsafe impl Send for JitCompiler {}
#[cfg(feature = "jit-asm")]
unsafe impl Sync for JitCompiler {}

#[cfg(feature = "jit-asm")]
impl JitCompiler {
    /// Create a new JIT compiler.
    pub fn new() -> Result<Self, String> {
        Ok(JitCompiler {
            stratum_stage4_fn_cache: FxHashMap::default(),
            stratum_stage4_native_fn_cache: FxHashMap::default(),
            asm_buffers: Vec::new(),
            var_count: 0,
            dedup_cap_hints: FxHashMap::default(),
            tuple_count_hints: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            stratum_runtime_pool: FxHashMap::default(),
            #[cfg(all(feature = "jit-asm", feature = "specialized"))]
            edb_native_total_cache: FxHashMap::default(),
        })
    }

    /// Eligibility check for Stage 4 asm path.
    ///
    /// Accepts `!rel(var...)` negation clauses (arity ≤ 3, all args must be plain variables).
    fn packed_eligible_reason_stage4(rule: &CRule) -> Result<(), &'static str> {
        Self::packed_eligible_reason_inner(rule, true)
    }

    fn packed_eligible_reason_inner(rule: &CRule, allow_not: bool) -> Result<(), &'static str> {
        let clause_count = rule
            .body
            .iter()
            .filter(|item| matches!(item, CBodyItem::Clause(_)))
            .count();
        if clause_count == 0 {
            if !allow_not {
                return Err("no clause body items");
            }
            // Pure-aggregation rule: allow if all non-condition body items are count/sum/min/max.
            let has_valid_agg = rule.body.iter().any(|item| {
                matches!(item, CBodyItem::Aggregation(a) if matches!(a.aggregator_name.as_str(), "count" | "sum" | "min" | "max"))
            });
            if !has_valid_agg {
                return Err("no clause body items and no supported aggregation");
            }
            for item in &rule.body {
                match item {
                    CBodyItem::Aggregation(a) => {
                        match a.aggregator_name.as_str() {
                            "count" => {
                                if a.result_vars.len() != 1 {
                                    return Err("count aggregation must have exactly 1 result var");
                                }
                            }
                            "sum" | "min" | "max" => {
                                if a.result_vars.len() != 1 {
                                    return Err("sum/min/max must have exactly 1 result var");
                                }
                                if a.bound_vars.len() != 1 {
                                    return Err("sum/min/max must have exactly 1 bound var");
                                }
                                for arg in &a.args {
                                    if !matches!(arg, CAggArg::Var(_)) {
                                        return Err("aggregation args must be plain variables");
                                    }
                                }
                            }
                            _ => return Err("unsupported aggregator in asm backend"),
                        }
                    }
                    CBodyItem::Condition(CCondition::If(_)) => {}
                    _ => return Err("unsupported body item in pure-aggregation rule"),
                }
            }
            for head in &rule.heads {
                for arg in &head.args {
                    if !is_supported_packed_expr(arg) {
                        return Err("head arg uses unsupported expression");
                    }
                }
            }
            return Ok(());
        }
        if clause_count > 4 {
            return Err("more than 4 clause body items");
        }
        for item in &rule.body {
            match item {
                CBodyItem::Clause(clause) => {
                    for arg in &clause.args {
                        match arg {
                            CClauseArg::Var(_) => {}
                            CClauseArg::Expr(expr) => {
                                if !is_supported_packed_expr(expr) {
                                    return Err("unsupported clause arg expression");
                                }
                            }
                        }
                    }
                    for cond in &clause.conditions {
                        match cond {
                            CCondition::If(expr) => {
                                if !is_supported_packed_expr(expr) {
                                    return Err("unsupported clause condition expression");
                                }
                            }
                            _ => return Err("clause has IfLet/Let condition (not supported)"),
                        }
                    }
                }
                CBodyItem::Condition(CCondition::If(expr)) => {
                    if !is_supported_packed_expr(expr) {
                        return Err("unsupported condition expression");
                    }
                }
                CBodyItem::Aggregation(a) if allow_not && a.aggregator_name == "not" => {
                    if a.args.len() > 3 {
                        return Err("not-clause arity > 3 not supported in asm backend");
                    }
                    for arg in &a.args {
                        if !matches!(arg, CAggArg::Var(_)) {
                            return Err("not-clause has non-variable argument");
                        }
                    }
                }
                _ => return Err("unsupported body item (aggregation or other)"),
            }
        }
        for head in &rule.heads {
            for arg in &head.args {
                if !is_supported_packed_expr(arg) {
                    return Err("head arg uses unsupported expression");
                }
            }
        }
        Ok(())
    }

    /// Compile or retrieve a Stage 4 stratum function for the given rules (non-native asm).
    ///
    /// Returns `None` if any rule is not packed-JIT eligible.
    pub fn compile_stratum_stage4(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Option<packed_helpers::StratumStage4Fn> {
        if let Some(cached) = self.stratum_stage4_fn_cache.get(&stratum_key) {
            return *cached;
        }

        for (i, rule) in rules.iter().enumerate() {
            if let Err(reason) = Self::packed_eligible_reason_stage4(rule) {
                if std::env::var("ASCENT_DUMP_JIT").is_ok() {
                    eprintln!("JIT: stratum {stratum_key} rule {i} not eligible for stage4: {reason}");
                }
                self.stratum_stage4_fn_cache.insert(stratum_key, None);
                return None;
            }
        }

        #[allow(clippy::type_complexity)]
        let rule_data: Vec<(Vec<crate::compiled::CClause>, Vec<crate::compiled::CHeadClause>, Vec<crate::compiled::CExpr>, Vec<CAggregation>, Vec<CAggregation>)> = rules
            .iter()
            .map(|rule| {
                let clauses: Vec<crate::compiled::CClause> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Clause(c) => Some(c.clone()),
                        _ => None,
                    })
                    .collect();
                let conditions: Vec<crate::compiled::CExpr> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Condition(CCondition::If(expr)) => Some(expr.clone()),
                        _ => None,
                    })
                    .collect();
                let not_clauses: Vec<CAggregation> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Aggregation(a) if a.aggregator_name == "not" => {
                            Some(a.clone())
                        }
                        _ => None,
                    })
                    .collect();
                let agg_clauses: Vec<CAggregation> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Aggregation(a) if a.aggregator_name != "not" => {
                            Some(a.clone())
                        }
                        _ => None,
                    })
                    .collect();
                (clauses, rule.heads.clone(), conditions, not_clauses, agg_clauses)
            })
            .collect();
        let rules_refs: Vec<asm_codegen::AsmRuleRef<'_>> = rule_data
            .iter()
            .map(|(c, h, conds, nots, aggs)| (c.as_slice(), h.as_slice(), conds.as_slice(), nots.as_slice(), aggs.as_slice()))
            .collect();

        match asm_codegen::codegen_stratum_asm(
            &rules_refs,
            self.var_count,
            packed_helpers::jit_stratum_advance_s4 as usize,
            packed_helpers::packed_try_insert as usize,
            packed_helpers::packed_count as usize,
            packed_helpers::packed_data_ptr as usize,
            packed_helpers::packed_recent_ptr as usize,
        ) {
            Ok(asm_stratum) => {
                let fn_ptr = asm_stratum.fn_ptr;
                self.asm_buffers.push(asm_stratum);
                self.stratum_stage4_fn_cache.insert(stratum_key, Some(fn_ptr));
                Some(fn_ptr)
            }
            Err(reason) => {
                if std::env::var("ASCENT_DUMP_JIT").is_ok() {
                    eprintln!("asm backend skipped stratum {stratum_key}: {reason}");
                }
                self.stratum_stage4_fn_cache.insert(stratum_key, None);
                None
            }
        }
    }

    /// Compile or retrieve the native Stage 4 stratum function (direct JitRelData reads).
    ///
    /// Returns `None` if the asm backend returns an error or the stratum has negation/aggregation.
    pub fn compile_stratum_stage4_native(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
        head_is_sink: &[bool],
    ) -> Option<packed_helpers::StratumStage4Fn> {
        if let Some(cached) = self.stratum_stage4_native_fn_cache.get(&stratum_key) {
            return *cached;
        }

        // Native path does not support negation or aggregation.
        let has_negation_or_agg = rules.iter().any(|rule| {
            rule.body.iter().any(|item| matches!(item, CBodyItem::Aggregation(_)))
        });
        if has_negation_or_agg {
            self.stratum_stage4_native_fn_cache.insert(stratum_key, None);
            return None;
        }

        let rule_data: Vec<(Vec<crate::compiled::CClause>, Vec<crate::compiled::CHeadClause>, Vec<crate::compiled::CExpr>)> = rules
            .iter()
            .map(|rule| {
                let clauses: Vec<crate::compiled::CClause> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Clause(c) => Some(c.clone()),
                        _ => None,
                    })
                    .collect();
                let conditions: Vec<crate::compiled::CExpr> = rule
                    .body
                    .iter()
                    .filter_map(|item| match item {
                        CBodyItem::Condition(CCondition::If(expr)) => Some(expr.clone()),
                        _ => None,
                    })
                    .collect();
                (clauses, rule.heads.clone(), conditions)
            })
            .collect();
        let rules_refs: Vec<(&[crate::compiled::CClause], &[crate::compiled::CHeadClause], &[crate::compiled::CExpr])> = rule_data
            .iter()
            .map(|(c, h, conds)| (c.as_slice(), h.as_slice(), conds.as_slice()))
            .collect();

        match asm_codegen::codegen_stratum_asm_native(
            &rules_refs,
            self.var_count,
            packed_helpers::jit_advance_native as usize,
            head_is_sink,
        ) {
            Ok(asm_stratum) => {
                let fn_ptr = asm_stratum.fn_ptr;
                self.asm_buffers.push(asm_stratum);
                self.stratum_stage4_native_fn_cache.insert(stratum_key, Some(fn_ptr));
                Some(fn_ptr)
            }
            Err(reason) => {
                if std::env::var("ASCENT_DUMP_JIT").is_ok() {
                    eprintln!("asm native backend skipped stratum {stratum_key}: {reason}");
                }
                self.stratum_stage4_native_fn_cache.insert(stratum_key, None);
                None
            }
        }
    }
}

/// Check whether a CExpr can be compiled by the packed JIT condition emitter.
#[cfg(feature = "specialized")]
fn is_supported_packed_expr(expr: &CExpr) -> bool {
    match expr {
        CExpr::Var(_) | CExpr::DerefVar(_) => true,
        CExpr::Literal(crate::value::Value::I32(_)) | CExpr::Literal(crate::value::Value::Bool(_)) => true,
        CExpr::VarBinVar(op, _, _) => is_supported_packed_binop(*op),
        CExpr::VarBinLit(op, _, crate::value::Value::I32(_))
        | CExpr::VarBinLit(op, _, crate::value::Value::Bool(_)) => is_supported_packed_binop(*op),
        CExpr::LitBinVar(op, crate::value::Value::I32(_), _)
        | CExpr::LitBinVar(op, crate::value::Value::Bool(_), _) => is_supported_packed_binop(*op),
        CExpr::Binary(op, a, b) => {
            is_supported_packed_binop(*op)
                && is_supported_packed_expr(a)
                && is_supported_packed_expr(b)
        }
        CExpr::Unary(op, inner) => {
            matches!(op, CUnOp::Not | CUnOp::Neg | CUnOp::Deref) && is_supported_packed_expr(inner)
        }
        _ => false,
    }
}

#[cfg(feature = "specialized")]
fn is_supported_packed_binop(op: CBinOp) -> bool {
    matches!(
        op,
        CBinOp::Eq
            | CBinOp::Ne
            | CBinOp::Lt
            | CBinOp::Le
            | CBinOp::Gt
            | CBinOp::Ge
            | CBinOp::Add
            | CBinOp::Sub
            | CBinOp::Mul
            | CBinOp::Div
            | CBinOp::Rem
            | CBinOp::BitAnd
            | CBinOp::BitOr
            | CBinOp::BitXor
            | CBinOp::Shl
            | CBinOp::Shr
            | CBinOp::And
            | CBinOp::Or
    )
}
