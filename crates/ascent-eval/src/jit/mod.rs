//! Cranelift JIT compiler for rule bodies.
//!
//! Compiles eligible rules into native code, replacing the interpreted
//! `process_body_recursive` path. Rules with only Clause and simple
//! Condition (If) body items are eligible; others fall back to interpretation.

mod codegen;
pub(crate) mod helpers;
pub(crate) mod layout;
pub mod storage;
#[cfg(feature = "specialized")]
mod packed_codegen;
#[cfg(feature = "specialized")]
pub(crate) mod packed_helpers;
#[cfg(feature = "specialized")]
mod stratum_codegen;
#[cfg(all(feature = "specialized", feature = "jit-asm"))]
mod asm_codegen;
#[cfg(test)]
mod tests;

use cranelift_codegen::ir::types::{I8, I32};
use cranelift_codegen::ir::{AbiParam, Signature};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use rustc_hash::FxHashMap;

#[cfg(feature = "specialized")]
use crate::compiled::{CBinOp, CClauseArg, CExpr, CUnOp};
use crate::compiled::{CAggArg, CAggregation, CBodyItem, CCondition, CRule};

pub(crate) use self::helpers::JitContext;

/// Type alias for JIT-compiled function pointer.
type JitFn = unsafe extern "C" fn(*mut JitContext);

/// A compiled rule with semi-naive variants.
pub struct JitCompiledRule {
    /// `variants[0]` = no-recent (initial iteration).
    /// `variants[i+1]` = `recent_clause_idx = i` (semi-naive variant for clause i).
    variants: Vec<Option<JitFn>>,
}

/// IDs of helper functions declared in the JIT module.
pub(crate) struct JitHelperIds {
    rel_lookup: FuncId,
    rel_get_tuple: FuncId,
    rel_count: FuncId,
    rel_tuple_at: FuncId,
    rel_contains: FuncId,
    value_clone: FuncId,
    value_eq: FuncId,
    slot_clear: FuncId,
    slot_set: FuncId,
    slot_get: FuncId,
    eval_condition: FuncId,
    emit_all_heads: FuncId,
    drop_value: FuncId,
}

/// IDs of packed helper functions declared in the JIT module.
#[cfg(feature = "specialized")]
pub(crate) struct PackedJitHelperIds {
    pub(crate) packed_count: FuncId,
    pub(crate) packed_data_ptr: FuncId,
    pub(crate) packed_recent_idx: FuncId,
    pub(crate) packed_recent_ptr: FuncId,
    pub(crate) packed_lookup: FuncId,
    pub(crate) packed_push_result: FuncId,
    pub(crate) stratum_flush_advance: FuncId,
    // Stage 3 helpers
    #[allow(dead_code)]
    pub(crate) packed_try_insert: FuncId,
    pub(crate) stratum_advance: FuncId,
    // Stage 4: advance + handle refresh
    pub(crate) stratum_advance_s4: FuncId,
}

/// A packed-JIT compiled rule with semi-naive variants.
#[cfg(feature = "specialized")]
pub struct PackedJitCompiledRule {
    /// `variants[0]` = no-recent. `variants[i+1]` = recent for clause i.
    variants: Vec<Option<packed_helpers::PackedJitFn>>,
}

/// A Stage 3 packed-JIT compiled rule: direct-insert variants.
#[cfg(feature = "specialized")]
pub struct PackedJitCompiledRuleV3 {
    /// `variants[0]` = no-recent. `variants[i+1]` = recent for clause i.
    variants: Vec<Option<packed_helpers::PackedJitFnV3>>,
}

#[cfg(feature = "specialized")]
impl PackedJitCompiledRuleV3 {
    pub fn full_variant(&self) -> Option<packed_helpers::PackedJitFnV3> {
        self.variants.first().copied().flatten()
    }

    pub fn recent_variant(&self, clause_seq_idx: usize) -> Option<packed_helpers::PackedJitFnV3> {
        self.variants.get(clause_seq_idx + 1).copied().flatten()
    }
}

#[cfg(feature = "specialized")]
impl PackedJitCompiledRule {
    pub fn full_variant(&self) -> Option<packed_helpers::PackedJitFn> {
        self.variants.first().copied().flatten()
    }

    pub fn recent_variant(&self, clause_seq_idx: usize) -> Option<packed_helpers::PackedJitFn> {
        self.variants.get(clause_seq_idx + 1).copied().flatten()
    }
}

/// Drop a Value in place.
///
/// # Safety
/// `ptr` must point to a valid, initialized Value.
#[unsafe(no_mangle)]
unsafe extern "C" fn jit_drop_value(ptr: *mut crate::value::Value) {
    unsafe { std::ptr::drop_in_place(ptr) };
}

impl std::fmt::Debug for JitCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompiler")
            .field("cached_rules", &self.cache.len())
            .finish()
    }
}

/// The JIT compiler.
pub struct JitCompiler {
    module: JITModule,
    builder_ctx: FunctionBuilderContext,
    codegen_ctx: cranelift_codegen::Context,
    helpers: JitHelperIds,
    #[cfg(feature = "specialized")]
    packed_helpers: PackedJitHelperIds,
    /// Cache of trampoline-compiled rules: rule_idx -> compiled (None = not eligible).
    pub(crate) cache: FxHashMap<usize, Option<JitCompiledRule>>,
    /// Cache of packed-compiled rules: rule_idx -> compiled (None = not eligible).
    #[cfg(feature = "specialized")]
    pub(crate) packed_cache: FxHashMap<usize, Option<PackedJitCompiledRule>>,
    /// Cache of stratum meta-functions: stratum_key -> fn_ptr (None = not eligible).
    #[cfg(feature = "specialized")]
    pub(crate) stratum_fn_cache: FxHashMap<usize, Option<packed_helpers::StratumMetaFn>>,
    /// Cache of Stage 3 packed compiled rules.
    #[cfg(feature = "specialized")]
    pub(crate) packed_cache_v3: FxHashMap<usize, Option<PackedJitCompiledRuleV3>>,
    /// Cache of Stage 3 stratum functions.
    #[cfg(feature = "specialized")]
    pub(crate) stratum_stage3_fn_cache: FxHashMap<usize, Option<packed_helpers::StratumStage3Fn>>,
    /// Cache of Stage 4 stratum functions (inlined rule bodies).
    #[cfg(feature = "specialized")]
    pub(crate) stratum_stage4_fn_cache: FxHashMap<usize, Option<packed_helpers::StratumStage4Fn>>,
    /// Cache of Stage 4 native stratum functions (JitRelData direct read path).
    /// Keyed by stratum_key. None = not eligible (or not attempted yet).
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
    pub(crate) stratum_stage4_native_fn_cache:
        FxHashMap<usize, Option<packed_helpers::StratumStage4Fn>>,
    /// Whether the stage4 fn for a given stratum was compiled by the non-native asm backend
    /// (true) or Cranelift (false).  Used to skip unnecessary `jit_used_in_cranelift_strata`
    /// marking: non-native asm uses packed_data_ptr callbacks and doesn't need JitHashIndex.
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
    pub(crate) stratum_stage4_fn_is_asm: FxHashMap<usize, bool>,
    /// Asm-backend stratum buffers: kept alive so the fn_ptr remains valid.
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
    pub(crate) asm_buffers: Vec<asm_codegen::AsmStratum>,
    /// Total number of interned variables (set by Engine before each compilation batch).
    /// Used to declare the right number of Cranelift Variables in JIT-compiled functions.
    #[cfg(feature = "specialized")]
    pub(crate) var_count: usize,
    /// Peak dedup-table capacity observed per relation (relation name → cap).
    ///
    /// Updated after each stratum run completes; used to pre-size dedup tables
    /// on the next engine's first stratum execution, avoiding repeated reallocs.
    #[cfg(feature = "specialized")]
    pub(crate) dedup_cap_hints: rustc_hash::FxHashMap<String, u32>,
    /// Peak tuple count observed per IDB head relation (relation name → count).
    ///
    /// Used to pre-size `packed_data` (capacity = count × arity) and `delta`
    /// (capacity = count) before the stratum runs, eliminating Vec growth reallocs.
    #[cfg(feature = "specialized")]
    pub(crate) tuple_count_hints: rustc_hash::FxHashMap<String, u32>,
}

// Safety: JitCompiler is only accessed from one thread at a time (guarded by Mutex).
// JITModule contains raw pointers but they are stable heap allocations owned by the compiler.
unsafe impl Send for JitCompiler {}
unsafe impl Sync for JitCompiler {}

impl JitCompiler {
    /// Create a new JIT compiler.
    pub fn new() -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| format!("set opt_level: {e}"))?;
        let isa_builder = cranelift_codegen::isa::lookup(::target_lexicon::Triple::host())
            .map_err(|e| format!("ISA lookup: {e}"))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("ISA finish: {e}"))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register helper function symbols
        jit_builder.symbol("jit_rel_lookup", helpers::jit_rel_lookup as *const u8);
        jit_builder.symbol("jit_rel_get_tuple", helpers::jit_rel_get_tuple as *const u8);
        jit_builder.symbol("jit_rel_count", helpers::jit_rel_count as *const u8);
        jit_builder.symbol("jit_rel_tuple_at", helpers::jit_rel_tuple_at as *const u8);
        jit_builder.symbol("jit_rel_contains", helpers::jit_rel_contains as *const u8);
        jit_builder.symbol("jit_value_clone", helpers::jit_value_clone as *const u8);
        jit_builder.symbol("jit_value_eq", helpers::jit_value_eq as *const u8);
        jit_builder.symbol("jit_slot_clear", helpers::jit_slot_clear as *const u8);
        jit_builder.symbol("jit_slot_set", helpers::jit_slot_set as *const u8);
        jit_builder.symbol("jit_slot_get", helpers::jit_slot_get as *const u8);
        jit_builder.symbol(
            "jit_eval_condition",
            helpers::jit_eval_condition as *const u8,
        );
        jit_builder.symbol(
            "jit_emit_all_heads",
            helpers::jit_emit_all_heads as *const u8,
        );
        jit_builder.symbol("jit_drop_value", jit_drop_value as *const u8);

        // Packed JIT helpers (only when specialized storage is available)
        #[cfg(feature = "specialized")]
        {
            jit_builder.symbol("packed_count", packed_helpers::packed_count as *const u8);
            jit_builder.symbol(
                "packed_data_ptr",
                packed_helpers::packed_data_ptr as *const u8,
            );
            jit_builder.symbol(
                "packed_recent_idx",
                packed_helpers::packed_recent_idx as *const u8,
            );
            jit_builder.symbol(
                "packed_recent_ptr",
                packed_helpers::packed_recent_ptr as *const u8,
            );
            jit_builder.symbol("packed_lookup", packed_helpers::packed_lookup as *const u8);
            jit_builder.symbol(
                "packed_push_result",
                packed_helpers::packed_push_result as *const u8,
            );
            jit_builder.symbol(
                "jit_stratum_flush_advance",
                packed_helpers::jit_stratum_flush_advance as *const u8,
            );
            jit_builder.symbol(
                "packed_try_insert",
                packed_helpers::packed_try_insert as *const u8,
            );
            jit_builder.symbol(
                "jit_stratum_advance",
                packed_helpers::jit_stratum_advance as *const u8,
            );
            jit_builder.symbol(
                "jit_stratum_advance_s4",
                packed_helpers::jit_stratum_advance_s4 as *const u8,
            );
        }

        let mut module = JITModule::new(jit_builder);
        let helpers = declare_helpers(&mut module)?;
        #[cfg(feature = "specialized")]
        let packed_helpers = declare_packed_helpers(&mut module)?;

        Ok(JitCompiler {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            codegen_ctx: cranelift_codegen::Context::new(),
            helpers,
            #[cfg(feature = "specialized")]
            packed_helpers,
            cache: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            packed_cache: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            stratum_fn_cache: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            packed_cache_v3: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            stratum_stage3_fn_cache: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            stratum_stage4_fn_cache: FxHashMap::default(),
            #[cfg(all(feature = "specialized", feature = "jit-asm"))]
            stratum_stage4_native_fn_cache: FxHashMap::default(),
            #[cfg(all(feature = "specialized", feature = "jit-asm"))]
            stratum_stage4_fn_is_asm: FxHashMap::default(),
            #[cfg(all(feature = "specialized", feature = "jit-asm"))]
            asm_buffers: Vec::new(),
            #[cfg(feature = "specialized")]
            var_count: 0,
            #[cfg(feature = "specialized")]
            dedup_cap_hints: FxHashMap::default(),
            #[cfg(feature = "specialized")]
            tuple_count_hints: FxHashMap::default(),
        })
    }

    /// Check if a rule is eligible for the typed packed JIT.
    ///
    /// Accepts Clause and Condition(If(expr)) body items where expr uses only
    /// supported packed ops. Literal I32/Bool clause args are also accepted.
    ///
    /// Returns `Ok(())` if eligible, or `Err(reason)` explaining why not.
    #[cfg(feature = "specialized")]
    /// Eligibility check for packed JIT (Stages 2–3, Cranelift path).
    ///
    /// Negation (`!rel(...)`) is NOT accepted here because the Cranelift
    /// codegen does not implement anti-join probes.  Use
    /// `packed_eligible_reason_stage4` for the asm Stage 4 path.
    pub fn packed_eligible_reason(rule: &CRule) -> Result<(), &'static str> {
        Self::packed_eligible_reason_inner(rule, false)
    }

    /// Eligibility check for Stage 4 asm path.
    ///
    /// Extends the base check to accept `!rel(var...)` negation clauses (arity ≤ 3,
    /// all args must be plain variables), which the asm backend handles via
    /// `check_not_packed_N` probes.
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
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
            return Err("no clause body items");
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

    /// Get or compile the packed JIT variant. Returns None if not eligible.
    #[cfg(feature = "specialized")]
    pub fn get_or_compile_packed(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
    ) -> Option<&PackedJitCompiledRule> {
        if self.packed_cache.contains_key(&rule_idx) {
            return self.packed_cache[&rule_idx].as_ref();
        }
        if let Err(reason) = Self::packed_eligible_reason(rule) {
            eprintln!("JIT: rule {rule_idx} not eligible for packed JIT: {reason}");
            self.packed_cache.insert(rule_idx, None);
            return None;
        }
        match self.compile_packed_rule(rule_idx, rule) {
            Ok(compiled) => {
                self.packed_cache.insert(rule_idx, Some(compiled));
                self.packed_cache[&rule_idx].as_ref()
            }
            Err(e) => panic!("JIT: eligible rule {rule_idx} failed to compile: {e}"),
        }
    }

    #[cfg(feature = "specialized")]
    fn compile_packed_rule(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
    ) -> Result<PackedJitCompiledRule, String> {
        let clause_count = rule
            .body
            .iter()
            .filter(|item| matches!(item, CBodyItem::Clause(_)))
            .count();

        let mut variants = Vec::with_capacity(clause_count + 1);

        let fn_ptr = self.compile_packed_variant(rule_idx, rule, None)?;
        variants.push(Some(fn_ptr));

        for (body_idx, item) in rule.body.iter().enumerate() {
            if matches!(item, CBodyItem::Clause(_)) {
                let fn_ptr = self.compile_packed_variant(rule_idx, rule, Some(body_idx))?;
                variants.push(Some(fn_ptr));
            }
        }

        Ok(PackedJitCompiledRule { variants })
    }

    #[cfg(feature = "specialized")]
    fn compile_packed_variant(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
        recent_clause_idx: Option<usize>,
    ) -> Result<packed_helpers::PackedJitFn, String> {
        let suffix = match recent_clause_idx {
            None => "packed_full".to_string(),
            Some(idx) => format!("packed_recent_{idx}"),
        };
        let name = format!("rule_{rule_idx}_{suffix}");

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let ptr_type = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare_function: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        packed_codegen::codegen_packed_rule_body(
            rule,
            recent_clause_idx,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
            &self.packed_helpers,
        )?;

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: packed_helpers::PackedJitFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(fn_ptr)
    }

    /// Get or compile the Stage 3 (direct-insert) packed JIT variant.
    #[cfg(feature = "specialized")]
    pub fn get_or_compile_packed_v3(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
    ) -> Option<&PackedJitCompiledRuleV3> {
        if self.packed_cache_v3.contains_key(&rule_idx) {
            return self.packed_cache_v3[&rule_idx].as_ref();
        }
        if let Err(reason) = Self::packed_eligible_reason(rule) {
            eprintln!("JIT: rule {rule_idx} not eligible for packed JIT v3: {reason}");
            self.packed_cache_v3.insert(rule_idx, None);
            return None;
        }
        match self.compile_packed_rule_v3(rule_idx, rule) {
            Ok(compiled) => {
                self.packed_cache_v3.insert(rule_idx, Some(compiled));
                self.packed_cache_v3[&rule_idx].as_ref()
            }
            Err(e) => panic!("JIT: eligible rule {rule_idx} failed to compile (v3): {e}"),
        }
    }

    #[cfg(feature = "specialized")]
    fn compile_packed_rule_v3(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
    ) -> Result<PackedJitCompiledRuleV3, String> {
        let clause_count = rule
            .body
            .iter()
            .filter(|item| matches!(item, CBodyItem::Clause(_)))
            .count();

        let mut variants = Vec::with_capacity(clause_count + 1);

        let fn_ptr = self.compile_packed_variant_v3(rule_idx, rule, None)?;
        variants.push(Some(fn_ptr));

        for (body_idx, item) in rule.body.iter().enumerate() {
            if matches!(item, CBodyItem::Clause(_)) {
                let fn_ptr = self.compile_packed_variant_v3(rule_idx, rule, Some(body_idx))?;
                variants.push(Some(fn_ptr));
            }
        }

        Ok(PackedJitCompiledRuleV3 { variants })
    }

    #[cfg(feature = "specialized")]
    fn compile_packed_variant_v3(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
        recent_clause_idx: Option<usize>,
    ) -> Result<packed_helpers::PackedJitFnV3, String> {
        let suffix = match recent_clause_idx {
            None => "v3_full".to_string(),
            Some(idx) => format!("v3_recent_{idx}"),
        };
        let name = format!("rule_{rule_idx}_{suffix}");

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let ptr_type = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare_function v3: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        packed_codegen::codegen_packed_rule_body_v3(
            rule,
            recent_clause_idx,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
            &self.packed_helpers,
            self.var_count,
        )?;

        if std::env::var("ASCENT_DUMP_JIT").is_ok() {
            eprintln!("=== V3 CLIF IR ({name}) ===\n{}", self.codegen_ctx.func.display());
        }

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize v3: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: packed_helpers::PackedJitFnV3 = unsafe { std::mem::transmute(code_ptr) };
        Ok(fn_ptr)
    }

    /// Compile or retrieve a Stage 3 stratum function for the given rules.
    ///
    /// Returns `None` if any rule is not packed-JIT eligible.
    #[cfg(feature = "specialized")]
    pub fn compile_stratum_stage3(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Option<packed_helpers::StratumStage3Fn> {
        if let Some(cached) = self.stratum_stage3_fn_cache.get(&stratum_key) {
            return *cached;
        }

        // All rules must have V3 compiled variants
        for rule in rules {
            let rule_idx = *rule as *const CRule as usize;
            if !self.packed_cache_v3.get(&rule_idx).is_some_and(|v| v.is_some()) {
                self.stratum_stage3_fn_cache.insert(stratum_key, None);
                return None;
            }
        }

        match self.compile_stratum_stage3_inner(stratum_key) {
            Ok(fn_ptr) => {
                self.stratum_stage3_fn_cache.insert(stratum_key, Some(fn_ptr));
                Some(fn_ptr)
            }
            Err(e) => panic!("JIT: stratum {stratum_key} failed stage3 compile: {e}"),
        }
    }

    #[cfg(feature = "specialized")]
    fn compile_stratum_stage3_inner(
        &mut self,
        stratum_key: usize,
    ) -> Result<packed_helpers::StratumStage3Fn, String> {
        let name = format!("stratum_stage3_{stratum_key}");
        let ptr_type = self.module.target_config().pointer_type();

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare stratum_stage3: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        stratum_codegen::codegen_stratum_stage3_fn(
            self.packed_helpers.stratum_advance,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
        )?;

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize stratum_stage3: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: packed_helpers::StratumStage3Fn = unsafe { std::mem::transmute(code_ptr) };
        Ok(fn_ptr)
    }

    /// Compile or retrieve a Stage 4 stratum function for the given rules.
    ///
    /// Stage 4 inlines all rule bodies directly into one Cranelift function,
    /// eliminating `call_indirect` overhead.
    ///
    /// Returns `None` if any rule is not packed-JIT eligible.
    #[cfg(feature = "specialized")]
    pub fn compile_stratum_stage4(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Option<packed_helpers::StratumStage4Fn> {
        if let Some(cached) = self.stratum_stage4_fn_cache.get(&stratum_key) {
            return *cached;
        }

        // All rules must be packed-JIT eligible (Stage 4 asm allows negation).
        #[cfg(feature = "jit-asm")]
        for (i, rule) in rules.iter().enumerate() {
            if let Err(reason) = Self::packed_eligible_reason_stage4(rule) {
                eprintln!("JIT: stratum {stratum_key} rule {i} not eligible for stage4: {reason}");
                self.stratum_stage4_fn_cache.insert(stratum_key, None);
                return None;
            }
        }
        #[cfg(not(feature = "jit-asm"))]
        for (i, rule) in rules.iter().enumerate() {
            if let Err(reason) = Self::packed_eligible_reason(rule) {
                eprintln!("JIT: stratum {stratum_key} rule {i} not eligible for stage4: {reason}");
                self.stratum_stage4_fn_cache.insert(stratum_key, None);
                return None;
            }
        }

        match self.compile_stratum_stage4_inner(stratum_key, rules) {
            Ok(fn_ptr) => {
                self.stratum_stage4_fn_cache.insert(stratum_key, Some(fn_ptr));
                Some(fn_ptr)
            }
            Err(e) => panic!("JIT: eligible stratum {stratum_key} failed stage4 compile: {e}"),
        }
    }

    /// Returns `true` if the stage4 fn for this stratum was compiled by the non-native asm
    /// backend, `false` if by Cranelift.  Panics if the stratum has not been compiled yet.
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
    pub(crate) fn stratum_stage4_fn_is_asm(&self, stratum_key: usize) -> bool {
        *self.stratum_stage4_fn_is_asm.get(&stratum_key)
            .expect("stratum_stage4_fn_is_asm queried before compilation")
    }

    #[cfg(feature = "specialized")]
    fn compile_stratum_stage4_inner(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Result<packed_helpers::StratumStage4Fn, String> {
        // Try the lightweight asm backend first; fall back to Cranelift on Err.
        #[cfg(feature = "jit-asm")]
        {
            #[allow(clippy::type_complexity)]
            let rule_data: Vec<(Vec<crate::compiled::CClause>, Vec<crate::compiled::CHeadClause>, Vec<crate::compiled::CExpr>, Vec<CAggregation>)> = rules
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
                    (clauses, rule.heads.clone(), conditions, not_clauses)
                })
                .collect();
            let rules_refs: Vec<asm_codegen::AsmRuleRef<'_>> = rule_data
                .iter()
                .map(|(c, h, conds, nots)| (c.as_slice(), h.as_slice(), conds.as_slice(), nots.as_slice()))
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
                    self.stratum_stage4_fn_is_asm.insert(stratum_key, true);
                    return Ok(fn_ptr);
                }
                Err(reason) => {
                    if std::env::var("ASCENT_DUMP_JIT").is_ok() {
                        eprintln!("asm backend skipped stratum {stratum_key}: {reason}");
                    }
                }
            }
        }

        let name = format!("stratum_stage4_{stratum_key}");
        let ptr_type = self.module.target_config().pointer_type();

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare stratum_stage4: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        // Pre-extract clauses, heads, and conditions per rule.
        // We need owned data since we can't hold borrows into `rules` across the
        // mutable `module` calls.
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
            .map(|(clauses, heads, conditions)| {
                (clauses.as_slice(), heads.as_slice(), conditions.as_slice())
            })
            .collect();

        stratum_codegen::codegen_stratum_stage4_fn(
            self.packed_helpers.stratum_advance_s4,
            &rules_refs,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
            &self.packed_helpers,
            self.var_count,
        )?;

        if std::env::var("ASCENT_DUMP_JIT").is_ok() {
            eprintln!("=== Stage4 CLIF IR (stratum {stratum_key}) ===\n{}", self.codegen_ctx.func.display());
        }

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize stratum_stage4: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: packed_helpers::StratumStage4Fn = unsafe { std::mem::transmute(code_ptr) };
        #[cfg(feature = "jit-asm")]
        self.stratum_stage4_fn_is_asm.insert(stratum_key, false);
        Ok(fn_ptr)
    }

    /// Compile or retrieve the native Stage 4 stratum function (Step 4).
    ///
    /// The native function reads scan data directly from `JitRelData` fields,
    /// eliminating all Rust callbacks from the inner loop READ side.
    ///
    /// Returns `None` if the asm backend is unavailable or returns an error.
    #[cfg(all(feature = "specialized", feature = "jit-asm"))]
    pub fn compile_stratum_stage4_native(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Option<packed_helpers::StratumStage4Fn> {
        if let Some(cached) = self.stratum_stage4_native_fn_cache.get(&stratum_key) {
            return *cached;
        }

        // Native path does not support negation: skip and use non-native asm instead.
        let has_negation = rules.iter().any(|rule| {
            rule.body.iter().any(|item| matches!(item, CBodyItem::Aggregation(a) if a.aggregator_name == "not"))
        });
        if has_negation {
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

    /// Compile or retrieve a stratum meta-function for the given rules.
    ///
    /// Returns `None` if any rule is not packed-JIT eligible.
    #[cfg(feature = "specialized")]
    pub fn compile_stratum_meta(
        &mut self,
        stratum_key: usize,
        rules: &[&CRule],
    ) -> Option<packed_helpers::StratumMetaFn> {
        if let Some(cached) = self.stratum_fn_cache.get(&stratum_key) {
            return *cached;
        }

        // All rules must be packed-JIT compiled
        for rule in rules {
            let rule_idx = *rule as *const CRule as usize;
            if !self.packed_cache.get(&rule_idx).is_some_and(|v| v.is_some()) {
                self.stratum_fn_cache.insert(stratum_key, None);
                return None;
            }
        }

        match self.compile_stratum_meta_inner(stratum_key) {
            Ok(fn_ptr) => {
                self.stratum_fn_cache.insert(stratum_key, Some(fn_ptr));
                Some(fn_ptr)
            }
            Err(_) => {
                self.stratum_fn_cache.insert(stratum_key, None);
                None
            }
        }
    }

    #[cfg(feature = "specialized")]
    fn compile_stratum_meta_inner(
        &mut self,
        stratum_key: usize,
    ) -> Result<packed_helpers::StratumMetaFn, String> {
        let name = format!("stratum_meta_{stratum_key}");
        let ptr_type = self.module.target_config().pointer_type();

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare stratum_meta: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        stratum_codegen::codegen_stratum_meta_fn(
            self.packed_helpers.stratum_flush_advance,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
        )?;

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize stratum_meta: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: packed_helpers::StratumMetaFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(fn_ptr)
    }

    /// Check if a rule is eligible for JIT compilation.
    pub fn is_eligible(rule: &CRule) -> bool {
        // Max 4 clauses to avoid code size explosion
        let clause_count = rule
            .body
            .iter()
            .filter(|item| matches!(item, CBodyItem::Clause(_)))
            .count();
        if clause_count == 0 || clause_count > 4 {
            return false;
        }

        // Must have at least one body item
        if rule.body.is_empty() {
            return false;
        }

        for item in &rule.body {
            match item {
                CBodyItem::Clause(clause) => {
                    // Check that clause args don't have Expr args (only Var)
                    for arg in &clause.args {
                        if matches!(arg, crate::compiled::CClauseArg::Expr(_)) {
                            return false;
                        }
                    }
                    // Clause-level conditions not yet supported in JIT
                    if !clause.conditions.is_empty() {
                        return false;
                    }
                }
                CBodyItem::Condition(cond) => {
                    // Only simple If conditions
                    if !matches!(cond, CCondition::If(_)) {
                        return false;
                    }
                }
                // Generator and Aggregation are not eligible
                CBodyItem::Generator(_) | CBodyItem::Aggregation(_) => return false,
            }
        }

        true
    }

    /// Get or compile a rule. Returns None if not eligible.
    pub fn get_or_compile(&mut self, rule_idx: usize, rule: &CRule) -> Option<&JitCompiledRule> {
        if self.cache.contains_key(&rule_idx) {
            return self.cache[&rule_idx].as_ref();
        }

        if !Self::is_eligible(rule) {
            self.cache.insert(rule_idx, None);
            return None;
        }

        match self.compile_rule(rule_idx, rule) {
            Ok(compiled) => {
                self.cache.insert(rule_idx, Some(compiled));
                self.cache[&rule_idx].as_ref()
            }
            Err(_e) => {
                // Compilation failed; mark as ineligible
                self.cache.insert(rule_idx, None);
                None
            }
        }
    }

    /// Compile a single rule into all semi-naive variants.
    fn compile_rule(&mut self, rule_idx: usize, rule: &CRule) -> Result<JitCompiledRule, String> {
        let clause_count = rule
            .body
            .iter()
            .filter(|item| matches!(item, CBodyItem::Clause(_)))
            .count();

        // Generate: variant 0 = no recent, variants 1..=N = recent for each clause position
        let mut variants = Vec::with_capacity(clause_count + 1);

        // Variant 0: no recent (initial iteration)
        let fn_ptr = self.compile_variant(rule_idx, rule, None)?;
        variants.push(Some(fn_ptr));

        // Variants 1..=N: each clause body item index as recent
        for (body_idx, item) in rule.body.iter().enumerate() {
            if matches!(item, CBodyItem::Clause(_)) {
                let fn_ptr = self.compile_variant(rule_idx, rule, Some(body_idx))?;
                variants.push(Some(fn_ptr));
            }
        }

        Ok(JitCompiledRule { variants })
    }

    /// Compile one variant (specific recent_clause_idx) of a rule.
    fn compile_variant(
        &mut self,
        rule_idx: usize,
        rule: &CRule,
        recent_clause_idx: Option<usize>,
    ) -> Result<JitFn, String> {
        let variant_suffix = match recent_clause_idx {
            None => "full".to_string(),
            Some(idx) => format!("recent_{idx}"),
        };
        let name = format!("rule_{rule_idx}_{variant_suffix}");

        self.codegen_ctx.clear();
        self.codegen_ctx.func = cranelift_codegen::ir::Function::new();

        let ptr_type = self.module.target_config().pointer_type();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(ptr_type));

        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(|e| format!("declare_function: {e}"))?;

        self.codegen_ctx.func.signature = sig;

        codegen::codegen_rule_body(
            rule,
            recent_clause_idx,
            func_id,
            &mut self.module,
            &mut self.builder_ctx,
            &mut self.codegen_ctx,
            &self.helpers,
        )?;

        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;

        let code_ptr = self.module.get_finalized_function(func_id);
        let fn_ptr: JitFn = unsafe { std::mem::transmute(code_ptr) };
        Ok(fn_ptr)
    }
}

/// Check whether a CExpr can be compiled by the packed JIT condition emitter.
#[cfg(feature = "specialized")]
fn is_supported_packed_expr(expr: &CExpr) -> bool {
    match expr {
        CExpr::Var(_) => true,
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
        CExpr::Unary(CUnOp::Not, inner) | CExpr::Unary(CUnOp::Neg, inner) => {
            is_supported_packed_expr(inner)
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
    )
}

/// Declare all helper function signatures in the JIT module.
fn declare_helpers(module: &mut JITModule) -> Result<JitHelperIds, String> {
    let ptr = module.target_config().pointer_type();
    let cc = module.target_config().default_call_conv;

    let map_err = |e: cranelift_module::ModuleError| format!("declare helper: {e}");

    // jit_rel_lookup(rel: ptr, col: i32, val: ptr, use_recent: i32) -> (ptr, ptr)
    // Returns LookupResult { ptr, len } — we model as two return values
    let mut sig = Signature::new(cc);
    sig.params = vec![
        AbiParam::new(ptr),
        AbiParam::new(I32),
        AbiParam::new(ptr),
        AbiParam::new(I32),
    ];
    sig.returns = vec![AbiParam::new(ptr), AbiParam::new(ptr)]; // (indices_ptr, len as usize)
    let rel_lookup = module
        .declare_function("jit_rel_lookup", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_rel_get_tuple(rel: ptr, tuple_idx: ptr) -> ptr
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr)]; // usize = ptr-sized
    sig.returns = vec![AbiParam::new(ptr)];
    let rel_get_tuple = module
        .declare_function("jit_rel_get_tuple", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_rel_count(rel: ptr, use_recent: i32) -> ptr (usize)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(ptr)];
    let rel_count = module
        .declare_function("jit_rel_count", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_rel_tuple_at(rel: ptr, seq_idx: ptr, use_recent: i32) -> ptr
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(ptr)];
    let rel_tuple_at = module
        .declare_function("jit_rel_tuple_at", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_rel_contains(rel: ptr, tuple: ptr, arity: i32) -> i8 (bool)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(I8)];
    let rel_contains = module
        .declare_function("jit_rel_contains", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_value_clone(src: ptr, dst: ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr)];
    let value_clone = module
        .declare_function("jit_value_clone", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_value_eq(a: ptr, b: ptr) -> i8 (bool)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(I8)];
    let value_eq = module
        .declare_function("jit_value_eq", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_slot_clear(slot: ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    let slot_clear = module
        .declare_function("jit_slot_clear", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_slot_set(slot: ptr, value: ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr)];
    let slot_set = module
        .declare_function("jit_slot_set", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_slot_get(slot: ptr) -> ptr
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(ptr)];
    let slot_get = module
        .declare_function("jit_slot_get", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_eval_condition(cond: ptr, bindings: ptr, registry: ptr, interner: ptr) -> i8
    let mut sig = Signature::new(cc);
    sig.params = vec![
        AbiParam::new(ptr),
        AbiParam::new(ptr),
        AbiParam::new(ptr),
        AbiParam::new(ptr),
    ];
    sig.returns = vec![AbiParam::new(I8)];
    let eval_condition = module
        .declare_function("jit_eval_condition", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_emit_all_heads(ctx: ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    let emit_all_heads = module
        .declare_function("jit_emit_all_heads", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_drop_value(ptr: ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    let drop_value = module
        .declare_function("jit_drop_value", Linkage::Import, &sig)
        .map_err(map_err)?;

    Ok(JitHelperIds {
        rel_lookup,
        rel_get_tuple,
        rel_count,
        rel_tuple_at,
        rel_contains,
        value_clone,
        value_eq,
        slot_clear,
        slot_set,
        slot_get,
        eval_condition,
        emit_all_heads,
        drop_value,
    })
}

/// Declare packed helper function signatures in the JIT module.
#[cfg(feature = "specialized")]
fn declare_packed_helpers(module: &mut JITModule) -> Result<PackedJitHelperIds, String> {
    let ptr = module.target_config().pointer_type();
    let cc = module.target_config().default_call_conv;
    let map_err = |e: cranelift_module::ModuleError| format!("declare packed helper: {e}");

    // packed_count(rel: ptr, use_recent: i32) -> ptr (usize)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(ptr)];
    let packed_count = module
        .declare_function("packed_count", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_data_ptr(rel: ptr) -> ptr (*const u32)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(ptr)];
    let packed_data_ptr = module
        .declare_function("packed_data_ptr", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_recent_idx(rel: ptr, seq_idx: ptr) -> ptr (usize)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(ptr)];
    let packed_recent_idx = module
        .declare_function("packed_recent_idx", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_recent_ptr(rel: ptr) -> ptr (*const usize)
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(ptr)];
    let packed_recent_ptr = module
        .declare_function("packed_recent_ptr", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_lookup(rel: ptr, col: i32, key: i32, use_recent: i32) -> (ptr, ptr)
    let mut sig = Signature::new(cc);
    sig.params = vec![
        AbiParam::new(ptr),
        AbiParam::new(I32),
        AbiParam::new(I32),
        AbiParam::new(I32),
    ];
    sig.returns = vec![AbiParam::new(ptr), AbiParam::new(ptr)];
    let packed_lookup = module
        .declare_function("packed_lookup", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_push_result(results: ptr, head_idx: ptr, tuple: ptr, arity: i32)
    let mut sig = Signature::new(cc);
    sig.params = vec![
        AbiParam::new(ptr),
        AbiParam::new(ptr),
        AbiParam::new(ptr),
        AbiParam::new(I32),
    ];
    let packed_push_result = module
        .declare_function("packed_push_result", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_stratum_flush_advance(flusher: ptr) -> i8
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(I8)];
    let stratum_flush_advance = module
        .declare_function("jit_stratum_flush_advance", Linkage::Import, &sig)
        .map_err(map_err)?;

    // packed_try_insert(rel: ptr, tuple: ptr, arity: i32) -> i8
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(I8)];
    let packed_try_insert = module
        .declare_function("packed_try_insert", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_stratum_advance(rels: ptr, n_rels: i32) -> i8
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr), AbiParam::new(I32)];
    sig.returns = vec![AbiParam::new(I8)];
    let stratum_advance = module
        .declare_function("jit_stratum_advance", Linkage::Import, &sig)
        .map_err(map_err)?;

    // jit_stratum_advance_s4(ctx: ptr) -> i8
    let mut sig = Signature::new(cc);
    sig.params = vec![AbiParam::new(ptr)];
    sig.returns = vec![AbiParam::new(I8)];
    let stratum_advance_s4 = module
        .declare_function("jit_stratum_advance_s4", Linkage::Import, &sig)
        .map_err(map_err)?;

    Ok(PackedJitHelperIds {
        packed_count,
        packed_data_ptr,
        packed_recent_idx,
        packed_recent_ptr,
        packed_lookup,
        packed_push_result,
        stratum_flush_advance,
        packed_try_insert,
        stratum_advance,
        stratum_advance_s4,
    })
}

// ─── Integration with Engine ────────────────────────────────────────

impl JitCompiledRule {
    /// Get the no-recent variant (for initial iteration).
    pub fn full_variant(&self) -> Option<JitFn> {
        self.variants.first().copied().flatten()
    }

    /// Get the variant for a specific recent clause body index.
    /// `clause_body_indices` maps clause sequential index to body index.
    pub fn recent_variant(&self, clause_seq_idx: usize) -> Option<JitFn> {
        self.variants.get(clause_seq_idx + 1).copied().flatten()
    }
}
