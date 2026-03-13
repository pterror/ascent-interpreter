//! Cranelift JIT compiler for rule bodies.
//!
//! Compiles eligible rules into native code, replacing the interpreted
//! `process_body_recursive` path. Rules with only Clause and simple
//! Condition (If) body items are eligible; others fall back to interpretation.

mod codegen;
pub(crate) mod helpers;
pub(crate) mod layout;
#[cfg(test)]
mod tests;

use cranelift_codegen::ir::types::{I8, I32};
use cranelift_codegen::ir::{AbiParam, Signature};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use rustc_hash::FxHashMap;

use crate::compiled::{CBodyItem, CCondition, CRule};

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
    /// Cache of compiled rules: rule_idx -> compiled (None = not eligible).
    pub(crate) cache: FxHashMap<usize, Option<JitCompiledRule>>,
}

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

        let mut module = JITModule::new(jit_builder);
        let helpers = declare_helpers(&mut module)?;

        Ok(JitCompiler {
            module,
            builder_ctx: FunctionBuilderContext::new(),
            codegen_ctx: cranelift_codegen::Context::new(),
            helpers,
            cache: FxHashMap::default(),
        })
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
