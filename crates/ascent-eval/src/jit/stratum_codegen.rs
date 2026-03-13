//! Cranelift codegen for the stratum meta-function.
//!
//! The meta-function owns the `while has_delta` fixpoint loop, calling each
//! packed JIT rule variant in sequence and delegating flush+advance to a
//! Rust helper.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::I32;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use cranelift_codegen::ir::Block;

use crate::compiled::{CClause, CExpr, CHeadClause};
use crate::jit::PackedJitHelperIds;
use crate::jit::packed_codegen::{FuncRefsV3, gen_clauses_v3};


/// Generate the stratum meta-function body.
///
/// The meta-function signature is `fn(*mut StratumMetaCtx)`.
///
/// `StratumMetaCtx` layout (repr C, 64-bit):
/// ```text
///   offset  0: full_fns:    *const PackedJitFn         (ptr_size bytes)
///   offset  8: full_ctxs:   *const *mut PackedJitCtx   (ptr_size bytes)
///   offset 16: num_full:    u32                         (4 bytes)
///   offset 20: num_recent:  u32                         (4 bytes)
///   offset 24: recent_fns:  *const PackedJitFn         (ptr_size bytes)
///   offset 32: recent_ctxs: *const *mut PackedJitCtx   (ptr_size bytes)
///   offset 40: flusher:     *mut StratumFlusher         (ptr_size bytes)
/// ```
///
/// Loop structure (pseudo-code):
/// ```text
/// for i in 0..num_full: full_fns[i](full_ctxs[i])
/// if !flush_advance(flusher): return
/// loop:
///   for i in 0..num_recent: recent_fns[i](recent_ctxs[i])
///   if !flush_advance(flusher): return
/// ```
pub(crate) fn codegen_stratum_meta_fn(
    flush_advance_fn: FuncId,
    func_id: FuncId,
    module: &mut JITModule,
    builder_ctx: &mut FunctionBuilderContext,
    codegen_ctx: &mut cranelift_codegen::Context,
) -> Result<(), String> {
    let ptr_t = module.target_config().pointer_type();
    let ptr_size = ptr_t.bytes() as i32;

    let flush_advance_ref =
        module.declare_func_in_func(flush_advance_fn, &mut codegen_ctx.func);

    let mut builder = FunctionBuilder::new(&mut codegen_ctx.func, builder_ctx);

    // Build blocks
    let entry = builder.create_block();
    let full_hdr = builder.create_block();
    let full_body = builder.create_block();
    let post_full = builder.create_block();
    let inner_hdr = builder.create_block();
    let inner_body = builder.create_block();
    let post_inner = builder.create_block();
    let exit = builder.create_block();

    // Signature for indirect calls to packed rule variants: fn(*mut PackedJitCtx)
    let mut rule_sig = module.make_signature();
    rule_sig.params.push(AbiParam::new(ptr_t));
    let rule_sig_ref = builder.import_signature(rule_sig);

    // ─── entry ──────────────────────────────────────────────────────────
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let meta_ctx = builder.block_params(entry)[0];

    let full_fns = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 0i32);
    let full_ctxs = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, ptr_size);
    let num_full_i32 = builder.ins().load(I32, MemFlags::trusted(), meta_ctx, 2 * ptr_size);
    let num_recent_i32 =
        builder.ins().load(I32, MemFlags::trusted(), meta_ctx, 2 * ptr_size + 4i32);
    let recent_fns = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 3 * ptr_size);
    let recent_ctxs = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 4 * ptr_size);
    let flusher = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 5 * ptr_size);

    let num_full = builder.ins().uextend(ptr_t, num_full_i32);
    let num_recent = builder.ins().uextend(ptr_t, num_recent_i32);
    let zero = builder.ins().iconst(ptr_t, 0);
    let one = builder.ins().iconst(ptr_t, 1);

    builder.ins().jump(full_hdr, &[zero]);

    // ─── full_hdr(i): loop over full variants ───────────────────────────
    builder.append_block_param(full_hdr, ptr_t);
    builder.switch_to_block(full_hdr);
    // seal deferred: back-edge from full_body

    let i_full = builder.block_params(full_hdr)[0];
    let done_full =
        builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i_full, num_full);
    builder.ins().brif(done_full, post_full, &[], full_body, &[]);

    // ─── full_body ───────────────────────────────────────────────────────
    builder.switch_to_block(full_body);
    builder.seal_block(full_body);

    let byte_off_full = builder.ins().imul_imm(i_full, ptr_size as i64);
    let fn_addr_full = builder.ins().iadd(full_fns, byte_off_full);
    let fn_ptr_full =
        builder.ins().load(ptr_t, MemFlags::trusted(), fn_addr_full, 0i32);
    let ctx_addr_full = builder.ins().iadd(full_ctxs, byte_off_full);
    let ctx_ptr_full =
        builder.ins().load(ptr_t, MemFlags::trusted(), ctx_addr_full, 0i32);
    builder.ins().call_indirect(rule_sig_ref, fn_ptr_full, &[ctx_ptr_full]);
    let i_full_next = builder.ins().iadd(i_full, one);
    builder.ins().jump(full_hdr, &[i_full_next]);
    builder.seal_block(full_hdr);

    // ─── post_full: initial flush+advance ────────────────────────────────
    builder.switch_to_block(post_full);
    builder.seal_block(post_full);

    let call0 = builder.ins().call(flush_advance_ref, &[flusher]);
    let changed0 = builder.inst_results(call0)[0]; // i8: 0 = no change
    builder.ins().brif(changed0, inner_hdr, &[zero], exit, &[]);

    // ─── inner_hdr(i): semi-naive recent-variant loop header ────────────
    builder.append_block_param(inner_hdr, ptr_t);
    builder.switch_to_block(inner_hdr);
    // seal deferred: back-edges from inner_body and post_inner

    let i_inner = builder.block_params(inner_hdr)[0];
    let done_inner =
        builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i_inner, num_recent);
    builder.ins().brif(done_inner, post_inner, &[], inner_body, &[]);

    // ─── inner_body ──────────────────────────────────────────────────────
    builder.switch_to_block(inner_body);
    builder.seal_block(inner_body);

    let byte_off_inner = builder.ins().imul_imm(i_inner, ptr_size as i64);
    let fn_addr_inner = builder.ins().iadd(recent_fns, byte_off_inner);
    let fn_ptr_inner =
        builder.ins().load(ptr_t, MemFlags::trusted(), fn_addr_inner, 0i32);
    let ctx_addr_inner = builder.ins().iadd(recent_ctxs, byte_off_inner);
    let ctx_ptr_inner =
        builder.ins().load(ptr_t, MemFlags::trusted(), ctx_addr_inner, 0i32);
    builder.ins().call_indirect(rule_sig_ref, fn_ptr_inner, &[ctx_ptr_inner]);
    let i_inner_next = builder.ins().iadd(i_inner, one);
    builder.ins().jump(inner_hdr, &[i_inner_next]);

    // ─── post_inner: flush+advance + loop-back decision ──────────────────
    builder.switch_to_block(post_inner);
    builder.seal_block(post_inner);

    let call1 = builder.ins().call(flush_advance_ref, &[flusher]);
    let changed1 = builder.inst_results(call1)[0];
    builder.ins().brif(changed1, inner_hdr, &[zero], exit, &[]);
    builder.seal_block(inner_hdr); // all predecessors now known

    // ─── exit ─────────────────────────────────────────────────────────────
    builder.switch_to_block(exit);
    builder.seal_block(exit);
    builder.ins().return_(&[]);

    builder.finalize();

    module
        .define_function(func_id, codegen_ctx)
        .map_err(|e| format!("define stratum_meta_fn: {e}"))?;

    Ok(())
}

/// Generate the Stage 3 stratum function body.
///
/// Identical loop structure to `codegen_stratum_meta_fn` but uses
/// `PackedJitFnV3` (direct-insert) rule variants and calls
/// `jit_stratum_advance(all_rels, n_all_rels)` instead of `flush_advance`.
///
/// `StratumStage3Ctx` layout (repr C, 64-bit):
/// ```text
///   offset  0: full_fns    *const PackedJitFnV3           (ptr_size)
///   offset  8: full_ctxs   *const *mut PackedJitContextV3 (ptr_size)
///   offset 16: num_full    u32
///   offset 20: num_recent  u32
///   offset 24: recent_fns  *const PackedJitFnV3           (ptr_size)
///   offset 32: recent_ctxs *const *mut PackedJitContextV3 (ptr_size)
///   offset 40: all_rels    *const *mut PackedStorage       (ptr_size)
///   offset 48: n_all_rels  u32
/// ```
pub(crate) fn codegen_stratum_stage3_fn(
    advance_fn: FuncId,
    func_id: FuncId,
    module: &mut JITModule,
    builder_ctx: &mut FunctionBuilderContext,
    codegen_ctx: &mut cranelift_codegen::Context,
) -> Result<(), String> {
    let ptr_t = module.target_config().pointer_type();
    let ptr_size = ptr_t.bytes() as i32;

    // jit_stratum_advance(rels: ptr, n_rels: i32) -> i8
    let advance_ref = module.declare_func_in_func(advance_fn, &mut codegen_ctx.func);

    let mut builder = FunctionBuilder::new(&mut codegen_ctx.func, builder_ctx);

    let entry = builder.create_block();
    let full_hdr = builder.create_block();
    let full_body = builder.create_block();
    let post_full = builder.create_block();
    let inner_hdr = builder.create_block();
    let inner_body = builder.create_block();
    let post_inner = builder.create_block();
    let exit = builder.create_block();

    // Signature for indirect calls to V3 rule variants: fn(*mut PackedJitContextV3)
    let mut rule_sig = module.make_signature();
    rule_sig.params.push(AbiParam::new(ptr_t));
    let rule_sig_ref = builder.import_signature(rule_sig);

    // ─── entry ──────────────────────────────────────────────────────────
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let meta_ctx = builder.block_params(entry)[0];

    let full_fns = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 0i32);
    let full_ctxs = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, ptr_size);
    let num_full_i32 = builder.ins().load(I32, MemFlags::trusted(), meta_ctx, 2 * ptr_size);
    let num_recent_i32 =
        builder.ins().load(I32, MemFlags::trusted(), meta_ctx, 2 * ptr_size + 4i32);
    let recent_fns = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 3 * ptr_size);
    let recent_ctxs = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 4 * ptr_size);
    // all_rels @ offset 40 = 5 * ptr_size; n_all_rels @ offset 48 = 6 * ptr_size
    let all_rels = builder.ins().load(ptr_t, MemFlags::trusted(), meta_ctx, 5 * ptr_size);
    let n_all_rels = builder.ins().load(I32, MemFlags::trusted(), meta_ctx, 6 * ptr_size);

    let num_full = builder.ins().uextend(ptr_t, num_full_i32);
    let num_recent = builder.ins().uextend(ptr_t, num_recent_i32);
    let zero = builder.ins().iconst(ptr_t, 0);
    let one = builder.ins().iconst(ptr_t, 1);

    builder.ins().jump(full_hdr, &[zero]);

    // ─── full_hdr(i) ─────────────────────────────────────────────────
    builder.append_block_param(full_hdr, ptr_t);
    builder.switch_to_block(full_hdr);

    let i_full = builder.block_params(full_hdr)[0];
    let done_full = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i_full, num_full);
    builder.ins().brif(done_full, post_full, &[], full_body, &[]);

    // ─── full_body ────────────────────────────────────────────────────
    builder.switch_to_block(full_body);
    builder.seal_block(full_body);

    let byte_off_full = builder.ins().imul_imm(i_full, ptr_size as i64);
    let fn_addr_full = builder.ins().iadd(full_fns, byte_off_full);
    let fn_ptr_full = builder.ins().load(ptr_t, MemFlags::trusted(), fn_addr_full, 0i32);
    let ctx_addr_full = builder.ins().iadd(full_ctxs, byte_off_full);
    let ctx_ptr_full = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_addr_full, 0i32);
    builder.ins().call_indirect(rule_sig_ref, fn_ptr_full, &[ctx_ptr_full]);
    let i_full_next = builder.ins().iadd(i_full, one);
    builder.ins().jump(full_hdr, &[i_full_next]);
    builder.seal_block(full_hdr);

    // ─── post_full: initial advance ──────────────────────────────────
    builder.switch_to_block(post_full);
    builder.seal_block(post_full);

    let call0 = builder.ins().call(advance_ref, &[all_rels, n_all_rels]);
    let changed0 = builder.inst_results(call0)[0];
    builder.ins().brif(changed0, inner_hdr, &[zero], exit, &[]);

    // ─── inner_hdr(i) ─────────────────────────────────────────────────
    builder.append_block_param(inner_hdr, ptr_t);
    builder.switch_to_block(inner_hdr);

    let i_inner = builder.block_params(inner_hdr)[0];
    let done_inner =
        builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i_inner, num_recent);
    builder.ins().brif(done_inner, post_inner, &[], inner_body, &[]);

    // ─── inner_body ───────────────────────────────────────────────────
    builder.switch_to_block(inner_body);
    builder.seal_block(inner_body);

    let byte_off_inner = builder.ins().imul_imm(i_inner, ptr_size as i64);
    let fn_addr_inner = builder.ins().iadd(recent_fns, byte_off_inner);
    let fn_ptr_inner = builder.ins().load(ptr_t, MemFlags::trusted(), fn_addr_inner, 0i32);
    let ctx_addr_inner = builder.ins().iadd(recent_ctxs, byte_off_inner);
    let ctx_ptr_inner = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_addr_inner, 0i32);
    builder.ins().call_indirect(rule_sig_ref, fn_ptr_inner, &[ctx_ptr_inner]);
    let i_inner_next = builder.ins().iadd(i_inner, one);
    builder.ins().jump(inner_hdr, &[i_inner_next]);

    // ─── post_inner: advance + loop-back ──────────────────────────────
    builder.switch_to_block(post_inner);
    builder.seal_block(post_inner);

    let call1 = builder.ins().call(advance_ref, &[all_rels, n_all_rels]);
    let changed1 = builder.inst_results(call1)[0];
    builder.ins().brif(changed1, inner_hdr, &[zero], exit, &[]);
    builder.seal_block(inner_hdr);

    // ─── exit ─────────────────────────────────────────────────────────
    builder.switch_to_block(exit);
    builder.seal_block(exit);
    builder.ins().return_(&[]);

    builder.finalize();

    module
        .define_function(func_id, codegen_ctx)
        .map_err(|e| format!("define stratum_stage3_fn: {e}"))?;

    Ok(())
}

/// Generate the Stage 4 stratum function body.
///
/// Stage 4 inlines all rule bodies directly into a single Cranelift function,
/// eliminating `call_indirect` overhead and enabling cross-rule optimization.
///
/// `StratumStage4Ctx` layout (repr C, 64-bit):
/// ```text
///   offset  0: rule_ctxs   *const *mut PackedJitContextV3 (ptr_size)
///   offset  8: num_rules   u32                            (4 bytes)
///   offset 12: _pad        u32                            (4 bytes)
///   offset 16: all_rels    *const *mut PackedStorage       (ptr_size)
///   offset 24: n_all_rels  u32                            (4 bytes)
///   offset 32: handles_buf *mut JitLookupHandle           (ptr_size)
///   offset 40: lookup_specs *const LookupSpec             (ptr_size)
///   offset 48: total_handles u32                          (4 bytes)
/// ```
///
/// Loop structure:
/// ```text
/// full_body:
///   for each rule i:
///     ctx_i = rule_ctxs[i]
///     inline full body of rule i
///   jit_stratum_advance_s4(stage4_ctx) -> changed
///   if !changed: return
/// loop:
///   for each rule i, for each clause j:
///     ctx_i = rule_ctxs[i]
///     inline recent-j body of rule i
///   jit_stratum_advance_s4(stage4_ctx) -> changed
///   if !changed: return
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn codegen_stratum_stage4_fn(
    advance_s4_fn: FuncId,
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    func_id: FuncId,
    module: &mut JITModule,
    builder_ctx: &mut FunctionBuilderContext,
    codegen_ctx: &mut cranelift_codegen::Context,
    helpers: &PackedJitHelperIds,
) -> Result<(), String> {
    let ptr_t = module.target_config().pointer_type();

    let advance_ref = module.declare_func_in_func(advance_s4_fn, &mut codegen_ctx.func);

    let func_refs = FuncRefsV3 {
        packed_count: module.declare_func_in_func(helpers.packed_count, &mut codegen_ctx.func),
        packed_data_ptr: module.declare_func_in_func(helpers.packed_data_ptr, &mut codegen_ctx.func),
        packed_recent_idx: module.declare_func_in_func(helpers.packed_recent_idx, &mut codegen_ctx.func),
        packed_try_insert: module.declare_func_in_func(helpers.packed_try_insert, &mut codegen_ctx.func),
    };

    let mut builder = FunctionBuilder::new(&mut codegen_ctx.func, builder_ctx);

    // ─── Pre-allocate key blocks ─────────────────────────────────────────────
    let entry = builder.create_block();
    let full_body = builder.create_block();
    let post_full = builder.create_block();
    let inner_hdr = builder.create_block();
    let post_inner = builder.create_block();
    let exit = builder.create_block();

    // ─── entry ──────────────────────────────────────────────────────────────
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let stage4_ctx = builder.block_params(entry)[0];

    // Load StratumStage4Ctx fields
    let rule_ctxs_val = builder.ins().load(ptr_t, MemFlags::trusted(), stage4_ctx, 0i32);

    // Load per-rule ctx pointers once (they don't change during execution)
    let mut rule_ctx_vals = Vec::with_capacity(rules.len());
    for i in 0..rules.len() {
        let offset = builder.ins().iconst(ptr_t, (i as i64) * 8);
        let addr = builder.ins().iadd(rule_ctxs_val, offset);
        let ctx_i = builder.ins().load(ptr_t, MemFlags::trusted(), addr, 0);
        rule_ctx_vals.push(ctx_i);
    }

    builder.ins().jump(full_body, &[]);

    // ─── full_body: inline full variant of each rule ─────────────────────────
    builder.switch_to_block(full_body);
    builder.seal_block(full_body);

    let mut next_var = 0usize;

    emit_rule_bodies(
        &mut builder,
        rules,
        &rule_ctx_vals,
        None, // full variant — no recent clause
        &func_refs,
        ptr_t,
        &mut next_var,
        post_full,
    );

    // ─── post_full: initial advance ──────────────────────────────────────────
    builder.switch_to_block(post_full);
    builder.seal_block(post_full);

    // jit_stratum_advance_s4(stage4_ctx) -> i8
    let call0 = builder.ins().call(advance_ref, &[stage4_ctx]);
    let changed0 = builder.inst_results(call0)[0];
    builder.ins().brif(changed0, inner_hdr, &[], exit, &[]);

    // ─── inner_hdr: recent-variant fixpoint loop ─────────────────────────────
    builder.switch_to_block(inner_hdr);
    // Will be sealed after post_inner back-edge is known.

    // Inline all recent variants for all rules. Each rule has one recent
    // variant per clause (recent_clause_idx = body index of that clause).
    // We iterate: for each rule, for each clause_seq_idx, emit recent body.
    let mut recent_next_var = next_var;

    // Build a flat list of (rule_idx, clause_body_idx) for all recent variants
    // so we can emit them sequentially.
    //
    // clause_body_idx mirrors the `body_idx` used in packed_codegen — it is the
    // sequential index of each CClause in the flat clauses slice for the rule.
    // Since `rules[i].0` is already the pre-filtered slice of only CClause items,
    // we iterate over their sequential positions (0..clauses.len()).
    let mut recent_emissions: Vec<(usize, usize)> = Vec::new();
    for (rule_i, (clauses, _, _)) in rules.iter().enumerate() {
        for clause_seq in 0..clauses.len() {
            recent_emissions.push((rule_i, clause_seq));
        }
    }

    if recent_emissions.is_empty() {
        // No clauses => nothing to do in recent loop; jump straight to post_inner
        builder.ins().jump(post_inner, &[]);
    } else {
        emit_recent_rule_bodies(
            &mut builder,
            rules,
            &rule_ctx_vals,
            &recent_emissions,
            &func_refs,
            ptr_t,
            &mut recent_next_var,
            post_inner,
        );
    }

    // ─── post_inner: advance + loop-back decision ────────────────────────────
    builder.switch_to_block(post_inner);
    builder.seal_block(post_inner);

    let call1 = builder.ins().call(advance_ref, &[stage4_ctx]);
    let changed1 = builder.inst_results(call1)[0];
    builder.ins().brif(changed1, inner_hdr, &[], exit, &[]);
    // inner_hdr is now fully sealed — all predecessors (post_full, post_inner) known.
    builder.seal_block(inner_hdr);

    // ─── exit ────────────────────────────────────────────────────────────────
    builder.switch_to_block(exit);
    builder.seal_block(exit);
    builder.ins().return_(&[]);

    builder.finalize();

    module
        .define_function(func_id, codegen_ctx)
        .map_err(|e| format!("define stratum_stage4_fn: {e}"))?;

    Ok(())
}

/// Inline the full (no-recent) variant of every rule sequentially, then jump to `continuation`.
///
/// After the last rule body finishes, we emit a jump to `continuation`.
#[allow(clippy::too_many_arguments)]
fn emit_rule_bodies(
    builder: &mut FunctionBuilder,
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    rule_ctx_vals: &[cranelift_codegen::ir::Value],
    recent_clause_idx: Option<usize>,
    func_refs: &FuncRefsV3,
    ptr_t: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    continuation: Block,
) {
    if rules.is_empty() {
        builder.ins().jump(continuation, &[]);
        return;
    }

    for (rule_i, (clauses, heads, conditions)) in rules.iter().enumerate() {
        let ctx_i = rule_ctx_vals[rule_i];

        // Load PackedJitContextV3 fields: rels @ 0, bindings @ 16, head_rels @ 24, lookup_handles @ 32
        let rels_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 0i32);
        let bindings_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 16i32);
        let head_rels_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 24i32);
        let lookup_handles_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 32i32);

        // Build the (body_idx, &CClause) pairs as gen_clauses_v3 expects.
        // body_idx is the sequential clause index (0-based within the clauses slice).
        let indexed_clauses: Vec<(usize, &CClause)> =
            clauses.iter().enumerate().collect();

        let cond_refs: Vec<&CExpr> = conditions.iter().collect();

        gen_clauses_v3(
            builder,
            &indexed_clauses,
            0,
            recent_clause_idx,
            rels_i,
            bindings_i,
            head_rels_i,
            lookup_handles_i,
            func_refs,
            heads,
            ptr_t,
            next_var,
            &cond_refs,
        );
        // After gen_clauses_v3, the builder is at the loop_exit block of the
        // outermost scan. We need to connect to the next rule or the continuation.
        // gen_clauses_v3 leaves us positioned at loop_exit — emit the bridge jump.
        if rule_i + 1 < rules.len() {
            let next_rule_block = builder.create_block();
            builder.ins().jump(next_rule_block, &[]);
            builder.switch_to_block(next_rule_block);
            builder.seal_block(next_rule_block);
        } else {
            builder.ins().jump(continuation, &[]);
        }
    }
}

/// Inline recent variants for multiple (rule, clause_seq) pairs sequentially.
#[allow(clippy::too_many_arguments)]
fn emit_recent_rule_bodies(
    builder: &mut FunctionBuilder,
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    rule_ctx_vals: &[cranelift_codegen::ir::Value],
    recent_emissions: &[(usize, usize)],
    func_refs: &FuncRefsV3,
    ptr_t: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    continuation: Block,
) {
    for (emit_i, &(rule_i, clause_seq)) in recent_emissions.iter().enumerate() {
        let (clauses, heads, conditions) = &rules[rule_i];
        let ctx_i = rule_ctx_vals[rule_i];

        let rels_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 0i32);
        let bindings_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 16i32);
        let head_rels_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 24i32);
        let lookup_handles_i = builder.ins().load(ptr_t, MemFlags::trusted(), ctx_i, 32i32);

        let indexed_clauses: Vec<(usize, &CClause)> =
            clauses.iter().enumerate().collect();

        let cond_refs: Vec<&CExpr> = conditions.iter().collect();

        // recent_clause_idx = the body index of the "recent" clause in this variant.
        // Since indexed_clauses uses 0-based sequential indices, recent = clause_seq.
        gen_clauses_v3(
            builder,
            &indexed_clauses,
            0,
            Some(clause_seq),
            rels_i,
            bindings_i,
            head_rels_i,
            lookup_handles_i,
            func_refs,
            heads,
            ptr_t,
            next_var,
            &cond_refs,
        );

        if emit_i + 1 < recent_emissions.len() {
            let next_block = builder.create_block();
            builder.ins().jump(next_block, &[]);
            builder.switch_to_block(next_block);
            builder.seal_block(next_block);
        } else {
            builder.ins().jump(continuation, &[]);
        }
    }
}
