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
