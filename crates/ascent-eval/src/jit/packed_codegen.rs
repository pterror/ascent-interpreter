//! Cranelift IR generation for the typed packed JIT.
//!
//! Generates native code that reads u32 values directly from PackedStorage's
//! `packed_data` buffer, with u32 bindings and integer comparisons — no Value
//! enum, no cloning, no Option<Value> overhead.
//!
//! Generated function signature: `fn(*mut PackedJitContext) -> ()`
//!
//! Inner-loop body for a 2-clause rule is roughly:
//!   packed_buf = packed_data_ptr(rel)
//!   for each tuple_idx in index[bound_key]:
//!     v = load i32 at packed_buf + tuple_idx * arity * 4 + col * 4
//!     store v at bindings + var_id * 4         // bind fresh var
//!     check: load i32 from tuple_ptr, compare with bindings[var_id]  // bound check
//!     emit head: load bindings[var_id], push into results
//!
//! No function calls inside the inner loop except packed_lookup (once per
//! outer iteration) and packed_recent_idx (once per iteration, recent path only).

use cranelift_codegen::entity::EntityRef;
use cranelift_codegen::ir::Value as CValue;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types::I32;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Module};

use crate::compiled::{CBodyItem, CBinOp, CClause, CClauseArg, CExpr, CHeadClause, CRule, CUnOp};
use crate::jit::PackedJitHelperIds;
use crate::value::Value;

/// Pre-declared function references used during code generation.
struct FuncRefs {
    packed_count: cranelift_codegen::ir::FuncRef,
    packed_data_ptr: cranelift_codegen::ir::FuncRef,
    packed_recent_idx: cranelift_codegen::ir::FuncRef,
    packed_lookup: cranelift_codegen::ir::FuncRef,
    packed_push_result: cranelift_codegen::ir::FuncRef,
}

/// Function references for Stage 3 direct-insert code generation.
pub(crate) struct FuncRefsV3 {
    pub(crate) packed_count: cranelift_codegen::ir::FuncRef,
    pub(crate) packed_data_ptr: cranelift_codegen::ir::FuncRef,
    pub(crate) packed_recent_ptr: cranelift_codegen::ir::FuncRef,
    pub(crate) packed_try_insert: cranelift_codegen::ir::FuncRef,
}

/// Fields loaded from PackedJitContext at function entry.
struct CtxFields {
    rels: CValue,
    bindings: CValue,
    results: CValue,
}

/// Generate a packed JIT function for a rule variant.
///
/// `recent_clause_idx`: None = all clauses use full data, Some(body_offset) = that clause uses recent.
pub fn codegen_packed_rule_body(
    rule: &CRule,
    recent_clause_idx: Option<usize>,
    func_id: FuncId,
    module: &mut impl Module,
    builder_ctx: &mut FunctionBuilderContext,
    ctx: &mut cranelift_codegen::Context,
    helpers: &PackedJitHelperIds,
) -> Result<(), String> {
    let ptr_type = module.target_config().pointer_type();

    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.returns.clear();

    let func_refs = FuncRefs {
        packed_count: module.declare_func_in_func(helpers.packed_count, &mut ctx.func),
        packed_data_ptr: module.declare_func_in_func(helpers.packed_data_ptr, &mut ctx.func),
        packed_recent_idx: module.declare_func_in_func(helpers.packed_recent_idx, &mut ctx.func),
        packed_lookup: module.declare_func_in_func(helpers.packed_lookup, &mut ctx.func),
        packed_push_result: module.declare_func_in_func(helpers.packed_push_result, &mut ctx.func),
    };

    let mut builder = FunctionBuilder::new(&mut ctx.func, builder_ctx);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let ctx_ptr = builder.block_params(entry_block)[0];

    // Load fields from PackedJitContext (repr(C) offsets, see packed_helpers.rs)
    let flags = MemFlags::trusted();
    let rels = builder.ins().load(ptr_type, flags, ctx_ptr, 0_i32);
    let bindings = builder.ins().load(ptr_type, flags, ctx_ptr, 16_i32);
    let results = builder.ins().load(ptr_type, flags, ctx_ptr, 24_i32);

    let fields = CtxFields {
        rels,
        bindings,
        results,
    };

    // Collect clauses in body order with their body index (for recent dispatch)
    let clauses: Vec<(usize, &CClause)> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, item)| match item {
            CBodyItem::Clause(c) => Some((i, c)),
            _ => None,
        })
        .collect();

    // Collect all If-conditions from body items
    let conditions: Vec<&CExpr> = rule
        .body
        .iter()
        .filter_map(|item| match item {
            CBodyItem::Condition(crate::compiled::CCondition::If(expr)) => Some(expr),
            _ => None,
        })
        .collect();

    let mut next_var = 0usize;

    gen_clauses(
        &mut builder,
        &clauses,
        0,
        recent_clause_idx,
        &fields,
        &func_refs,
        &rule.heads,
        ptr_type,
        &mut next_var,
        &conditions,
    );

    builder.ins().return_(&[]);
    builder.finalize();

    module
        .define_function(func_id, ctx)
        .map_err(|e| format!("define_function: {e}"))?;

    Ok(())
}

/// Recursively generate code for clauses starting at `clause_offset`.
#[allow(clippy::too_many_arguments)]
fn gen_clauses(
    builder: &mut FunctionBuilder,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    fields: &CtxFields,
    func_refs: &FuncRefs,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
) {
    if clause_offset >= clauses.len() {
        if !conditions.is_empty() {
            let done_block = builder.create_block();
            for &expr in conditions {
                let cond_val = compile_packed_expr(builder, expr, fields.bindings)
                    .expect("compile_packed_expr: should succeed for eligible exprs");
                let pass_block = builder.create_block();
                builder.ins().brif(cond_val, pass_block, &[], done_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            gen_emit_heads(builder, heads, fields, func_refs, ptr_type);
            builder.ins().jump(done_block, &[]);
            builder.switch_to_block(done_block);
            builder.seal_block(done_block);
        } else {
            gen_emit_heads(builder, heads, fields, func_refs, ptr_type);
        }
        return;
    }

    let (body_idx, clause) = clauses[clause_offset];
    let use_recent = recent_clause_idx == Some(body_idx);
    let rel_seq_idx = clause_offset; // index into rels[] array in context

    // Load this clause's relation pointer: rels[rel_seq_idx]
    let rel_ptr_offset = builder.ins().iconst(ptr_type, (rel_seq_idx as i64) * 8);
    let rel_ptr_addr = builder.ins().iadd(fields.rels, rel_ptr_offset);
    let rel_ptr = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), rel_ptr_addr, 0);

    // Get packed_data pointer for this relation (used in both full and index scan)
    let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
    let packed_buf = builder.inst_results(call)[0];

    let arity = clause.args.len();
    let use_recent_val = builder.ins().iconst(I32, if use_recent { 1 } else { 0 });

    if clause.bound_cols.is_empty() {
        gen_full_scan(
            builder,
            clause,
            rel_ptr,
            packed_buf,
            arity,
            use_recent,
            use_recent_val,
            clauses,
            clause_offset,
            recent_clause_idx,
            fields,
            func_refs,
            heads,
            ptr_type,
            next_var,
            conditions,
        );
    } else {
        gen_index_scan(
            builder,
            clause,
            rel_ptr,
            packed_buf,
            arity,
            use_recent_val,
            clauses,
            clause_offset,
            recent_clause_idx,
            fields,
            func_refs,
            heads,
            ptr_type,
            next_var,
            conditions,
        );
    }
}

/// Full scan: no bound columns, iterate all (or recent) tuples.
#[allow(clippy::too_many_arguments)]
fn gen_full_scan(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    rel_ptr: CValue,
    packed_buf: CValue,
    arity: usize,
    use_recent: bool,
    use_recent_val: CValue,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    fields: &CtxFields,
    func_refs: &FuncRefs,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
) {
    // count = packed_count(rel, use_recent)
    let call = builder
        .ins()
        .call(func_refs.packed_count, &[rel_ptr, use_recent_val]);
    let count = builder.inst_results(call)[0];

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let var_i = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_i, ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(var_i, zero);
    builder.ins().jump(loop_header, &[]);

    // Loop header: if i >= count, exit
    builder.switch_to_block(loop_header);
    let i = builder.use_var(var_i);
    let cmp = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, i, count);
    builder.ins().brif(cmp, loop_exit, &[], loop_body, &[]);

    // Loop body
    builder.switch_to_block(loop_body);
    let i = builder.use_var(var_i);

    // Resolve actual tuple index (full: i itself; recent: packed_recent_idx(rel, i))
    let tuple_idx = if use_recent {
        let call = builder
            .ins()
            .call(func_refs.packed_recent_idx, &[rel_ptr, i]);
        builder.inst_results(call)[0]
    } else {
        i
    };

    // tuple_ptr = packed_buf + tuple_idx * arity * 4
    let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

    // Literal arg checks: for each literal clause arg, compare against packed value.
    // On mismatch, jump to continue_block (increment i and loop again).
    let continue_block = builder.create_block();
    for (col, arg) in clause.args.iter().enumerate() {
        match arg {
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, *n as i64);
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            _ => {}
        }
    }

    // Bind fresh vars: store packed_data[col] into bindings[var_id]
    for &(col, var_id) in &clause.fresh_cols {
        let val = load_packed_col(builder, tuple_ptr, col);
        store_binding(builder, fields.bindings, var_id, val);
    }

    // Recurse to next clause
    gen_clauses(
        builder,
        clauses,
        clause_offset + 1,
        recent_clause_idx,
        fields,
        func_refs,
        heads,
        ptr_type,
        next_var,
        conditions,
    );

    builder.ins().jump(continue_block, &[]);
    builder.switch_to_block(continue_block);
    builder.seal_block(continue_block);

    // i += 1
    let i = builder.use_var(var_i);
    let one = builder.ins().iconst(ptr_type, 1);
    let i_next = builder.ins().iadd(i, one);
    builder.def_var(var_i, i_next);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);
}

/// Index scan: use first bound column for index lookup, check remaining bound cols with icmp.
#[allow(clippy::too_many_arguments)]
fn gen_index_scan(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    rel_ptr: CValue,
    packed_buf: CValue,
    arity: usize,
    use_recent_val: CValue,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    fields: &CtxFields,
    func_refs: &FuncRefs,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
) {
    // Primary bound column: use for index lookup
    let primary_col = clause.bound_cols[0];

    // key = bound value (from bindings for Var, or iconst for Literal)
    let key_i32 = load_bound_val(builder, clause, primary_col, fields.bindings);

    // (indices_ptr, indices_len) = packed_lookup(rel, col, key, use_recent)
    let col_val = builder.ins().iconst(I32, primary_col as i64);
    let call = builder.ins().call(
        func_refs.packed_lookup,
        &[rel_ptr, col_val, key_i32, use_recent_val],
    );
    let results = builder.inst_results(call);
    let indices_ptr = results[0];
    let indices_len = results[1];

    // Loop over matching tuple indices
    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let var_j = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_j, ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(var_j, zero);
    builder.ins().jump(loop_header, &[]);

    // Loop header: if j >= indices_len, exit
    builder.switch_to_block(loop_header);
    let j = builder.use_var(var_j);
    let cmp = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, j, indices_len);
    builder.ins().brif(cmp, loop_exit, &[], loop_body, &[]);

    // Loop body
    builder.switch_to_block(loop_body);
    let j = builder.use_var(var_j);

    // tuple_idx = indices[j]  (usize = ptr_type)
    let idx_byte_offset = builder.ins().imul_imm(j, 8_i64); // usize = 8 bytes
    let idx_addr = builder.ins().iadd(indices_ptr, idx_byte_offset);
    let tuple_idx = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), idx_addr, 0);

    // tuple_ptr = packed_buf + tuple_idx * arity * 4
    let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

    // Check secondary bound columns with integer comparison
    // Each failing check jumps to the continue block (increment j)
    let continue_block = builder.create_block();
    let mut inner_blocks_to_seal = Vec::new();

    // Literal arg checks (before secondary bound col checks)
    for (col, arg) in clause.args.iter().enumerate() {
        match arg {
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, *n as i64);
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            _ => {}
        }
    }

    for &col in &clause.bound_cols[1..] {
        let actual = load_packed_col(builder, tuple_ptr, col);
        let expected = load_bound_val(builder, clause, col, fields.bindings);
        // Compare as I32 (u32 equality — intern IDs, bit-cast i32s, etc.)
        let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
        let pass_block = builder.create_block();
        builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
        inner_blocks_to_seal.push(pass_block);
        builder.switch_to_block(pass_block);
    }

    // All checks passed: bind fresh vars and recurse
    for &(col, var_id) in &clause.fresh_cols {
        let val = load_packed_col(builder, tuple_ptr, col);
        store_binding(builder, fields.bindings, var_id, val);
    }

    gen_clauses(
        builder,
        clauses,
        clause_offset + 1,
        recent_clause_idx,
        fields,
        func_refs,
        heads,
        ptr_type,
        next_var,
        conditions,
    );

    builder.ins().jump(continue_block, &[]);

    // Seal inner blocks (one per secondary bound col check)
    builder.switch_to_block(continue_block);
    for blk in inner_blocks_to_seal {
        builder.seal_block(blk);
    }
    builder.seal_block(continue_block);

    // j += 1
    let j = builder.use_var(var_j);
    let one = builder.ins().iconst(ptr_type, 1);
    let j_next = builder.ins().iadd(j, one);
    builder.def_var(var_j, j_next);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);
}

/// Emit all head tuples at the innermost match point.
fn gen_emit_heads(
    builder: &mut FunctionBuilder,
    heads: &[CHeadClause],
    fields: &CtxFields,
    func_refs: &FuncRefs,
    ptr_type: cranelift_codegen::ir::Type,
) {
    for (head_idx, head) in heads.iter().enumerate() {
        let arity = head.args.len();
        if arity == 0 {
            // Zero-arity head: push empty tuple
            let null = builder.ins().iconst(ptr_type, 0);
            let head_idx_val = builder.ins().iconst(ptr_type, head_idx as i64);
            let arity_val = builder.ins().iconst(I32, 0);
            builder.ins().call(
                func_refs.packed_push_result,
                &[fields.results, head_idx_val, null, arity_val],
            );
            continue;
        }

        // Stack slot for head tuple: arity * 4 bytes
        let slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (arity * 4) as u32,
            2, // align = 4 bytes (2^2)
        ));

        // Copy each head arg (must be CExpr::Var) from bindings into the slot
        for (col, arg) in head.args.iter().enumerate() {
            let var_id = match arg {
                CExpr::Var(id) => *id,
                _ => panic!("packed JIT: non-Var head arg (should be caught by eligibility check)"),
            };
            let val = load_binding(builder, fields.bindings, var_id);
            let slot_addr = builder.ins().stack_addr(ptr_type, slot, (col * 4) as i32);
            builder.ins().store(MemFlags::trusted(), val, slot_addr, 0);
        }

        let slot_addr = builder.ins().stack_addr(ptr_type, slot, 0);
        let head_idx_val = builder.ins().iconst(ptr_type, head_idx as i64);
        let arity_val = builder.ins().iconst(I32, arity as i64);
        builder.ins().call(
            func_refs.packed_push_result,
            &[fields.results, head_idx_val, slot_addr, arity_val],
        );
    }
}

// ─── Inline utilities ────────────────────────────────────────────────

/// Compute `packed_buf + tuple_idx * arity * 4`.
fn compute_tuple_ptr(
    builder: &mut FunctionBuilder,
    packed_buf: CValue,
    tuple_idx: CValue,
    arity: usize,
) -> CValue {
    let byte_offset = builder.ins().imul_imm(tuple_idx, (arity * 4) as i64);
    builder.ins().iadd(packed_buf, byte_offset)
}

/// Load packed value at `tuple_ptr + col * 4` as I32.
fn load_packed_col(builder: &mut FunctionBuilder, tuple_ptr: CValue, col: usize) -> CValue {
    builder
        .ins()
        .load(I32, MemFlags::trusted(), tuple_ptr, (col * 4) as i32)
}

/// Load binding value for `var_id` from the flat u32 bindings array.
fn load_binding(builder: &mut FunctionBuilder, bindings_ptr: CValue, var_id: u32) -> CValue {
    let addr = builder.ins().iadd_imm(bindings_ptr, (var_id as i64) * 4);
    builder.ins().load(I32, MemFlags::trusted(), addr, 0)
}

/// Store `val` (I32) into bindings[var_id].
fn store_binding(builder: &mut FunctionBuilder, bindings_ptr: CValue, var_id: u32, val: CValue) {
    let addr = builder.ins().iadd_imm(bindings_ptr, (var_id as i64) * 4);
    builder.ins().store(MemFlags::trusted(), val, addr, 0);
}

/// Load the I32 key value for a bound column: from bindings for Var args, or iconst for literals.
fn load_bound_val(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    col: usize,
    bindings_ptr: CValue,
) -> CValue {
    match &clause.args[col] {
        CClauseArg::Var(id) => load_binding(builder, bindings_ptr, *id),
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            builder.ins().iconst(I32, *n as i64)
        }
        CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
            builder.ins().iconst(I32, if *b { 1 } else { 0 })
        }
        CClauseArg::Expr(_) => {
            panic!("packed JIT: non-literal Expr arg in bound column (should be caught by eligibility)")
        }
    }
}

// ─── Condition compilation ───────────────────────────────────────────

/// Compile a CExpr to a Cranelift I32 value for use in conditions.
/// Comparison results are extended to I32 via `select`.
fn compile_packed_expr(
    builder: &mut FunctionBuilder,
    expr: &CExpr,
    bindings: CValue,
) -> Result<CValue, String> {
    match expr {
        CExpr::Var(id) => Ok(load_binding(builder, bindings, *id)),
        CExpr::Literal(Value::I32(n)) => Ok(builder.ins().iconst(I32, *n as i64)),
        CExpr::Literal(Value::Bool(b)) => Ok(builder.ins().iconst(I32, if *b { 1 } else { 0 })),
        CExpr::VarBinVar(op, a, b) => {
            let av = load_binding(builder, bindings, *a);
            let bv = load_binding(builder, bindings, *b);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::VarBinLit(op, a, Value::I32(n)) => {
            let av = load_binding(builder, bindings, *a);
            let bv = builder.ins().iconst(I32, *n as i64);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::LitBinVar(op, Value::I32(n), b) => {
            let av = builder.ins().iconst(I32, *n as i64);
            let bv = load_binding(builder, bindings, *b);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::Binary(op, a, b) => {
            let av = compile_packed_expr(builder, a, bindings)?;
            let bv = compile_packed_expr(builder, b, bindings)?;
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::Unary(CUnOp::Not, inner) => {
            let v = compile_packed_expr(builder, inner, bindings)?;
            let one = builder.ins().iconst(I32, 1);
            Ok(builder.ins().bxor(v, one))
        }
        CExpr::Unary(CUnOp::Neg, inner) => {
            let v = compile_packed_expr(builder, inner, bindings)?;
            Ok(builder.ins().ineg(v))
        }
        _ => Err(format!("packed JIT: unsupported expr in condition: {expr:?}")),
    }
}

fn compile_packed_binop(
    builder: &mut FunctionBuilder,
    op: CBinOp,
    a: CValue,
    b: CValue,
) -> Result<CValue, String> {
    let one = builder.ins().iconst(I32, 1);
    let zero = builder.ins().iconst(I32, 0);
    match op {
        CBinOp::Add => Ok(builder.ins().iadd(a, b)),
        CBinOp::Sub => Ok(builder.ins().isub(a, b)),
        CBinOp::Mul => Ok(builder.ins().imul(a, b)),
        CBinOp::Eq => {
            let cmp = builder.ins().icmp(IntCC::Equal, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        CBinOp::Ne => {
            let cmp = builder.ins().icmp(IntCC::NotEqual, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        CBinOp::Lt => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        CBinOp::Le => {
            let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        CBinOp::Gt => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        CBinOp::Ge => {
            let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
            Ok(builder.ins().select(cmp, one, zero))
        }
        _ => Err(format!("packed JIT: unsupported binop in condition: {op:?}")),
    }
}

// ─── Stage 3: direct-insert rule codegen ─────────────────────────────

/// Generate a Stage 3 packed JIT function for a rule variant.
///
/// Identical to `codegen_packed_rule_body` except head tuples are written
/// directly to `PackedJitContextV3.head_rels[head_idx]` via `packed_try_insert`
/// instead of being pushed onto a results buffer.
pub(crate) fn codegen_packed_rule_body_v3(
    rule: &CRule,
    recent_clause_idx: Option<usize>,
    func_id: cranelift_module::FuncId,
    module: &mut impl cranelift_module::Module,
    builder_ctx: &mut cranelift_frontend::FunctionBuilderContext,
    ctx: &mut cranelift_codegen::Context,
    helpers: &PackedJitHelperIds,
) -> Result<(), String> {
    let ptr_type = module.target_config().pointer_type();

    ctx.func.signature.params.push(AbiParam::new(ptr_type));
    ctx.func.signature.returns.clear();

    let func_refs = FuncRefsV3 {
        packed_count: module.declare_func_in_func(helpers.packed_count, &mut ctx.func),
        packed_data_ptr: module.declare_func_in_func(helpers.packed_data_ptr, &mut ctx.func),
        packed_recent_ptr: module.declare_func_in_func(helpers.packed_recent_ptr, &mut ctx.func),
        packed_try_insert: module.declare_func_in_func(helpers.packed_try_insert, &mut ctx.func),
    };

    let mut builder = FunctionBuilder::new(&mut ctx.func, builder_ctx);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let ctx_ptr = builder.block_params(entry_block)[0];

    let flags = MemFlags::trusted();
    // Load context fields (PackedJitContextV3 layout)
    let rels = builder.ins().load(ptr_type, flags, ctx_ptr, 0_i32);
    let bindings = builder.ins().load(ptr_type, flags, ctx_ptr, 16_i32);
    let head_rels = builder.ins().load(ptr_type, flags, ctx_ptr, 24_i32); // V3: head_rels
    let lookup_handles = builder.ins().load(ptr_type, flags, ctx_ptr, 32_i32); // inline probe handles

    let clauses: Vec<(usize, &CClause)> = rule
        .body
        .iter()
        .enumerate()
        .filter_map(|(i, item)| match item {
            CBodyItem::Clause(c) => Some((i, c)),
            _ => None,
        })
        .collect();

    let conditions: Vec<&CExpr> = rule
        .body
        .iter()
        .filter_map(|item| match item {
            CBodyItem::Condition(crate::compiled::CCondition::If(expr)) => Some(expr),
            _ => None,
        })
        .collect();

    let mut next_var = 0usize;

    gen_clauses_v3(
        &mut builder,
        &clauses,
        0,
        recent_clause_idx,
        rels,
        bindings,
        head_rels,
        lookup_handles,
        &func_refs,
        &rule.heads,
        ptr_type,
        &mut next_var,
        &conditions,
    );

    builder.ins().return_(&[]);
    builder.finalize();

    module
        .define_function(func_id, ctx)
        .map_err(|e| format!("define_function v3: {e}"))?;

    Ok(())
}

/// Recursively generate Stage 3 clause-matching code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gen_clauses_v3(
    builder: &mut FunctionBuilder,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    rels: CValue,
    bindings: CValue,
    head_rels: CValue,
    lookup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
) {
    if clause_offset >= clauses.len() {
        if !conditions.is_empty() {
            let done_block = builder.create_block();
            for &expr in conditions {
                let cond_val = compile_packed_expr(builder, expr, bindings)
                    .expect("compile_packed_expr: should succeed for eligible exprs");
                let pass_block = builder.create_block();
                builder.ins().brif(cond_val, pass_block, &[], done_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            gen_emit_heads_v3(builder, heads, head_rels, bindings, func_refs, ptr_type);
            builder.ins().jump(done_block, &[]);
            builder.switch_to_block(done_block);
            builder.seal_block(done_block);
        } else {
            gen_emit_heads_v3(builder, heads, head_rels, bindings, func_refs, ptr_type);
        }
        return;
    }

    let (body_idx, clause) = clauses[clause_offset];
    let use_recent = recent_clause_idx == Some(body_idx);
    let rel_seq_idx = clause_offset;

    let rel_ptr_offset = builder.ins().iconst(ptr_type, (rel_seq_idx as i64) * 8);
    let rel_ptr_addr = builder.ins().iadd(rels, rel_ptr_offset);
    let rel_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), rel_ptr_addr, 0);

    let arity = clause.args.len();
    let use_recent_val = builder.ins().iconst(I32, if use_recent { 1 } else { 0 });

    // A rule is recursive w.r.t. this clause if its head writes to the same
    // relation that this clause reads from.  When recursive, packed_data_ptr
    // must be re-fetched inside the scan loop because a direct head insert can
    // reallocate the buffer.  When non-recursive the pointer is stable and can
    // be cached once before the loop.
    let is_recursive = heads.iter().any(|h| h.relation == clause.relation);

    if clause.bound_cols.is_empty() {
        gen_full_scan_v3(
            builder, clause, rel_ptr, arity, use_recent, use_recent_val,
            clauses, clause_offset, recent_clause_idx, rels, bindings, head_rels,
            lookup_handles, func_refs, heads, ptr_type, next_var, conditions,
            is_recursive,
        );
    } else {
        gen_index_scan_v3(
            builder, clause, rel_ptr, arity, use_recent,
            clauses, clause_offset, recent_clause_idx, rels, bindings, head_rels,
            lookup_handles, func_refs, heads, ptr_type, next_var, conditions,
            is_recursive,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gen_full_scan_v3(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    rel_ptr: CValue,
    arity: usize,
    use_recent: bool,
    use_recent_val: CValue,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    rels: CValue,
    bindings: CValue,
    head_rels: CValue,
    lookup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    is_recursive: bool,
) {
    let call = builder.ins().call(func_refs.packed_count, &[rel_ptr, use_recent_val]);
    let count = builder.inst_results(call)[0];

    // For non-recursive rules the packed_data buffer cannot be reallocated during
    // the scan, so we fetch the pointer once here (before the loop) and reuse it.
    // For recursive rules (head writes to this clause's relation) we must re-fetch
    // on every iteration because packed_try_insert may grow the Vec.
    let cached_packed_buf = if !is_recursive {
        let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
        Some(builder.inst_results(call)[0])
    } else {
        None
    };

    // For recent scans, fetch the recent-index array pointer once before the loop.
    // recent[i] is a usize (pointer-sized) stored at recent_ptr + i * ptr_size.
    // This replaces the per-iteration packed_recent_idx call with an inline load.
    let cached_recent_ptr = if use_recent {
        let call = builder.ins().call(func_refs.packed_recent_ptr, &[rel_ptr]);
        Some(builder.inst_results(call)[0])
    } else {
        None
    };

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    let var_i = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_i, ptr_type);
    let zero = builder.ins().iconst(ptr_type, 0);
    builder.def_var(var_i, zero);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_header);
    let i = builder.use_var(var_i);
    let cmp = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i, count);
    builder.ins().brif(cmp, loop_exit, &[], loop_body, &[]);

    builder.switch_to_block(loop_body);
    let i = builder.use_var(var_i);

    let tuple_idx = if use_recent {
        // recent_ptr[i] is a usize (pointer-sized); load it directly.
        let byte_off = builder.ins().imul_imm(i, std::mem::size_of::<usize>() as i64);
        let elem_addr = builder.ins().iadd(cached_recent_ptr.unwrap(), byte_off);
        builder.ins().load(ptr_type, MemFlags::trusted(), elem_addr, 0)
    } else {
        i
    };

    // Use cached pointer for non-recursive rules; re-fetch for recursive ones.
    let packed_buf = if let Some(buf) = cached_packed_buf {
        buf
    } else {
        let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
        builder.inst_results(call)[0]
    };
    let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

    let continue_block = builder.create_block();
    for (col, arg) in clause.args.iter().enumerate() {
        match arg {
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, *n as i64);
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                let actual = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                let eq = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            _ => {}
        }
    }

    for &(col, var_id) in &clause.fresh_cols {
        let val = load_packed_col(builder, tuple_ptr, col);
        store_binding(builder, bindings, var_id, val);
    }

    gen_clauses_v3(
        builder, clauses, clause_offset + 1, recent_clause_idx,
        rels, bindings, head_rels, lookup_handles, func_refs, heads, ptr_type, next_var, conditions,
    );

    builder.ins().jump(continue_block, &[]);
    builder.switch_to_block(continue_block);
    builder.seal_block(continue_block);

    let i = builder.use_var(var_i);
    let one = builder.ins().iconst(ptr_type, 1);
    let i_next = builder.ins().iadd(i, one);
    builder.def_var(var_i, i_next);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);
}

/// Inline hash-probe index scan (Stage 3 / Stage 4).
///
/// Replaces the old `call packed_lookup` with a direct open-addressed probe
/// into the `JitHashIndex` stored in `lookup_handles[clause_offset * 2 + use_recent]`.
///
/// Handle layout (`JitLookupHandle`, 24 bytes):
/// - offset  0: entries_ptr  *const JitIndexEntry
/// - offset  8: values_ptr   *const u32
/// - offset 16: mask         u32   (capacity - 1)
///
/// `JitIndexEntry` layout (16 bytes):
/// - offset  0: key    u32
/// - offset  4: head   u32  (index of first node in linked-list chain; SENTINEL = empty)
///
/// Values layout (linked-list nodes, stride 8 bytes):
/// - values[node * 2 + 0] = tuple_idx  (u32)
/// - values[node * 2 + 1] = next node  (u32; SENTINEL = end of chain)
#[allow(clippy::too_many_arguments)]
fn gen_index_scan_v3(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    rel_ptr: CValue,
    arity: usize,
    use_recent: bool,
    clauses: &[(usize, &CClause)],
    clause_offset: usize,
    recent_clause_idx: Option<usize>,
    rels: CValue,
    bindings: CValue,
    head_rels: CValue,
    lookup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    is_recursive: bool,
) {
    let primary_col = clause.bound_cols[0];
    let key_i32 = load_bound_val(builder, clause, primary_col, bindings);

    // ─── Load handle (compile-time constant byte offset) ────────────────────
    // handle index = clause_offset * 2 + use_recent
    let handle_byte_offset = (clause_offset * 2 + use_recent as usize) * 24;
    let handle_ptr = if handle_byte_offset == 0 {
        lookup_handles
    } else {
        let off = builder.ins().iconst(ptr_type, handle_byte_offset as i64);
        builder.ins().iadd(lookup_handles, off)
    };

    let entries_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), handle_ptr, 0i32);
    let values_ptr  = builder.ins().load(ptr_type, MemFlags::trusted(), handle_ptr, 8i32);
    let mask_i32    = builder.ins().load(I32,      MemFlags::trusted(), handle_ptr, 16i32);
    let mask        = builder.ins().uextend(ptr_type, mask_i32);

    // ─── Hash probe ──────────────────────────────────────────────────────────
    // slot = (key * KNUTH_32) & mask
    let key_ext  = builder.ins().uextend(ptr_type, key_i32);
    let golden   = builder.ins().iconst(ptr_type, 2_654_435_761_i64);
    let hash_ext = builder.ins().imul(key_ext, golden);
    // Mask to 32 bits then extend back (mirrors the Rust impl using u32 arithmetic)
    let hash32     = builder.ins().ireduce(I32, hash_ext);
    let hash32_ext = builder.ins().uextend(ptr_type, hash32);
    let init_slot  = builder.ins().band(hash32_ext, mask);

    // Use Cranelift variable for mutable probe slot
    let var_slot = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_slot, ptr_type);
    builder.def_var(var_slot, init_slot);

    // Probe loop with block params to carry entry_head out.
    // Block structure:
    //   probe_loop:  check empty → empty_exit; check found → probe_found; else → probe_miss
    //   probe_miss:  increment slot, back-edge to probe_loop
    //   probe_found: load head at offset 4, jump after_probe(head)
    //   empty_exit:  check overflow slot; select head or SENTINEL, jump after_probe(head)
    //   after_probe(entry_head: I32):  inner linked-list value loop

    let probe_loop       = builder.create_block();
    let probe_check_found= builder.create_block();
    let probe_found      = builder.create_block();
    let probe_miss       = builder.create_block();
    let empty_exit       = builder.create_block();
    let after_probe      = builder.create_block();

    // after_probe receives entry_head: I32 (head of linked-list chain, SENTINEL = empty)
    builder.append_block_param(after_probe, I32);

    // Cache packed_data_ptr before entering the probe/value loops.
    // Non-recursive rules: pointer is stable (no head insert can reallocate it).
    // Recursive rules: must re-fetch inside the value loop after each potential insert.
    let cached_packed_buf = if !is_recursive {
        let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
        Some(builder.inst_results(call)[0])
    } else {
        None
    };

    builder.ins().jump(probe_loop, &[]);

    // ─── probe_loop ──────────────────────────────────────────────────────────
    builder.switch_to_block(probe_loop);
    // sealed after probe_miss back-edge

    let slot = builder.use_var(var_slot);
    // entry_ptr = entries_ptr + slot * 16  (sizeof JitIndexEntry = 16)
    let entry_byte_off = builder.ins().imul_imm(slot, 16_i64);
    let entry_ptr_probe = builder.ins().iadd(entries_ptr, entry_byte_off);
    let entry_key_probe = builder.ins().load(I32, MemFlags::trusted(), entry_ptr_probe, 0i32);

    let empty_sentinel = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
    let is_empty = builder.ins().icmp(IntCC::Equal, entry_key_probe, empty_sentinel);
    // brif: empty → empty_exit, else → probe_check_found
    builder.ins().brif(is_empty, empty_exit, &[], probe_check_found, &[]);

    // ─── probe_check_found ───────────────────────────────────────────────────
    builder.switch_to_block(probe_check_found);
    builder.seal_block(probe_check_found);

    // entry_key_probe dominates here (SSA: it was computed in probe_loop which
    // is the only predecessor of probe_check_found).
    let is_found = builder.ins().icmp(IntCC::Equal, entry_key_probe, key_i32);
    builder.ins().brif(is_found, probe_found, &[], probe_miss, &[]);

    // ─── probe_miss ──────────────────────────────────────────────────────────
    builder.switch_to_block(probe_miss);
    builder.seal_block(probe_miss);

    let slot_miss = builder.use_var(var_slot);
    let one_ptr   = builder.ins().iconst(ptr_type, 1);
    let slot_next = builder.ins().iadd(slot_miss, one_ptr);
    let slot_wrap = builder.ins().band(slot_next, mask);
    builder.def_var(var_slot, slot_wrap);
    builder.ins().jump(probe_loop, &[]);
    // Now seal probe_loop — both predecessors (initial jump + probe_miss back-edge) known.
    builder.seal_block(probe_loop);

    // ─── probe_found ─────────────────────────────────────────────────────────
    builder.switch_to_block(probe_found);
    builder.seal_block(probe_found);

    // Re-compute entry_ptr from the final slot value (same slot as probe_loop above).
    // The value of `slot` from probe_loop dominates probe_found through probe_check_found.
    // We need to re-read it via the variable because it was defined before the loop.
    let slot_found     = builder.use_var(var_slot);
    let ebo_found      = builder.ins().imul_imm(slot_found, 16_i64);
    let entry_ptr_found= builder.ins().iadd(entries_ptr, ebo_found);
    // head is at offset 4 in JitIndexEntry
    let found_head     = builder.ins().load(I32, MemFlags::trusted(), entry_ptr_found, 4i32);
    builder.ins().jump(after_probe, &[found_head]);

    // ─── empty_exit ──────────────────────────────────────────────────────────
    // Before giving up, check the overflow slot at entries[mask + 1] which
    // holds any entry for key == EMPTY_KEY (0xFFFFFFFF, i.e. i32 value -1).
    builder.switch_to_block(empty_exit);
    builder.seal_block(empty_exit);

    // overflow slot index = mask + 1
    let mask_plus_one = builder.ins().iadd_imm(mask, 1);
    let ovf_byte_off  = builder.ins().imul_imm(mask_plus_one, 16_i64);
    let ovf_entry_ptr = builder.ins().iadd(entries_ptr, ovf_byte_off);
    let ovf_key       = builder.ins().load(I32, MemFlags::trusted(), ovf_entry_ptr, 0i32);
    let ovf_match     = builder.ins().icmp(IntCC::Equal, ovf_key, key_i32);
    // Load head from overflow slot (at offset 4)
    let ovf_head      = builder.ins().load(I32, MemFlags::trusted(), ovf_entry_ptr, 4i32);
    let sentinel_i32  = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
    let sel_head      = builder.ins().select(ovf_match, ovf_head, sentinel_i32);
    builder.ins().jump(after_probe, &[sel_head]);

    // ─── after_probe(entry_head: I32) ────────────────────────────────────────
    builder.switch_to_block(after_probe);
    builder.seal_block(after_probe);

    let entry_head = builder.block_params(after_probe)[0]; // I32, SENTINEL = no entries

    // ─── Inner value loop (linked-list traversal) ─────────────────────────────
    // ptr starts at entry_head, advance via values_ptr[ptr*2+1] until SENTINEL.
    let loop_header = builder.create_block();
    let loop_body   = builder.create_block();
    let loop_exit   = builder.create_block();

    let var_ptr = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_ptr, I32);
    builder.def_var(var_ptr, entry_head);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_header);
    let ptr = builder.use_var(var_ptr);
    let sentinel = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
    let is_done = builder.ins().icmp(IntCC::Equal, ptr, sentinel);
    builder.ins().brif(is_done, loop_exit, &[], loop_body, &[]);

    builder.switch_to_block(loop_body);
    let ptr = builder.use_var(var_ptr);

    // values layout: stride 8 bytes per node (two u32s)
    // values_ptr[ptr*8 + 0] = tuple_idx (u32)
    // values_ptr[ptr*8 + 4] = next_ptr  (u32)
    let ptr_ext    = builder.ins().uextend(ptr_type, ptr);
    let byte_off   = builder.ins().imul_imm(ptr_ext, 8_i64);
    let node_addr  = builder.ins().iadd(values_ptr, byte_off);
    let tuple_idx_u32 = builder.ins().load(I32, MemFlags::trusted(), node_addr, 0i32);
    let next_ptr      = builder.ins().load(I32, MemFlags::trusted(), node_addr, 4i32);
    let tuple_idx     = builder.ins().uextend(ptr_type, tuple_idx_u32);

    // Store next_ptr in a variable so continue_block can advance the chain.
    let var_next_ptr = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_next_ptr, I32);
    builder.def_var(var_next_ptr, next_ptr);

    // Use pre-fetched pointer for non-recursive rules; re-fetch for recursive ones.
    let packed_buf = if let Some(buf) = cached_packed_buf {
        buf
    } else {
        let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
        builder.inst_results(call)[0]
    };
    let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

    let continue_block = builder.create_block();
    let mut inner_blocks_to_seal = Vec::new();

    for (col, arg) in clause.args.iter().enumerate() {
        match arg {
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                let actual   = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, *n as i64);
                let eq       = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                let actual   = load_packed_col(builder, tuple_ptr, col);
                let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                let eq       = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            _ => {}
        }
    }

    for &col in &clause.bound_cols[1..] {
        let actual   = load_packed_col(builder, tuple_ptr, col);
        let expected = load_bound_val(builder, clause, col, bindings);
        let eq       = builder.ins().icmp(IntCC::Equal, actual, expected);
        let pass_block = builder.create_block();
        builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
        inner_blocks_to_seal.push(pass_block);
        builder.switch_to_block(pass_block);
    }

    for &(col, var_id) in &clause.fresh_cols {
        let val = load_packed_col(builder, tuple_ptr, col);
        store_binding(builder, bindings, var_id, val);
    }

    gen_clauses_v3(
        builder, clauses, clause_offset + 1, recent_clause_idx,
        rels, bindings, head_rels, lookup_handles, func_refs, heads, ptr_type, next_var, conditions,
    );

    builder.ins().jump(continue_block, &[]);

    builder.switch_to_block(continue_block);
    for blk in inner_blocks_to_seal {
        builder.seal_block(blk);
    }
    builder.seal_block(continue_block);

    // Advance to next node in chain.
    let next_ptr_val = builder.use_var(var_next_ptr);
    builder.def_var(var_ptr, next_ptr_val);
    builder.ins().jump(loop_header, &[]);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_header);
    builder.seal_block(loop_body);
    builder.seal_block(loop_exit);
}

/// Emit head tuples directly into head relations via `packed_try_insert`.
fn gen_emit_heads_v3(
    builder: &mut FunctionBuilder,
    heads: &[CHeadClause],
    head_rels: CValue,
    bindings: CValue,
    func_refs: &FuncRefsV3,
    ptr_type: cranelift_codegen::ir::Type,
) {
    for (head_idx, head) in heads.iter().enumerate() {
        let arity = head.args.len();

        // Load head_rels[head_idx] -> *mut PackedStorage
        let offset = builder.ins().iconst(ptr_type, (head_idx as i64) * 8);
        let rel_addr = builder.ins().iadd(head_rels, offset);
        let head_rel = builder.ins().load(ptr_type, MemFlags::trusted(), rel_addr, 0);

        if arity == 0 {
            let null = builder.ins().iconst(ptr_type, 0);
            let arity_val = builder.ins().iconst(I32, 0);
            builder.ins().call(func_refs.packed_try_insert, &[head_rel, null, arity_val]);
            continue;
        }

        let slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (arity * 4) as u32,
            2,
        ));

        for (col, arg) in head.args.iter().enumerate() {
            let var_id = match arg {
                CExpr::Var(id) => *id,
                _ => panic!("packed JIT v3: non-Var head arg"),
            };
            let val = load_binding(builder, bindings, var_id);
            let slot_addr = builder.ins().stack_addr(ptr_type, slot, (col * 4) as i32);
            builder.ins().store(MemFlags::trusted(), val, slot_addr, 0);
        }

        let slot_addr = builder.ins().stack_addr(ptr_type, slot, 0);
        let arity_val = builder.ins().iconst(I32, arity as i64);
        builder.ins().call(func_refs.packed_try_insert, &[head_rel, slot_addr, arity_val]);
    }
}
