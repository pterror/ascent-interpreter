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

use crate::compiled::{CBodyItem, CBinOp, CClause, CClauseArg, CCondition, CExpr, CHeadClause, CRule, CUnOp};
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

    // Emit per-clause conditions (merged from `if expr` conditions by the optimizer)
    for cond in &clause.conditions {
        if let CCondition::If(expr) = cond {
            let cond_val = compile_packed_expr(builder, expr, fields.bindings)
                .expect("clause condition: supported by eligibility check");
            let pass_block = builder.create_block();
            builder.ins().brif(cond_val, pass_block, &[], continue_block, &[]);
            builder.switch_to_block(pass_block);
            builder.seal_block(pass_block);
        }
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

    // Emit per-clause conditions (merged from `if expr` conditions by the optimizer)
    for cond in &clause.conditions {
        if let CCondition::If(expr) = cond {
            let cond_val = compile_packed_expr(builder, expr, fields.bindings)
                .expect("clause condition: supported by eligibility check");
            let pass_block = builder.create_block();
            builder.ins().brif(cond_val, pass_block, &[], continue_block, &[]);
            inner_blocks_to_seal.push(pass_block);
            builder.switch_to_block(pass_block);
        }
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

        // Copy each head arg from bindings into the slot (expressions supported by eligibility)
        for (col, arg) in head.args.iter().enumerate() {
            let val = compile_packed_expr(builder, arg, fields.bindings)
                .expect("head expr: supported by eligibility check");
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
        CClauseArg::Expr(expr) => compile_packed_expr(builder, expr, bindings_ptr)
            .expect("packed JIT: bound clause expr should be caught by eligibility check"),
    }
}

// ─── Variable-based binding access (Stage 3/4) ───────────────────────

/// Read binding variable `var_id` from a Cranelift Variable slice (Stage 3/4).
#[inline]
fn use_binding(builder: &mut FunctionBuilder, vars: &[Variable], var_id: u32) -> CValue {
    builder.use_var(vars[var_id as usize])
}

/// Write `val` into binding Variable `var_id` (Stage 3/4).
#[inline]
fn def_binding(builder: &mut FunctionBuilder, vars: &[Variable], var_id: u32, val: CValue) {
    builder.def_var(vars[var_id as usize], val);
}

/// Load the I32 key value for a bound column using Variables (Stage 3/4).
fn load_bound_val_vars(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    col: usize,
    vars: &[Variable],
) -> CValue {
    match &clause.args[col] {
        CClauseArg::Var(id) => use_binding(builder, vars, *id),
        CClauseArg::Expr(expr) => compile_packed_expr_vars(builder, expr, vars)
            .expect("packed JIT: bound clause expr should be caught by eligibility check"),
    }
}

/// Compile a CExpr using Cranelift Variables for bindings (Stage 3/4).
fn compile_packed_expr_vars(
    builder: &mut FunctionBuilder,
    expr: &CExpr,
    vars: &[Variable],
) -> Result<CValue, String> {
    match expr {
        CExpr::Var(id) => Ok(use_binding(builder, vars, *id)),
        CExpr::Literal(Value::I32(n)) => Ok(builder.ins().iconst(I32, *n as i64)),
        CExpr::Literal(Value::Bool(b)) => Ok(builder.ins().iconst(I32, if *b { 1 } else { 0 })),
        CExpr::VarBinVar(op, a, b) => {
            let av = use_binding(builder, vars, *a);
            let bv = use_binding(builder, vars, *b);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::VarBinLit(op, a, Value::I32(n)) => {
            let av = use_binding(builder, vars, *a);
            let bv = builder.ins().iconst(I32, *n as i64);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::LitBinVar(op, Value::I32(n), b) => {
            let av = builder.ins().iconst(I32, *n as i64);
            let bv = use_binding(builder, vars, *b);
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::Binary(op, a, b) => {
            let av = compile_packed_expr_vars(builder, a, vars)?;
            let bv = compile_packed_expr_vars(builder, b, vars)?;
            compile_packed_binop(builder, *op, av, bv)
        }
        CExpr::Unary(CUnOp::Not, inner) => {
            let v = compile_packed_expr_vars(builder, inner, vars)?;
            let one = builder.ins().iconst(I32, 1);
            Ok(builder.ins().bxor(v, one))
        }
        CExpr::Unary(CUnOp::Neg, inner) => {
            let v = compile_packed_expr_vars(builder, inner, vars)?;
            Ok(builder.ins().ineg(v))
        }
        _ => Err(format!("packed JIT: unsupported expr in condition: {expr:?}")),
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
/// instead of being pushed onto a results buffer.  Bindings are held in
/// Cranelift Variables (register-allocated) rather than a heap u32 array.
#[allow(clippy::too_many_arguments)]
pub(crate) fn codegen_packed_rule_body_v3(
    rule: &CRule,
    recent_clause_idx: Option<usize>,
    func_id: cranelift_module::FuncId,
    module: &mut impl cranelift_module::Module,
    builder_ctx: &mut cranelift_frontend::FunctionBuilderContext,
    ctx: &mut cranelift_codegen::Context,
    helpers: &PackedJitHelperIds,
    var_count: usize,
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

    // Declare one Cranelift Variable per VarId (0..var_count) for register-allocated bindings.
    // Initialize to 0 so all paths are SSA-valid even before the first write.
    let zero_i32 = builder.ins().iconst(I32, 0);
    let vars: Vec<Variable> = (0..var_count)
        .map(|i| {
            let v = Variable::new(i);
            builder.declare_var(v, I32);
            builder.def_var(v, zero_i32);
            v
        })
        .collect();

    builder.seal_block(entry_block);

    let ctx_ptr = builder.block_params(entry_block)[0];

    let flags = MemFlags::trusted();
    // Load context fields (PackedJitContextV3 layout — no bindings ptr, offsets shifted)
    let rels = builder.ins().load(ptr_type, flags, ctx_ptr, 0_i32);
    let head_rels = builder.ins().load(ptr_type, flags, ctx_ptr, 16_i32);
    let lookup_handles = builder.ins().load(ptr_type, flags, ctx_ptr, 24_i32);
    let head_dedup_handles = builder.ins().load(ptr_type, flags, ctx_ptr, 32_i32);

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

    // Loop-counter Variables start after binding Variables to avoid ID conflicts.
    let mut next_var = var_count;

    // Hoist packed_data_ptr for non-recursive clause relations out of all loop nesting.
    // For EDB (non-recursive) relations the packed buffer pointer is stable across the
    // entire rule evaluation; compute it once here instead of once per outer-loop iteration.
    let precomputed_packed_bufs: Vec<Option<CValue>> = clauses
        .iter()
        .enumerate()
        .map(|(clause_offset, (_, clause))| {
            let is_recursive = rule.heads.iter().any(|h| h.relation == clause.relation);
            if !is_recursive {
                let rel_off = builder.ins().iconst(ptr_type, (clause_offset as i64) * 8);
                let rel_addr = builder.ins().iadd(rels, rel_off);
                let rel_p = builder.ins().load(ptr_type, MemFlags::trusted(), rel_addr, 0);
                let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_p]);
                Some(builder.inst_results(call)[0])
            } else {
                None
            }
        })
        .collect();

    gen_clauses_v3(
        &mut builder,
        &clauses,
        0,
        recent_clause_idx,
        rels,
        &vars,
        head_rels,
        lookup_handles,
        head_dedup_handles,
        &func_refs,
        &rule.heads,
        ptr_type,
        &mut next_var,
        &conditions,
        &precomputed_packed_bufs,
        None, // Stage 3 single-rule path: no tuple_sets_buf available
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
    vars: &[Variable],
    head_rels: CValue,
    lookup_handles: CValue,
    head_dedup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    precomputed_packed_bufs: &[Option<CValue>],
    tuple_sets_ptr: Option<CValue>,
) {
    if clause_offset >= clauses.len() {
        if !conditions.is_empty() {
            let done_block = builder.create_block();
            for &expr in conditions {
                let cond_val = compile_packed_expr_vars(builder, expr, vars)
                    .expect("compile_packed_expr_vars: should succeed for eligible exprs");
                let pass_block = builder.create_block();
                builder.ins().brif(cond_val, pass_block, &[], done_block, &[]);
                builder.switch_to_block(pass_block);
                builder.seal_block(pass_block);
            }
            gen_emit_heads_v3(builder, heads, head_rels, head_dedup_handles, vars, func_refs, ptr_type, next_var);
            builder.ins().jump(done_block, &[]);
            builder.switch_to_block(done_block);
            builder.seal_block(done_block);
        } else {
            gen_emit_heads_v3(builder, heads, head_rels, head_dedup_handles, vars, func_refs, ptr_type, next_var);
        }
        return;
    }

    let (body_idx, clause) = clauses[clause_offset];
    let use_recent = recent_clause_idx == Some(body_idx);
    let rel_seq_idx = clause_offset;

    // O(1) existence check for fully-bound inner clauses using JitTupleSet.
    // Applies when: not clause 0, no fresh cols (all bound), and Stage 4 context
    // provides the tuple_sets_ptr (Stage 3 single-rule path passes None).
    if clause_offset > 0 && clause.fresh_cols.is_empty() && let Some(ts_ptr) = tuple_sets_ptr {
        gen_tuple_set_probe_v3(
            builder, clause, clause_offset, use_recent, ts_ptr,
            clauses, recent_clause_idx, rels, vars, head_rels,
            lookup_handles, head_dedup_handles, func_refs, heads, ptr_type, next_var,
            conditions, precomputed_packed_bufs,
        );
        return;
    }

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
            clauses, clause_offset, recent_clause_idx, rels, vars, head_rels,
            lookup_handles, head_dedup_handles, func_refs, heads, ptr_type, next_var, conditions,
            is_recursive, precomputed_packed_bufs, tuple_sets_ptr,
        );
    } else {
        gen_index_scan_v3(
            builder, clause, rel_ptr, arity, use_recent,
            clauses, clause_offset, recent_clause_idx, rels, vars, head_rels,
            lookup_handles, head_dedup_handles, func_refs, heads, ptr_type, next_var, conditions,
            is_recursive, precomputed_packed_bufs, tuple_sets_ptr,
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
    vars: &[Variable],
    head_rels: CValue,
    lookup_handles: CValue,
    head_dedup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    _is_recursive: bool,
    precomputed_packed_bufs: &[Option<CValue>],
    tuple_sets_ptr: Option<CValue>,
) {
    let call = builder.ins().call(func_refs.packed_count, &[rel_ptr, use_recent_val]);
    let count = builder.inst_results(call)[0];

    // For non-recursive rules the packed_data buffer cannot be reallocated during
    // the scan, so we use the pointer hoisted before all loop nesting.
    // For recursive rules (head writes to this clause's relation) we must re-fetch
    // on every iteration because packed_try_insert may grow the Vec.
    let cached_packed_buf = precomputed_packed_bufs.get(clause_offset).copied().flatten();

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
        def_binding(builder, vars, var_id, val);
    }

    // Emit per-clause conditions (merged from `if expr` conditions by the optimizer)
    for cond in &clause.conditions {
        if let CCondition::If(expr) = cond {
            let cond_val = compile_packed_expr_vars(builder, expr, vars)
                .expect("clause condition: supported by eligibility check");
            let pass_block = builder.create_block();
            builder.ins().brif(cond_val, pass_block, &[], continue_block, &[]);
            builder.switch_to_block(pass_block);
            builder.seal_block(pass_block);
        }
    }

    gen_clauses_v3(
        builder, clauses, clause_offset + 1, recent_clause_idx,
        rels, vars, head_rels, lookup_handles, head_dedup_handles,
        func_refs, heads, ptr_type, next_var, conditions, precomputed_packed_bufs,
        tuple_sets_ptr,
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
/// All relations use contiguous mode: entry.head = start index in values array,
/// entry.count = number of u32 values at values[head..head+count].
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
    vars: &[Variable],
    head_rels: CValue,
    lookup_handles: CValue,
    head_dedup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    is_recursive: bool,
    precomputed_packed_bufs: &[Option<CValue>],
    tuple_sets_ptr: Option<CValue>,
) {
    let primary_col = clause.bound_cols[0];
    let key_i32 = load_bound_val_vars(builder, clause, primary_col, vars);

    // ─── Load handle (compile-time constant byte offset) ────────────────────
    // handle index = clause_offset * 2 + use_recent; handle stride = 24 bytes
    let handle_byte_offset = (clause_offset * 2 + use_recent as usize) * 24;
    let handle_ptr = if handle_byte_offset == 0 {
        lookup_handles
    } else {
        let off = builder.ins().iconst(ptr_type, handle_byte_offset as i64);
        builder.ins().iadd(lookup_handles, off)
    };

    let entries_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), handle_ptr,  0i32);
    let values_ptr  = builder.ins().load(ptr_type, MemFlags::trusted(), handle_ptr,  8i32);
    let mask_i32    = builder.ins().load(I32,      MemFlags::trusted(), handle_ptr, 16i32);
    let mask        = builder.ins().uextend(ptr_type, mask_i32);

    // ─── Hash probe ──────────────────────────────────────────────────────────
    let key_ext  = builder.ins().uextend(ptr_type, key_i32);
    let golden   = builder.ins().iconst(ptr_type, 2_654_435_761_i64);
    let hash_ext = builder.ins().imul(key_ext, golden);
    let hash32     = builder.ins().ireduce(I32, hash_ext);
    let hash32_ext = builder.ins().uextend(ptr_type, hash32);
    let init_slot  = builder.ins().band(hash32_ext, mask);

    let var_slot = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_slot, ptr_type);
    builder.def_var(var_slot, init_slot);

    // Block structure:
    //   probe_loop → empty_exit | probe_check_found
    //   probe_check_found → probe_found | probe_miss
    //   probe_miss → probe_loop (back-edge)
    //   is_recursive (linked-list):   probe_found → after_probe(head)
    //                                 empty_exit  → after_probe(head)
    //                                 after_probe(head: I32) → linked-list value loop
    //   !is_recursive (contiguous):   probe_found → after_probe(head, count)
    //                                 empty_exit  → after_probe(head, count)
    //                                 after_probe(head: I32, count: I32) → sequential scan
    let probe_loop        = builder.create_block();
    let probe_check_found = builder.create_block();
    let probe_found       = builder.create_block();
    let probe_miss        = builder.create_block();
    let empty_exit        = builder.create_block();
    let after_probe       = builder.create_block();

    // after_probe params: linked-list → (head: I32); contiguous → (head: I32, count: I32).
    builder.append_block_param(after_probe, I32);
    if !is_recursive {
        builder.append_block_param(after_probe, I32);
    }

    // Use the packed_data_ptr hoisted before all loop nesting (stable for non-recursive rules).
    let cached_packed_buf = precomputed_packed_bufs.get(clause_offset).copied().flatten();

    builder.ins().jump(probe_loop, &[]);

    // ─── probe_loop ──────────────────────────────────────────────────────────
    builder.switch_to_block(probe_loop);
    let slot = builder.use_var(var_slot);
    // entry_ptr = entries_ptr + slot * 16  (sizeof JitIndexEntry = 16)
    let entry_byte_off  = builder.ins().imul_imm(slot, 16_i64);
    let entry_ptr_probe = builder.ins().iadd(entries_ptr, entry_byte_off);
    let entry_key_probe = builder.ins().load(I32, MemFlags::trusted(), entry_ptr_probe, 0i32);

    let empty_sentinel = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
    let is_empty = builder.ins().icmp(IntCC::Equal, entry_key_probe, empty_sentinel);
    builder.ins().brif(is_empty, empty_exit, &[], probe_check_found, &[]);

    // ─── probe_check_found ───────────────────────────────────────────────────
    builder.switch_to_block(probe_check_found);
    builder.seal_block(probe_check_found);
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
    builder.seal_block(probe_loop);

    // ─── probe_found ─────────────────────────────────────────────────────────
    builder.switch_to_block(probe_found);
    builder.seal_block(probe_found);
    let slot_found      = builder.use_var(var_slot);
    let ebo_found       = builder.ins().imul_imm(slot_found, 16_i64);
    let entry_ptr_found = builder.ins().iadd(entries_ptr, ebo_found);
    let found_head  = builder.ins().load(I32, MemFlags::trusted(), entry_ptr_found, 4i32);
    if is_recursive {
        builder.ins().jump(after_probe, &[found_head]);
    } else {
        let found_count = builder.ins().load(I32, MemFlags::trusted(), entry_ptr_found, 8i32);
        builder.ins().jump(after_probe, &[found_head, found_count]);
    }

    // ─── empty_exit ──────────────────────────────────────────────────────────
    builder.switch_to_block(empty_exit);
    builder.seal_block(empty_exit);
    if is_recursive {
        // Linked-list index: no overflow slot. Empty probe slot means key not found.
        let sentinel_i32 = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
        builder.ins().jump(after_probe, &[sentinel_i32]);
    } else {
        // Contiguous EDB index: check overflow slot at entries[mask+1] for keys that
        // hash to EMPTY_KEY. build_contiguous allocates one extra entry past cap.
        let mask_plus_one = builder.ins().iadd_imm(mask, 1);
        let ovf_byte_off  = builder.ins().imul_imm(mask_plus_one, 16_i64);
        let ovf_entry_ptr = builder.ins().iadd(entries_ptr, ovf_byte_off);
        let ovf_key       = builder.ins().load(I32, MemFlags::trusted(), ovf_entry_ptr, 0i32);
        let ovf_match     = builder.ins().icmp(IntCC::Equal, ovf_key, key_i32);
        let ovf_head      = builder.ins().load(I32, MemFlags::trusted(), ovf_entry_ptr, 4i32);
        let sentinel_i32  = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);
        let sel_head      = builder.ins().select(ovf_match, ovf_head, sentinel_i32);
        let ovf_count     = builder.ins().load(I32, MemFlags::trusted(), ovf_entry_ptr, 8i32);
        let zero_i32      = builder.ins().iconst(I32, 0);
        let sel_count     = builder.ins().select(ovf_match, ovf_count, zero_i32);
        builder.ins().jump(after_probe, &[sel_head, sel_count]);
    }

    // ─── after_probe ─────────────────────────────────────────────────────────
    builder.switch_to_block(after_probe);
    builder.seal_block(after_probe);
    let entry_head = builder.block_params(after_probe)[0];

    if is_recursive {
        // ─── Linked-list inner loop ───────────────────────────────────────────
        // Node layout: stride 8 bytes
        //   arity == 2: node[0] = col_value (free col), node[4] = next_ptr
        //   arity >  2: node[0] = tuple_idx,            node[4] = next_ptr
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
        let ptr_ext   = builder.ins().uextend(ptr_type, ptr);
        let byte_off  = builder.ins().imul_imm(ptr_ext, 8_i64);
        let node_addr = builder.ins().iadd(values_ptr, byte_off);
        let v0       = builder.ins().load(I32, MemFlags::trusted(), node_addr, 0i32);
        let next_ptr = builder.ins().load(I32, MemFlags::trusted(), node_addr, 4i32);

        let var_next_ptr = Variable::new(*next_var);
        *next_var += 1;
        builder.declare_var(var_next_ptr, I32);
        builder.def_var(var_next_ptr, next_ptr);

        let continue_block = builder.create_block();
        let mut inner_blocks_to_seal = Vec::new();

        if arity == 2 {
            // Col-value linked-list: v0 is the free column's value directly.
            // No packed_data_ptr call; mirrors the contiguous is_col_value path.
            let col_val  = v0;
            let free_col = 1 - primary_col;

            match &clause.args[free_col] {
                CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                    let expected = builder.ins().iconst(I32, *n as i64);
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                }
                CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                    let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                }
                _ => {}
            }
            for &col in &clause.bound_cols[1..] {
                if col == free_col {
                    let expected = load_bound_val_vars(builder, clause, col, vars);
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                    break;
                }
            }
            for &(col, var_id) in &clause.fresh_cols {
                if col == free_col {
                    def_binding(builder, vars, var_id, col_val);
                    break;
                }
            }
        } else {
            // Tuple-index linked-list (arity > 2): v0 is tuple_idx.
            let tuple_idx = builder.ins().uextend(ptr_type, v0);
            // Re-fetch packed_data_ptr: recursive head insert may have reallocated.
            let packed_buf = {
                let call = builder.ins().call(func_refs.packed_data_ptr, &[rel_ptr]);
                builder.inst_results(call)[0]
            };
            let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

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
                let expected = load_bound_val_vars(builder, clause, col, vars);
                let eq       = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            for &(col, var_id) in &clause.fresh_cols {
                let val = load_packed_col(builder, tuple_ptr, col);
                def_binding(builder, vars, var_id, val);
            }
        }

        for cond in &clause.conditions {
            if let CCondition::If(expr) = cond {
                let cond_val = compile_packed_expr_vars(builder, expr, vars)
                    .expect("clause condition: supported by eligibility check");
                let pass_block = builder.create_block();
                builder.ins().brif(cond_val, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
        }

        gen_clauses_v3(
            builder, clauses, clause_offset + 1, recent_clause_idx,
            rels, vars, head_rels, lookup_handles, head_dedup_handles,
            func_refs, heads, ptr_type, next_var, conditions, precomputed_packed_bufs,
            tuple_sets_ptr,
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
    } else {
        // ─── Sequential value scan loop ──────────────────────────────────────
        // values_ptr[entry_head .. entry_head+entry_count]:
        //   EDB arity-2: col-value mode — value IS free column data.
        //   All other non-recursive: tuple_idx mode.
        let entry_count = builder.block_params(after_probe)[1];
        let loop_header = builder.create_block();
        let loop_body   = builder.create_block();
        let loop_exit   = builder.create_block();

        let var_j = Variable::new(*next_var);
        *next_var += 1;
        builder.declare_var(var_j, I32);
        let zero_i32_j = builder.ins().iconst(I32, 0);
        builder.def_var(var_j, zero_i32_j);
        builder.ins().jump(loop_header, &[]);

        builder.switch_to_block(loop_header);
        let j = builder.use_var(var_j);
        let is_done = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, j, entry_count);
        builder.ins().brif(is_done, loop_exit, &[], loop_body, &[]);

        builder.switch_to_block(loop_body);
        let j = builder.use_var(var_j);
        // byte_offset = (entry_head + j) * 4
        let j_ext       = builder.ins().uextend(ptr_type, j);
        let start_ext   = builder.ins().uextend(ptr_type, entry_head);
        let idx_in_vals = builder.ins().iadd(start_ext, j_ext);
        let byte_off    = builder.ins().imul_imm(idx_in_vals, 4_i64);
        let elem_addr   = builder.ins().iadd(values_ptr, byte_off);

        let continue_block = builder.create_block();
        let mut inner_blocks_to_seal = Vec::new();

        // EDB arity-2: col-value mode (free column data stored directly).
        let is_col_value = arity == 2;
        if is_col_value {
            let col_val  = builder.ins().load(I32, MemFlags::trusted(), elem_addr, 0i32);
            let free_col = 1 - primary_col;

            match &clause.args[free_col] {
                CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                    let expected = builder.ins().iconst(I32, *n as i64);
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                }
                CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                    let expected = builder.ins().iconst(I32, if *b { 1 } else { 0 });
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                }
                _ => {}
            }
            for &col in &clause.bound_cols[1..] {
                if col == free_col {
                    let expected = load_bound_val_vars(builder, clause, col, vars);
                    let eq = builder.ins().icmp(IntCC::Equal, col_val, expected);
                    let pass_block = builder.create_block();
                    builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                    inner_blocks_to_seal.push(pass_block);
                    builder.switch_to_block(pass_block);
                    break;
                }
            }
            for &(col, var_id) in &clause.fresh_cols {
                if col == free_col {
                    def_binding(builder, vars, var_id, col_val);
                    break;
                }
            }
        } else {
            // Standard path: load tuple_idx, dereference through packed_data.
            let tuple_idx_u32 = builder.ins().load(I32, MemFlags::trusted(), elem_addr, 0i32);
            let tuple_idx     = builder.ins().uextend(ptr_type, tuple_idx_u32);

            let packed_buf = cached_packed_buf.unwrap(); // always Some for !is_recursive
            let tuple_ptr = compute_tuple_ptr(builder, packed_buf, tuple_idx, arity);

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
                let expected = load_bound_val_vars(builder, clause, col, vars);
                let eq       = builder.ins().icmp(IntCC::Equal, actual, expected);
                let pass_block = builder.create_block();
                builder.ins().brif(eq, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
            for &(col, var_id) in &clause.fresh_cols {
                let val = load_packed_col(builder, tuple_ptr, col);
                def_binding(builder, vars, var_id, val);
            }
        }

        for cond in &clause.conditions {
            if let CCondition::If(expr) = cond {
                let cond_val = compile_packed_expr_vars(builder, expr, vars)
                    .expect("clause condition: supported by eligibility check");
                let pass_block = builder.create_block();
                builder.ins().brif(cond_val, pass_block, &[], continue_block, &[]);
                inner_blocks_to_seal.push(pass_block);
                builder.switch_to_block(pass_block);
            }
        }

        gen_clauses_v3(
            builder, clauses, clause_offset + 1, recent_clause_idx,
            rels, vars, head_rels, lookup_handles, head_dedup_handles,
            func_refs, heads, ptr_type, next_var, conditions, precomputed_packed_bufs,
            tuple_sets_ptr,
        );

        builder.ins().jump(continue_block, &[]);
        builder.switch_to_block(continue_block);
        for blk in inner_blocks_to_seal {
            builder.seal_block(blk);
        }
        builder.seal_block(continue_block);

        // j++
        let j = builder.use_var(var_j);
        let one_i32 = builder.ins().iconst(I32, 1);
        let j_next  = builder.ins().iadd(j, one_i32);
        builder.def_var(var_j, j_next);
        builder.ins().jump(loop_header, &[]);

        builder.switch_to_block(loop_exit);
        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);
    }
}

/// Inline tuple-set probe for fully-bound inner clauses (Stage 4 only).
///
/// When all columns of a clause are bound (no fresh cols), we can replace the
/// O(n) column-index scan with an O(1) hash-set existence check against the
/// `JitTupleSet` built from the total relation before the fixpoint starts.
///
/// If the tuple is present → execute the rest of the rule (recursive call).
/// If the tuple is absent → fall through (not found).
///
/// Block structure:
/// ```text
/// probe_loop → not_found | probe_check_found
/// probe_check_found → field_check[0] | probe_miss
/// field_check[i] → field_check[i+1] | probe_miss
/// field_check[N-1] → body | probe_miss
/// body → (gen_clauses_v3 recursion) → not_found
/// probe_miss → probe_loop  (back-edge; seals probe_loop)
/// not_found: fall through
/// ```
///
/// `JitTupleSet` layout: slots @0, mask @8, len @16.
/// Slot layout: stride = (arity+1)*4 bytes; slot[0]=hash_tag (0=empty), slot[1..N]=fields.
/// Hash: `h = 0x9e3779b9; for w in fields: h = h*0x9e3779b9 + w; if h==0: h=1`.
#[allow(clippy::too_many_arguments)]
fn gen_tuple_set_probe_v3(
    builder: &mut FunctionBuilder,
    clause: &CClause,
    clause_offset: usize,
    use_recent: bool,
    tuple_sets_ptr: CValue,
    clauses: &[(usize, &CClause)],
    recent_clause_idx: Option<usize>,
    rels: CValue,
    vars: &[Variable],
    head_rels: CValue,
    lookup_handles: CValue,
    head_dedup_handles: CValue,
    func_refs: &FuncRefsV3,
    heads: &[CHeadClause],
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
    conditions: &[&CExpr],
    precomputed_packed_bufs: &[Option<CValue>],
) {
    let arity = clause.args.len();

    // Load *const JitTupleSet for this (clause_offset, use_recent) pair.
    // Index = clause_offset * 2 + use_recent; each entry is 8 bytes (pointer).
    let handle_byte_offset = (clause_offset * 2 + use_recent as usize) * 8;
    let ts_ptr_addr = if handle_byte_offset == 0 {
        tuple_sets_ptr
    } else {
        let off = builder.ins().iconst(ptr_type, handle_byte_offset as i64);
        builder.ins().iadd(tuple_sets_ptr, off)
    };
    let ts_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), ts_ptr_addr, 0);

    // Load JitTupleSet fields: slots @0, mask @8 (both pointer-sized).
    let slots_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), ts_ptr, 0i32);
    let mask      = builder.ins().load(ptr_type, MemFlags::trusted(), ts_ptr, 8i32);

    // Load all arg values — all columns are bound (fresh_cols is empty).
    let arg_vals: Vec<CValue> = (0..arity)
        .map(|col| load_bound_val_vars(builder, clause, col, vars))
        .collect();

    // Compute tuple hash: h = 0x9e3779b9; for each word: h = h*0x9e3779b9 + w.
    let mult = 0x9e3779b9u32 as i64;
    let mut h = builder.ins().iconst(I32, 0x9e3779b9u32 as i64);
    for &av in &arg_vals {
        h = builder.ins().imul_imm(h, mult);
        h = builder.ins().iadd(h, av);
    }
    // if h == 0: h = 1  (0 is the empty-slot sentinel)
    let zero_i32 = builder.ins().iconst(I32, 0);
    let one_i32  = builder.ins().iconst(I32, 1);
    let is_zero_h = builder.ins().icmp(IntCC::Equal, h, zero_i32);
    let hash_tag = builder.ins().select(is_zero_h, one_i32, h);

    // Initial slot = hash_tag & mask.
    let hash_ext  = builder.ins().uextend(ptr_type, hash_tag);
    let init_slot = builder.ins().band(hash_ext, mask);

    let var_slot = Variable::new(*next_var);
    *next_var += 1;
    builder.declare_var(var_slot, ptr_type);
    builder.def_var(var_slot, init_slot);

    let stride_bytes = (arity + 1) as i64 * 4;

    // Create all blocks upfront.
    let probe_loop        = builder.create_block();
    let probe_check_found = builder.create_block();
    let field_checks: Vec<cranelift_codegen::ir::Block> =
        (0..arity).map(|_| builder.create_block()).collect();
    let body        = builder.create_block();
    let probe_miss  = builder.create_block();
    let not_found   = builder.create_block();

    builder.ins().jump(probe_loop, &[]);

    // ─── probe_loop ───────────────────────────────────────────────────────────
    builder.switch_to_block(probe_loop);
    let slot = builder.use_var(var_slot);
    let slot_byte_off = builder.ins().imul_imm(slot, stride_bytes);
    let slot_ptr = builder.ins().iadd(slots_ptr, slot_byte_off);
    let tag = builder.ins().load(I32, MemFlags::trusted(), slot_ptr, 0i32);
    let is_empty = builder.ins().icmp(IntCC::Equal, tag, zero_i32);
    builder.ins().brif(is_empty, not_found, &[], probe_check_found, &[]);

    // ─── probe_check_found ────────────────────────────────────────────────────
    // probe_loop is the unique predecessor → seal immediately.
    builder.switch_to_block(probe_check_found);
    builder.seal_block(probe_check_found);
    let tag_matches = builder.ins().icmp(IntCC::Equal, tag, hash_tag);
    let after_tag_match = if arity == 0 { body } else { field_checks[0] };
    builder.ins().brif(tag_matches, after_tag_match, &[], probe_miss, &[]);

    // ─── field check chain ────────────────────────────────────────────────────
    // field_checks[i] has exactly one predecessor (the previous block → here).
    for col in 0..arity {
        builder.switch_to_block(field_checks[col]);
        builder.seal_block(field_checks[col]);
        let field_val = builder.ins().load(
            I32, MemFlags::trusted(), slot_ptr, (col as i32 + 1) * 4,
        );
        let field_match = builder.ins().icmp(IntCC::Equal, field_val, arg_vals[col]);
        let next = if col + 1 < arity { field_checks[col + 1] } else { body };
        builder.ins().brif(field_match, next, &[], probe_miss, &[]);
    }

    // ─── body: recursive gen_clauses_v3 ──────────────────────────────────────
    // Predecessors: last field_check (or probe_check_found for arity==0). One pred.
    builder.switch_to_block(body);
    builder.seal_block(body);
    gen_clauses_v3(
        builder, clauses, clause_offset + 1, recent_clause_idx,
        rels, vars, head_rels, lookup_handles, head_dedup_handles,
        func_refs, heads, ptr_type, next_var, conditions, precomputed_packed_bufs,
        Some(tuple_sets_ptr),
    );
    builder.ins().jump(not_found, &[]);

    // ─── probe_miss ───────────────────────────────────────────────────────────
    // Predecessors: probe_check_found + all field_checks. All emitted above → seal.
    builder.switch_to_block(probe_miss);
    builder.seal_block(probe_miss);
    let slot_m    = builder.use_var(var_slot);
    let one_ptr   = builder.ins().iconst(ptr_type, 1);
    let slot_next = builder.ins().iadd(slot_m, one_ptr);
    let slot_wrap = builder.ins().band(slot_next, mask);
    builder.def_var(var_slot, slot_wrap);
    builder.ins().jump(probe_loop, &[]);
    // probe_loop preds: initial jump (above) + probe_miss back-edge. Both done → seal.
    builder.seal_block(probe_loop);

    // ─── not_found ────────────────────────────────────────────────────────────
    // Predecessors: probe_loop (is_empty) + body (jump). Both emitted → seal.
    builder.switch_to_block(not_found);
    builder.seal_block(not_found);
    // Fall through — caller continues execution after this probe.
}

/// Emit head tuples with an inline JIT dedup probe before falling back to
/// `packed_try_insert`.
///
/// For each head:
/// 1. Build the candidate tuple in a stack slot.
/// 2. Load the JIT dedup handle (`JitDedupHandle`) for this head.
/// 3. If the table is empty (cap == 0, first iteration): call `packed_try_insert` directly.
/// 4. Otherwise: compute the polynomial hash inline, probe the open-addressed table.
///    - Empty slot → new tuple → call `packed_try_insert`.
///    - Occupied slot with matching hash → verify all `arity` fields against the
///      candidate tuple.  If all match: duplicate → skip entirely (zero Rust calls).
///      If any field differs (hash collision): continue probing.
///
/// The dedup table is a frozen snapshot built by `update_jit_indices()` between
/// iterations.  Tuples inserted in the *current* iteration are not present in the
/// snapshot; those fall through to `packed_try_insert` (which finds them in the
/// authoritative dedup and returns 0 = duplicate without re-inserting).
#[allow(clippy::too_many_arguments)]
fn gen_emit_heads_v3(
    builder: &mut FunctionBuilder,
    heads: &[CHeadClause],
    head_rels: CValue,
    head_dedup_handles: CValue,
    vars: &[Variable],
    func_refs: &FuncRefsV3,
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut usize,
) {
    use cranelift_codegen::ir::condcodes::IntCC;

    let sentinel_i32 = builder.ins().iconst(I32, 0xFFFF_FFFFu32 as i64);

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

        // Build candidate tuple in a stack slot (needed for packed_try_insert pointer).
        // Keep computed values in col_vals to avoid redundant stack reloads.
        let slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (arity * 4) as u32,
            2,
        ));
        let mut col_vals = Vec::with_capacity(arity);
        for (col, arg) in head.args.iter().enumerate() {
            let val = compile_packed_expr_vars(builder, arg, vars)
                .expect("head expr: supported by eligibility check");
            let slot_addr = builder.ins().stack_addr(ptr_type, slot, (col * 4) as i32);
            builder.ins().store(MemFlags::trusted(), val, slot_addr, 0);
            col_vals.push(val);
        }
        let slot_addr = builder.ins().stack_addr(ptr_type, slot, 0);
        let arity_val = builder.ins().iconst(I32, arity as i64);

        // Load dedup handle for this head: head_dedup_handles[head_idx] -> *mut JitDedupHandle
        let hdl_offset = builder.ins().iconst(ptr_type, (head_idx as i64) * 8);
        let hdl_addr = builder.ins().iadd(head_dedup_handles, hdl_offset);
        let hdl_ptr = builder.ins().load(ptr_type, MemFlags::trusted(), hdl_addr, 0);
        // JitDedupHandle: entries @ offset 0 (ptr), cap @ offset 8 (u32)
        let entries = builder.ins().load(ptr_type, MemFlags::trusted(), hdl_ptr, 0i32);
        let cap_i32 = builder.ins().load(I32, MemFlags::trusted(), hdl_ptr, 8i32);

        // If cap == 0 (empty table, first call): skip probe, go directly to insert.
        let probe_start = builder.create_block();
        let call_insert = builder.create_block();
        let after_emit = builder.create_block();

        let zero_i32 = builder.ins().iconst(I32, 0);
        let is_cap_zero = builder.ins().icmp(IntCC::Equal, cap_i32, zero_i32);
        builder.ins().brif(is_cap_zero, call_insert, &[], probe_start, &[]);

        // ── probe_start ──────────────────────────────────────────────────────────
        builder.switch_to_block(probe_start);
        builder.seal_block(probe_start);

        // Compute polynomial hash of candidate tuple (matches `jit_dedup_hash`):
        //   h = 0; for each col: h = h * 0x9e3779b9 + val; remap FFFF_FFFF → FFFF_FFFE.
        let mut hash_val = builder.ins().iconst(I32, 0);
        for &v in &col_vals {
            let mult = builder.ins().imul_imm(hash_val, 0x9e3779b9u32 as i64);
            hash_val = builder.ins().iadd(mult, v);
        }
        let remapped = builder.ins().iconst(I32, 0xFFFF_FFFEu32 as i64);
        let is_sentinel = builder.ins().icmp(IntCC::Equal, hash_val, sentinel_i32);
        hash_val = builder.ins().select(is_sentinel, remapped, hash_val);

        // cap − 1 = mask; slot = hash & mask (both as ptr_type for address arithmetic)
        let cap = builder.ins().uextend(ptr_type, cap_i32);
        let mask = builder.ins().iadd_imm(cap, -1i64);
        let hash_ext = builder.ins().uextend(ptr_type, hash_val);
        let init_slot = builder.ins().band(hash_ext, mask);

        let var_probe_slot = Variable::new(*next_var);
        *next_var += 1;
        builder.declare_var(var_probe_slot, ptr_type);
        builder.def_var(var_probe_slot, init_slot);

        // Block structure (same pattern as gen_index_scan_v3):
        //   probe_loop → probe_check_found | call_insert
        //   probe_check_found → probe_verify | probe_next
        //   probe_next → probe_loop  (back-edge; seal probe_loop after)
        //   probe_verify → per-column checks → after_emit (duplicate) | probe_next
        //   call_insert: packed_try_insert → after_emit
        let probe_loop        = builder.create_block();
        let probe_check_found = builder.create_block();
        let probe_next        = builder.create_block();
        let probe_verify      = builder.create_block();

        builder.ins().jump(probe_loop, &[]);

        // ── probe_loop ───────────────────────────────────────────────────────────
        // stride = arity + 1; each slot is stride * 4 bytes wide.
        let stride = arity as i64 + 1;
        builder.switch_to_block(probe_loop);
        // (not sealed yet — probe_next back-edge pending)
        let ps = builder.use_var(var_probe_slot);
        let byte_off = builder.ins().imul_imm(ps, stride * 4);
        let entry_ptr = builder.ins().iadd(entries, byte_off);
        let entry_hash = builder.ins().load(I32, MemFlags::trusted(), entry_ptr, 0i32);
        let is_empty_slot = builder.ins().icmp(IntCC::Equal, entry_hash, sentinel_i32);
        builder.ins().brif(is_empty_slot, call_insert, &[], probe_check_found, &[]);

        // ── probe_check_found ────────────────────────────────────────────────────
        builder.switch_to_block(probe_check_found);
        builder.seal_block(probe_check_found);
        let ps2 = builder.use_var(var_probe_slot);
        let byte_off2 = builder.ins().imul_imm(ps2, stride * 4);
        let entry_ptr2 = builder.ins().iadd(entries, byte_off2);
        let entry_hash2 = builder.ins().load(I32, MemFlags::trusted(), entry_ptr2, 0i32);
        let hash_matches = builder.ins().icmp(IntCC::Equal, entry_hash2, hash_val);
        builder.ins().brif(hash_matches, probe_verify, &[], probe_next, &[]);

        // ── probe_next ───────────────────────────────────────────────────────────
        builder.switch_to_block(probe_next);
        // (not sealed yet — probe_verify column-mismatch edges pending)
        let ps_miss = builder.use_var(var_probe_slot);
        let ps_next = builder.ins().iadd_imm(ps_miss, 1i64);
        let ps_wrap = builder.ins().band(ps_next, mask);
        builder.def_var(var_probe_slot, ps_wrap);
        builder.ins().jump(probe_loop, &[]);
        builder.seal_block(probe_loop); // both predecessors now known

        // ── probe_verify ─────────────────────────────────────────────────────────
        // Compare each stored data field against the candidate tuple field.
        // Any mismatch → probe_next (hash collision).  All match → duplicate → after_emit.
        builder.switch_to_block(probe_verify);
        builder.seal_block(probe_verify);

        let ps_v = builder.use_var(var_probe_slot);
        let byte_off_v = builder.ins().imul_imm(ps_v, stride * 4);
        let entry_ptr_v = builder.ins().iadd(entries, byte_off_v);

        let mut mismatch_predecessors: usize = 0; // count edges into probe_next from here
        for (col, &cand) in col_vals.iter().enumerate() {
            // stored field is at byte offset (1 + col) * 4 relative to entry_ptr
            let stored = builder.ins().load(
                I32, MemFlags::trusted(), entry_ptr_v, ((1 + col) * 4) as i32,
            );
            let not_eq = builder.ins().icmp(IntCC::NotEqual, stored, cand);
            let pass_block = builder.create_block();
            builder.ins().brif(not_eq, probe_next, &[], pass_block, &[]);
            builder.switch_to_block(pass_block);
            builder.seal_block(pass_block);
            mismatch_predecessors += 1;
        }
        // All fields matched: this is a duplicate — jump past the insert.
        builder.ins().jump(after_emit, &[]);
        // Now all probe_next predecessors are known: probe_check_found + arity mismatches.
        // (probe_check_found → probe_next was already sealed above with seal_block, but
        //  Cranelift's seal works on predecessor *count*, so we seal after all edges are added.)
        // Actually probe_next was NOT sealed above — we seal it now.
        let _ = mismatch_predecessors; // all edges have been emitted
        builder.seal_block(probe_next);

        // ── call_insert ──────────────────────────────────────────────────────────
        // Entered when: cap == 0 (initial jump) OR empty slot found during probe.
        // call_insert has two predecessors: the cap==0 branch and the empty-slot branch.
        builder.switch_to_block(call_insert);
        builder.seal_block(call_insert);
        builder.ins().call(func_refs.packed_try_insert, &[head_rel, slot_addr, arity_val]);
        builder.ins().jump(after_emit, &[]);

        // ── after_emit ───────────────────────────────────────────────────────────
        builder.switch_to_block(after_emit);
        builder.seal_block(after_emit);
    }
}
