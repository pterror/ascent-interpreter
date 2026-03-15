//! x86-64 JIT backend for Stage 4 stratum functions using dynasmrt.
//!
//! Supports rules with N `CClause` body items where clause 0 is a full scan
//! (no bound cols) and clauses 1..N-1 are index scans.  Returns `Err` for any
//! unsupported pattern so the caller can fall back to Cranelift.
//!
//! # ABI
//! Generated function: `unsafe extern "C" fn(*mut StratumStage4Ctx)` — System V
//! AMD64, rdi = first arg, void return.
//!
//! # Frame layout (all offsets from rbp; "push zone" = [rbp-40..rbp])
//! ```text
//!   [rbp- 8]  saved rbx
//!   [rbp-16]  saved r12
//!   [rbp-24]  saved r13
//!   [rbp-32]  saved r14
//!   [rbp-40]  saved r15
//!   --- frame allocated by `sub rsp, FRAME_SIZE` ---
//!   [rbp-48]  ctx ptr (*mut StratumStage4Ctx)          CTX_SLOT
//!   [rbp-52-id*4]  variable[id] (u32)                  var_slot(id)
//!   [rbp-52-V*4-col*4]  head-tuple scratch col         head_col(V, col)
//!   Per-level slots below the head region (base = -52 - V*4 - H*4):
//!   [base - level*8]                      level_vptr_slot(level) = values_base for level 1..MAX_DEPTH
//!   [base - MAX_DEPTH*8 - level*8]        level_node_save(level)  for level 1..MAX_DEPTH-1
//!   [base - 2*MAX_DEPTH*8 - level*8]      level_dptr_save(level)  for level 1..MAX_DEPTH-1
//! ```
//! where V = var_count, H = max_head_arity, MAX_DEPTH = max nesting depth.
//!
//! # Register use inside loop bodies
//! All callee-saved registers survive `call` instructions:
//!   r12 = outer loop count  (level 0 only)
//!   r13 = recent_ptr for outer clause  (only if use_recent0)
//!   r14 = outer loop counter i  (level 0 only)
//!   r15 = innermost active linked-list node (current; zero-extended u32)
//!   rbx = innermost active data_ptr
//!
//! When descending from level L to level L+1:
//!   - level L's r15 (next_node) is saved to level_node_save_slot(L)
//!   - level L's rbx (data_ptr)  is saved to level_dptr_save_slot(L)
//!   - level L's vptr remains in level_vptr_slot(L) (written at probe setup, not clobbered)
//! When level L+1 exhausts (sentinel), control passes to `sub_exhausted` where:
//!   - r15 is restored from level_node_save_slot(L)
//!   - rbx is restored from level_dptr_save_slot(L)
//!   - then loop back to level L's inner_hdr

use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, DynamicLabel};
use dynasmrt::x64::Assembler;

use crate::compiled::{CBinOp, CClause, CClauseArg, CCondition, CExpr, CHeadClause, CUnOp};
use crate::jit::packed_helpers::StratumStage4Fn;
use crate::value::Value;

/// A compiled stratum function produced by the asm backend.
pub struct AsmStratum {
    _buffer: dynasmrt::ExecutableBuffer,
    pub fn_ptr: StratumStage4Fn,
}

unsafe impl Send for AsmStratum {}
unsafe impl Sync for AsmStratum {}

// ─── Stack slot constants ─────────────────────────────────────────────────

const CTX_SLOT: i32 = -48;

fn var_slot(id: u32) -> i32 {
    -52 - (id as i32) * 4
}

// ─── Variable location tracking ───────────────────────────────────────────

/// Where a variable's value lives at JIT time.
///
/// Register encoding uses the x86-64 ModRM register numbering:
///   3 = rbx, 13 = r13.
/// Only callee-saved registers that are not already claimed by loop machinery
/// may appear here (r12/r14 are reserved; r15 is clobbered during inner probes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarLoc {
    /// rbp-relative stack slot (existing behaviour).
    Stack(i32),
    /// Callee-saved register (3=rbx, 13=r13).
    Reg(u8),
}

/// Decide where each variable lives for one rule/variant.
///
/// `outer_fresh_ids`: variable IDs first bound by the outer clause (clause0
///   fresh_cols), in ascending ID order.  These are "outer-stable": they are
///   written once in the outer loop body and must survive the entire inner loop.
///
/// `use_recent0`: if true, r13 is occupied by the recent_ptr for clause0 and
///   is NOT available for variable assignment.
///
/// `is_multi_clause`: if true (2+ clause rule) rbx is repurposed for the
///   innermost data_ptr and is NOT available for variable assignment.
///
/// Returns a Vec indexed by variable ID.
fn compute_var_locs(
    var_count: usize,
    outer_fresh_ids: &[u32],
    use_recent0: bool,
    is_multi_clause: bool,
) -> Vec<VarLoc> {
    // Start with everything on the stack.
    let mut locs: Vec<VarLoc> = (0..var_count as u32).map(|id| VarLoc::Stack(var_slot(id))).collect();

    // Build the list of available registers, priority-ordered.
    let mut available: Vec<u8> = Vec::new();

    if !use_recent0 {
        // r13 is free when we don't need it for the outer recent_ptr.
        available.push(13);
    }
    if !is_multi_clause {
        // For 1-clause rules, rbx is not used for data_ptr in a way
        // that overlaps with variable lifetime, so it's free.
        available.push(3);
    }
    // For multi-clause rules rbx is the innermost data_ptr so it must not
    // be assigned to variables.

    // Assign registers to outer-stable variables in ID order.
    let mut reg_iter = available.into_iter();
    for &var_id in outer_fresh_ids {
        if let Some(reg) = reg_iter.next() {
            locs[var_id as usize] = VarLoc::Reg(reg);
        } else {
            break;
        }
    }

    locs
}

/// Base of the per-level slot region (= top of head region = lowest address in head region,
/// minus any head columns).  All per-level slots are below this.
fn level_slots_base(var_count: usize, max_head_arity: usize) -> i32 {
    -52 - (var_count as i32) * 4 - (max_head_arity as i32) * 4
}

/// values_base (= values_ptr + head*4) spill slot for nesting level L (L >= 1).
/// Offset: base - L*8
fn level_vptr_slot(level: usize, var_count: usize, max_head_arity: usize) -> i32 {
    debug_assert!(level >= 1);
    level_slots_base(var_count, max_head_arity) - (level as i32) * 8
}

/// Slot to save r15 (next_node for level L) before descending into level L+1.
/// L >= 1.  Offset: base - MAX_DEPTH*8 - L*8
fn level_node_save_slot(level: usize, var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    debug_assert!(level >= 1);
    level_slots_base(var_count, max_head_arity) - (max_depth as i32) * 8 - (level as i32) * 8
}

/// Slot to save rbx (data_ptr for level L) before descending into level L+1.
/// L >= 1.  Offset: base - 2*MAX_DEPTH*8 - L*8
fn level_dptr_save_slot(level: usize, var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    debug_assert!(level >= 1);
    level_slots_base(var_count, max_head_arity) - 2 * (max_depth as i32) * 8 - (level as i32) * 8
}

/// Slot to store match count for contiguous-mode level L (L >= 1).
/// Used only when the clause at level L has a contiguous index.
/// Stores the u32 count loaded from entry.count (JitIndexEntry offset 8).
/// L >= 1.  Offset: base - 3*MAX_DEPTH*8 - L*8
fn level_count_slot(level: usize, var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    debug_assert!(level >= 1);
    level_slots_base(var_count, max_head_arity) - 3 * (max_depth as i32) * 8 - (level as i32) * 8
}

/// Compute `sub rsp, FRAME_SIZE` value.  Must be ≡ 8 (mod 16) because after
/// 5 callee-saved pushes rsp ≡ 8 (mod 16), and we need rsp ≡ 0 for calls.
fn frame_size(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    // Bytes below [rbp-40] (callee-save push zone):
    //   8      = ctx slot
    //   V*4    = variable slots
    //   H*4    = head-tuple scratch
    //   D*8    = per-level values_ptr spill (levels 1..max_depth)
    //   D*8    = per-level node save slots  (levels 1..max_depth)
    //   D*8    = per-level dptr save slots  (levels 1..max_depth)
    //   D*8    = per-level count slots      (levels 1..max_depth; contiguous mode)
    //   8      = guard / alignment slack
    let level_bytes = max_depth * 4 * 8;
    let raw = 8 + var_count * 4 + max_head_arity * 4 + level_bytes + 8;
    // Round up to next value ≡ 8 (mod 16)
    let rem = raw % 16;
    let pad = if rem <= 8 { 8 - rem } else { 24 - rem };
    (raw + pad) as i32
}

// ─── Assembly helpers ─────────────────────────────────────────────────────

/// Emit `mov rax, QWORD addr; call rax`.
macro_rules! call_abs {
    ($asm:expr, $addr:expr) => {
        dynasm!($asm; mov rax, QWORD $addr as _; call rax)
    };
}

/// Load `rule_ctxs[rule_i]` (a `*mut PackedJitContextV3`) into `rax`.
macro_rules! load_rule_ctx {
    ($asm:expr, $ri:expr) => {{
        let ri: i32 = $ri as i32;
        dynasm!($asm
            ; mov rax, [rbp + CTX_SLOT]  // ctx
            ; mov rax, [rax]              // rule_ctxs
            ; mov rax, [rax + ri * 8]    // rule_ctxs[ri]
        );
    }};
}

/// Load `rels[seq]` of the current rule into `rdi`.  Destroys rdi.
macro_rules! load_rel_rdi {
    ($asm:expr, $ri:expr, $seq:expr) => {{
        let ri: i32 = $ri as i32;
        let seq: i32 = $seq as i32;
        dynasm!($asm
            ; mov rdi, [rbp + CTX_SLOT]
            ; mov rdi, [rdi]
            ; mov rdi, [rdi + ri * 8]
            ; mov rdi, [rdi]          // rels ptr
            ; mov rdi, [rdi + seq * 8] // rels[seq]
        );
    }};
}

// ─── Variable access helpers ──────────────────────────────────────────────

/// Load variable `id` into `eax` (zero-extends to rax).
fn emit_load_var(asm: &mut Assembler, var_locs: &[VarLoc], id: u32) {
    match var_locs[id as usize] {
        VarLoc::Stack(slot) => dynasm!(asm; mov eax, [rbp + slot]),
        VarLoc::Reg(3)      => dynasm!(asm; mov eax, ebx),
        VarLoc::Reg(13)     => dynasm!(asm; mov eax, r13d),
        VarLoc::Reg(r)      => panic!("emit_load_var: unexpected reg {r}"),
    }
}

/// Load variable `id` into `ecx` (zero-extends to rcx).
fn emit_load_var_ecx(asm: &mut Assembler, var_locs: &[VarLoc], id: u32) {
    match var_locs[id as usize] {
        VarLoc::Stack(slot) => dynasm!(asm; mov ecx, [rbp + slot]),
        VarLoc::Reg(3)      => dynasm!(asm; mov ecx, ebx),
        VarLoc::Reg(13)     => dynasm!(asm; mov ecx, r13d),
        VarLoc::Reg(r)      => panic!("emit_load_var_ecx: unexpected reg {r}"),
    }
}

/// Store `edx` into variable `id`.
fn emit_store_var(asm: &mut Assembler, var_locs: &[VarLoc], id: u32) {
    match var_locs[id as usize] {
        VarLoc::Stack(slot) => dynasm!(asm; mov [rbp + slot], edx),
        VarLoc::Reg(3)      => dynasm!(asm; mov ebx, edx),
        VarLoc::Reg(13)     => dynasm!(asm; mov r13d, edx),
        VarLoc::Reg(r)      => panic!("emit_store_var: unexpected reg {r}"),
    }
}

// ─── Expression compilation ───────────────────────────────────────────────

fn check_expr(expr: &CExpr) -> Result<(), String> {
    match expr {
        CExpr::Var(_) | CExpr::Literal(Value::I32(_)) | CExpr::Literal(Value::Bool(_)) => Ok(()),
        CExpr::VarBinVar(op, _, _)
        | CExpr::VarBinLit(op, _, _)
        | CExpr::LitBinVar(op, _, _) => check_binop(*op),
        CExpr::Binary(op, a, b) => {
            check_binop(*op)?;
            check_expr(a)?;
            check_expr(b)
        }
        CExpr::Unary(CUnOp::Not | CUnOp::Neg, i) => check_expr(i),
        _ => Err(format!("asm: unsupported expr: {expr:?}")),
    }
}

fn check_binop(op: CBinOp) -> Result<(), String> {
    use CBinOp::*;
    match op {
        Add | Sub | Mul | Eq | Ne | Lt | Le | Gt | Ge | And | Or => Ok(()),
        _ => Err(format!("asm: unsupported binop: {op:?}")),
    }
}

/// Address of head-tuple column `col` on the stack.
fn head_col_slot(var_count: usize, max_head_arity: usize, col: usize) -> i32 {
    let base = -52 - (var_count as i32) * 4 - (max_head_arity as i32) * 4;
    base + (col as i32) * 4
}

/// Emit expression → eax (i32). Clobbers rcx, rdx (not across calls).
#[allow(clippy::only_used_in_recursion)]
fn emit_expr(asm: &mut Assembler, expr: &CExpr, var_locs: &[VarLoc]) -> Result<(), String> {
    match expr {
        CExpr::Var(id) => emit_load_var(asm, var_locs, *id),
        CExpr::Literal(Value::I32(n)) => dynasm!(asm; mov eax, *n),
        CExpr::Literal(Value::Bool(b)) => dynasm!(asm; mov eax, if *b { 1i32 } else { 0i32 }),
        CExpr::VarBinVar(op, a, b) => {
            emit_load_var(asm, var_locs, *a);
            emit_load_var_ecx(asm, var_locs, *b);
            emit_binop(asm, *op)?;
        }
        CExpr::VarBinLit(op, a, Value::I32(n)) => {
            emit_load_var(asm, var_locs, *a);
            dynasm!(asm; mov ecx, *n);
            emit_binop(asm, *op)?;
        }
        CExpr::LitBinVar(op, Value::I32(n), b) => {
            dynasm!(asm; mov eax, *n);
            emit_load_var_ecx(asm, var_locs, *b);
            emit_binop(asm, *op)?;
        }
        CExpr::Binary(op, a, b) => {
            emit_expr(asm, b, var_locs)?;
            dynasm!(asm; push rax);
            emit_expr(asm, a, var_locs)?;
            dynasm!(asm; pop rcx);
            emit_binop(asm, *op)?;
        }
        CExpr::Unary(CUnOp::Not, i) => {
            emit_expr(asm, i, var_locs)?;
            dynasm!(asm; xor eax, 1i8);
        }
        CExpr::Unary(CUnOp::Neg, i) => {
            emit_expr(asm, i, var_locs)?;
            dynasm!(asm; neg eax);
        }
        _ => return Err(format!("asm: unsupported expr: {expr:?}")),
    }
    Ok(())
}

/// Apply binary op: eax op ecx → eax.
fn emit_binop(asm: &mut Assembler, op: CBinOp) -> Result<(), String> {
    use CBinOp::*;
    match op {
        Add => dynasm!(asm; add eax, ecx),
        Sub => dynasm!(asm; sub eax, ecx),
        Mul => dynasm!(asm; imul eax, ecx),
        And => dynasm!(asm; and eax, ecx),
        Or  => dynasm!(asm; or eax, ecx),
        Eq  => dynasm!(asm; cmp eax, ecx; sete al; movzx eax, al),
        Ne  => dynasm!(asm; cmp eax, ecx; setne al; movzx eax, al),
        Lt  => dynasm!(asm; cmp eax, ecx; setl al; movzx eax, al),
        Le  => dynasm!(asm; cmp eax, ecx; setle al; movzx eax, al),
        Gt  => dynasm!(asm; cmp eax, ecx; setg al; movzx eax, al),
        Ge  => dynasm!(asm; cmp eax, ecx; setge al; movzx eax, al),
        _   => return Err(format!("asm: unsupported binop: {op:?}")),
    }
    Ok(())
}

// ─── Column binding ───────────────────────────────────────────────────────

/// Bind cols and check bound constraints for `clause`. `rax` = tuple_ptr.
/// Jumps to `skip` on constraint failure. Clobbers ecx, edx.
fn emit_bind_cols(
    asm: &mut Assembler,
    clause: &CClause,
    skip: DynamicLabel,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    for (col, arg) in clause.args.iter().enumerate() {
        let off = (col as i32) * 4;
        match arg {
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                dynasm!(asm; mov edx, [rax + off]; cmp edx, *n; jne =>skip);
            }
            CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
                let v = if *b { 1i32 } else { 0i32 };
                dynasm!(asm; mov edx, [rax + off]; cmp edx, v; jne =>skip);
            }
            _ => {}
        }
    }
    for &col in &clause.bound_cols {
        if let CClauseArg::Var(var_id) = &clause.args[col] {
            dynasm!(asm; mov edx, [rax + (col as i32)*4]);
            emit_load_var_ecx(asm, var_locs, *var_id);
            dynasm!(asm; cmp edx, ecx; jne =>skip);
        }
    }
    for &(col, var_id) in &clause.fresh_cols {
        dynasm!(asm; mov edx, [rax + (col as i32)*4]);
        emit_store_var(asm, var_locs, var_id);
    }
    Ok(())
}

/// Bind or check the free column value for arity-2 col-value contiguous probes.
///
/// `ecx` = the loaded col_value (= values_base[j], which is the free column's actual value).
/// `free_col` = which column index the value belongs to.
/// Jumps to `skip` if a bound-col check fails.  Clobbers eax (for bound comparison).
fn emit_bind_col_value(
    asm: &mut Assembler,
    clause: &CClause,
    free_col: usize,
    skip: DynamicLabel,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    match &clause.args[free_col] {
        CClauseArg::Var(var_id) => {
            if clause.fresh_cols.iter().any(|&(c, _)| c == free_col) {
                // Fresh var: bind from ecx
                dynasm!(asm; mov edx, ecx);
                emit_store_var(asm, var_locs, *var_id);
            } else {
                // Bound var: check ecx == var value
                emit_load_var(asm, var_locs, *var_id); // → eax
                dynasm!(asm; cmp ecx, eax; jne =>skip);
            }
        }
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            dynasm!(asm; cmp ecx, *n; jne =>skip);
        }
        CClauseArg::Expr(CExpr::Literal(Value::Bool(b))) => {
            let v = if *b { 1i32 } else { 0i32 };
            dynasm!(asm; cmp ecx, v; jne =>skip);
        }
        _ => return Err(format!(
            "asm: col_value: unsupported free col arg: {:?}",
            &clause.args[free_col]
        )),
    }
    Ok(())
}

// ─── Head emission ────────────────────────────────────────────────────────

/// Build and insert all heads: inline dedup probe + `packed_try_insert`.
fn emit_heads(
    asm: &mut Assembler,
    heads: &[CHeadClause],
    rule_i: usize,
    var_count: usize,
    max_head_arity: usize,
    pti: usize,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    for (hi, head) in heads.iter().enumerate() {
        let arity = head.args.len();
        let hoff: i32 = (hi as i32) * 8;

        for (col, arg) in head.args.iter().enumerate() {
            emit_expr(asm, arg, var_locs)?;
            dynasm!(asm; mov [rbp + head_col_slot(var_count, max_head_arity, col)], eax);
        }

        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 16i8]
            ; mov r8, [rax + hoff]
        );

        if arity == 0 {
            dynasm!(asm; mov rdi, r8; xor rsi, rsi; xor edx, edx);
            call_abs!(asm, pti);
            continue;
        }

        let t0_off = head_col_slot(var_count, max_head_arity, 0);

        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 32i8]
            ; mov r10, [rax + hoff]
            ; mov r11, [r10]
            ; mov r10d, [r10 + 8i8]
        );

        let call_insert = asm.new_dynamic_label();
        let after_emit = asm.new_dynamic_label();

        dynasm!(asm; test r10d, r10d; jz =>call_insert);

        dynasm!(asm; xor eax, eax);
        for col in 0..arity {
            dynasm!(asm
                ; imul eax, eax, 0x9e3779b9u32 as i32
                ; add eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
            );
        }
        dynasm!(asm; cmp eax, -1i32; jne >no_remap; dec eax; no_remap:);

        dynasm!(asm
            ; lea rcx, [r10 - 1]
            ; mov edx, eax
            ; and rdx, rcx
        );

        let stride_d = ((arity + 1) * 4) as i32;
        let probe_lp = asm.new_dynamic_label();
        let probe_nx = asm.new_dynamic_label();

        dynasm!(asm; =>probe_lp);
        dynasm!(asm
            ; imul rsi, rdx, stride_d
            ; add rsi, r11
            ; mov edi, [rsi]
            ; cmp edi, -1i32
            ; je =>call_insert
            ; cmp edi, eax
            ; jne =>probe_nx
        );
        for col in 0..arity {
            dynasm!(asm
                ; mov edi, [rsi + ((1+col)*4) as i32]
                ; mov r8d, [rbp + head_col_slot(var_count, max_head_arity, col)]
                ; cmp edi, r8d
                ; jne =>probe_nx
            );
        }
        dynasm!(asm; jmp =>after_emit);

        dynasm!(asm; =>probe_nx);
        dynasm!(asm; inc rdx; and rdx, rcx; jmp =>probe_lp);

        dynasm!(asm; =>call_insert);
        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 16i8]
            ; mov rdi, [rax + hoff]
            ; lea rsi, [rbp + t0_off]
            ; mov edx, arity as i32
        );
        call_abs!(asm, pti);

        dynasm!(asm; =>after_emit);
    }
    Ok(())
}

// ─── N-clause recursive scan emission ────────────────────────────────────

/// Parameters shared across all recursive calls.
struct EmitParams<'a> {
    rule_i: usize,
    clauses: &'a [CClause],
    heads: &'a [CHeadClause],
    rule_conds: &'a [CExpr],
    /// use_recent[i] = true iff clause i is the "recent" clause in this variant.
    use_recent: &'a [bool],
    /// is_rec[i] = true iff clause i's relation appears in a head (→ data_ptr may change).
    is_rec: &'a [bool],
    var_count: usize,
    max_head_arity: usize,
    /// Maximum nesting depth across all rules in this stratum (used for slot layout).
    max_depth: usize,
    pti: usize,
    pdptr: usize,
    prptr: usize,
    pcount: usize,
    var_locs: &'a [VarLoc],
}

/// Emit code for clause `level` and all deeper clauses.
///
/// `outer_exit`:    label to jump to when the level-0 loop is completely done.
/// `when_exhausted`: label to jump to when THIS level's linked-list is exhausted
///                   (sentinel hit).  For level 0 this equals `outer_exit`.
///
/// Control-flow contract:
///   - Falls through when the level-0 outer loop advances (i.e., after `outer_continue` in
///     the level-0 case, nothing extra is emitted here — the advance is internal).
///   - For level >= 1: emits the full inner loop, ending with a jump to `inner_hdr`
///     on `inner_continue`.  When the sentinel is hit the function jumps to
///     `when_exhausted` (without fallthrough).
fn emit_clause_level(
    asm: &mut Assembler,
    p: &EmitParams<'_>,
    level: usize,
    outer_exit: DynamicLabel,
    when_exhausted: DynamicLabel,
) -> Result<(), String> {
    let clause = &p.clauses[level];

    if level == 0 {
        // ── Level 0: full scan over clause0 ────────────────────────────────
        let arity = clause.args.len();
        let stride = (arity * 4) as i32;
        let use_recent = p.use_recent[0];
        let uri32 = if use_recent { 1i32 } else { 0i32 };

        // Count
        load_rel_rdi!(asm, p.rule_i, 0usize);
        dynasm!(asm; mov esi, uri32);
        call_abs!(asm, p.pcount);
        dynasm!(asm; mov r12, rax; test r12, r12; jz =>outer_exit);

        // Recent ptr (stable across loop; r13 = recent_ptr)
        if use_recent {
            load_rel_rdi!(asm, p.rule_i, 0usize);
            call_abs!(asm, p.prptr);
            dynasm!(asm; mov r13, rax);
        }

        dynasm!(asm; xor r14d, r14d);  // i = 0

        let loop_hdr = asm.new_dynamic_label();
        let outer_continue = asm.new_dynamic_label();

        dynasm!(asm; =>loop_hdr);
        dynasm!(asm; cmp r14, r12; jge =>outer_exit);

        // tuple_idx → rbx (survives the data_ptr call since rbx is callee-saved)
        if use_recent {
            dynasm!(asm; mov rbx, [r13 + r14*8]);
        } else {
            dynasm!(asm; mov rbx, r14);
        }

        // data_ptr for clause0
        load_rel_rdi!(asm, p.rule_i, 0usize);
        call_abs!(asm, p.pdptr);
        // rax = data_ptr, rbx = tuple_idx

        dynasm!(asm
            ; imul rcx, rbx, stride
            ; add rax, rcx             // rax = tuple_ptr
        );

        // Bind clause0 cols
        emit_bind_cols(asm, clause, outer_continue, p.var_locs)?;

        // Clause0 conditions
        for cond in &clause.conditions {
            if let CCondition::If(expr) = cond {
                emit_expr(asm, expr, p.var_locs)?;
                dynasm!(asm; test eax, eax; jz =>outer_continue);
            }
        }

        if p.clauses.len() == 1 {
            // 1-clause rule: emit rule conditions and heads here
            for expr in p.rule_conds {
                emit_expr(asm, expr, p.var_locs)?;
                dynasm!(asm; test eax, eax; jz =>outer_continue);
            }
            emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs)?;
        } else {
            // Multi-clause: recurse into level 1.
            // The level-1 code is emitted inline here; it ends by jumping to
            // outer_continue when exhausted.
            emit_clause_level(asm, p, 1, outer_exit, outer_continue)?;
        }

        dynasm!(asm; =>outer_continue);
        dynasm!(asm; inc r14; jmp =>loop_hdr);

    } else {
        // ── Level >= 1: hash probe + inner scan ────────────────────────────
        //
        // Contiguous mode (!is_rec): values array holds tuple_idxs sequentially.
        //   Inner loop: j = 0..count, load values[head + j]
        //   No pointer-chasing — sequential scan enables CPU speculation/prefetch.
        //
        // Linked-list mode (is_rec): values are stride-2 nodes (tuple_idx, next).
        //   Inner loop: follow next-pointers until SENTINEL.

        let arity = clause.args.len();
        let stride = (arity * 4) as i32;
        let use_recent = p.use_recent[level];
        let is_rec = p.is_rec[level];
        // Contiguous mode: EDB relations (!is_rec) use sequential scan.
        let is_contiguous = !is_rec;

        // Compute probe key into ecx
        if clause.bound_cols.is_empty() {
            return Err(format!(
                "asm: clause{level} has no bound_cols (unsupported)"
            ));
        }
        let primary_col = clause.bound_cols[0];
        match &clause.args[primary_col] {
            CClauseArg::Var(var_id) => {
                emit_load_var_ecx(asm, p.var_locs, *var_id);
            }
            CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
                dynasm!(asm; mov ecx, *n);
            }
            _ => return Err(format!("asm: clause{level} primary bound col has unsupported arg")),
        }

        // Save the ENCLOSING level's r15 and rbx before we clobber them with
        // the new probe.  The enclosing level is `level-1`.
        // When level == 1 this is the first probe; level-1's data (r12/r13/r14)
        // is not clobbered, so no save is needed.
        // When level >= 2 we must save level-1's r15/rbx.
        if level >= 2 {
            let save_level = level - 1;
            let node_slot = level_node_save_slot(save_level, p.var_count, p.max_head_arity, p.max_depth);
            let dptr_slot = level_dptr_save_slot(save_level, p.var_count, p.max_head_arity, p.max_depth);
            dynasm!(asm
                ; mov [rbp + node_slot], r15
                ; mov [rbp + dptr_slot], rbx
            );
        }

        // Probe: handle_off uses (clause_seq * 2 + use_recent) * 24
        let handle_off: i32 = ((level * 2 + if use_recent { 1 } else { 0 }) * 24) as i32;
        load_rule_ctx!(asm, p.rule_i);
        dynasm!(asm
            ; mov rax, [rax + 24i8]   // lookup_handles ptr
            ; add rax, handle_off
            ; mov r8,  [rax]          // entries_ptr
            ; mov r9,  [rax + 8i8]   // values_ptr
            ; mov r10d, [rax + 16i8] // mask
        );

        // Hash: slot = (key * KNUTH32) & mask  (ecx = key)
        dynasm!(asm
            ; mov eax, ecx
            ; imul eax, eax, 0x9e3779b1u32 as i32
            ; mov edx, eax
            ; and rdx, r10
        );

        let probe_lp  = asm.new_dynamic_label();
        let probe_ok  = asm.new_dynamic_label();
        let probe_ovf = asm.new_dynamic_label();
        let after_probe = asm.new_dynamic_label();

        dynasm!(asm; =>probe_lp);
        dynasm!(asm
            ; lea rax, [rdx + rdx]
            ; shl rax, 3
            ; add rax, r8
            ; mov esi, [rax]
            ; cmp esi, -1i32
            ; je =>probe_ovf
            ; cmp esi, ecx
            ; je =>probe_ok
            ; inc rdx
            ; and rdx, r10
            ; jmp =>probe_lp
        );
        // probe_ok: rax = entry_ptr, load head at +4, and count at +8 (contiguous mode)
        dynasm!(asm; =>probe_ok);
        if is_contiguous {
            dynasm!(asm
                ; mov esi, [rax + 4i8]    // esi = head (start offset)
                ; mov eax, [rax + 8i8]    // eax = count
                ; jmp =>after_probe
            );
        } else {
            dynasm!(asm; mov esi, [rax + 4i8]; jmp =>after_probe);
        }

        dynasm!(asm; =>probe_ovf);
        dynasm!(asm
            ; lea rax, [r10 + 1]
            ; lea rax, [rax + rax]
            ; shl rax, 3
            ; add rax, r8
            ; mov esi, [rax]
            ; cmp esi, ecx
            ; jne =>when_exhausted    // key not found → exhaust this level
        );
        if is_contiguous {
            dynasm!(asm
                ; mov esi, [rax + 4i8]    // esi = head
                ; mov eax, [rax + 8i8]    // eax = count
            );
        } else {
            dynasm!(asm; mov esi, [rax + 4i8]);
        }

        dynasm!(asm; =>after_probe);

        let vs = level_vptr_slot(level, p.var_count, p.max_head_arity);

        if is_contiguous {
            // Contiguous mode:
            //   esi = head (start index in flat values array)
            //   eax = count
            //   r15d = j counter (starts at 0)
            //
            // For arity-2 EDB relations, values[head+j] stores the free column value
            // directly (not tuple_idx), eliminating the data_ptr dereference.
            // In col-value mode, count lives in rbx (callee-saved, preserved by
            // existing dptr_slot save/restore when descending into sub-levels).
            let is_col_value = arity == 2;

            dynasm!(asm
                ; test eax, eax
                ; jz =>when_exhausted
                ; xor r15d, r15d         // j = 0
            );
            // Pre-compute values_base = values_ptr + head*4 (saves one lea per inner iter).
            // For col_value: keep count in rbx (no cnt_slot stack store/load).
            // For standard: spill count to cnt_slot, load data_ptr into rbx.
            if is_col_value {
                dynasm!(asm
                    ; mov rbx, rax                    // rbx = count (callee-saved)
                    ; lea r9, [r9 + rsi*4]            // r9 = values_base
                    ; mov [rbp + vs], r9              // save values_base
                );
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov [rbp + cnt_slot], eax       // save count to stack
                    ; lea r9, [r9 + rsi*4]            // r9 = values_base
                    ; mov [rbp + vs], r9              // save values_base
                );
                load_rel_rdi!(asm, p.rule_i, level);
                call_abs!(asm, p.pdptr);
                dynasm!(asm; mov rbx, rax);
            }

            // ── Inner contiguous scan loop ────────────────────────────────────
            let inner_hdr = asm.new_dynamic_label();
            let inner_continue = asm.new_dynamic_label();

            dynasm!(asm; =>inner_hdr);
            if is_col_value {
                // Count in rbx: no stack load needed.
                dynasm!(asm; cmp r15d, ebx; jge =>when_exhausted);
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov eax, [rbp + cnt_slot]
                    ; cmp r15d, eax
                    ; jge =>when_exhausted
                );
            }

            // Load values_base[j] into ecx (= col_value for arity-2, or tuple_idx otherwise).
            dynasm!(asm
                ; mov rax, [rbp + vs]                   // values_base
                ; mov ecx, [rax + r15*4]                // values_base[j]
                ; inc r15d                              // j++
            );

            if is_col_value {
                // Col-value mode: ecx = free column value; no tuple_ptr needed.
                let primary_col = clause.bound_cols[0];
                let free_col = 1 - primary_col;
                emit_bind_col_value(asm, clause, free_col, inner_continue, p.var_locs)?;
            } else {
                // Standard: ecx = tuple_idx → compute tuple_ptr via data_ptr (rbx).
                dynasm!(asm
                    ; imul rcx, rcx, stride
                    ; add rcx, rbx
                    ; mov rax, rcx
                );
                // Bind cols (rax = tuple_ptr)
                emit_bind_cols(asm, clause, inner_continue, p.var_locs)?;
            }

            // Clause conditions
            for cond in &clause.conditions {
                if let CCondition::If(expr) = cond {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>inner_continue);
                }
            }

            if level + 1 == p.clauses.len() {
                for expr in p.rule_conds {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>inner_continue);
                }
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs)?;
                // For col-value existence checks (all cols bound, values sorted unique):
                // at most one match can exist, so skip the remaining scan after emitting.
                if is_col_value && clause.fresh_cols.is_empty() {
                    dynasm!(asm; jmp =>when_exhausted);
                }
            } else {
                let sub_exhausted = asm.new_dynamic_label();
                emit_clause_level(asm, p, level + 1, outer_exit, sub_exhausted)?;

                dynasm!(asm; =>sub_exhausted);
                let node_slot = level_node_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                let dptr_slot = level_dptr_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov r15, [rbp + node_slot]
                    ; mov rbx, [rbp + dptr_slot]
                );
            }

            dynasm!(asm; =>inner_continue);
            dynasm!(asm; jmp =>inner_hdr);

        } else {
            // Linked-list mode:
            //   esi = head node index; SENTINEL = -1u32
            dynasm!(asm; mov [rbp + vs], r9);  // save values_ptr for inner loop
            dynasm!(asm; cmp esi, -1i32; je =>when_exhausted; mov r15d, esi);

            // Pre-fetch data_ptr into rbx (if non-recursive — always false here since is_rec=true)
            // Actually for linked-list we keep the existing behaviour: re-fetch inside loop.

            // ── Inner linked-list loop ──────────────────────────────────────
            let inner_hdr = asm.new_dynamic_label();
            let inner_continue = asm.new_dynamic_label();

            dynasm!(asm; =>inner_hdr);
            dynasm!(asm; cmp r15d, -1i32; je =>when_exhausted);

            // Load tuple_idx (ecx) and next_node (r15) from values_ptr[r15*8]
            dynasm!(asm
                ; mov rax, [rbp + vs]
                ; mov ecx, [rax + r15*8]
                ; mov esi, [rax + r15*8 + 4i8]
                ; mov r15d, esi            // advance r15 to next node
                ; mov eax, ecx             // rax = tuple_idx
            );

            // Re-fetch data_ptr (recursive: head insert can reallocate)
            dynasm!(asm
                ; push rax
                ; sub rsp, 8
            );
            load_rel_rdi!(asm, p.rule_i, level);
            call_abs!(asm, p.pdptr);
            dynasm!(asm
                ; mov rbx, rax
                ; add rsp, 8
                ; pop rax
            );

            // tuple_ptr = data_ptr + tuple_idx * stride
            dynasm!(asm
                ; imul rax, rax, stride
                ; add rax, rbx
            );

            // Bind cols (rax = tuple_ptr)
            emit_bind_cols(asm, clause, inner_continue, p.var_locs)?;

            // Clause conditions
            for cond in &clause.conditions {
                if let CCondition::If(expr) = cond {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>inner_continue);
                }
            }

            if level + 1 == p.clauses.len() {
                // Last clause: emit rule-level conditions and heads
                for expr in p.rule_conds {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>inner_continue);
                }
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs)?;
                // Fall to inner_continue
            } else {
                // Not last: recurse to level+1.
                let sub_exhausted = asm.new_dynamic_label();
                emit_clause_level(asm, p, level + 1, outer_exit, sub_exhausted)?;

                // sub_exhausted: level+1 is done; restore level's r15 and rbx.
                dynasm!(asm; =>sub_exhausted);
                let node_slot = level_node_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                let dptr_slot = level_dptr_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov r15, [rbp + node_slot]
                    ; mov rbx, [rbp + dptr_slot]
                );
                // Fall to inner_continue → jmp inner_hdr
            }

            dynasm!(asm; =>inner_continue);
            dynasm!(asm; jmp =>inner_hdr);
        } // end else (linked-list branch)
    } // end else (level >= 1)

    Ok(())
}

// ─── Rule variant emission ─────────────────────────────────────────────────

/// Emit one rule variant (full or recent).
#[allow(clippy::too_many_arguments)]
fn emit_rule_variant(
    asm: &mut Assembler,
    rule_i: usize,
    clauses: &[CClause],
    heads: &[CHeadClause],
    rule_conds: &[CExpr],
    recent_idx: Option<usize>,
    var_count: usize,
    max_head_arity: usize,
    max_depth: usize,
    pti: usize,
    pcount: usize,
    pdptr: usize,
    prptr: usize,
) -> Result<(), String> {
    let variant_exit = asm.new_dynamic_label();

    let use_recent0 = recent_idx == Some(0);
    let is_multi_clause = clauses.len() >= 2;
    let outer_fresh_ids: Vec<u32> = if clauses.is_empty() {
        vec![]
    } else {
        let mut ids: Vec<u32> = clauses[0].fresh_cols.iter().map(|&(_, id)| id).collect();
        ids.sort_unstable();
        ids
    };
    let var_locs = compute_var_locs(var_count, &outer_fresh_ids, use_recent0, is_multi_clause);

    let use_recent_vec: Vec<bool> = (0..clauses.len())
        .map(|i| recent_idx == Some(i))
        .collect();
    let is_rec_vec: Vec<bool> = clauses
        .iter()
        .map(|c| heads.iter().any(|h| h.relation == c.relation))
        .collect();

    if clauses.is_empty() {
        for expr in rule_conds {
            emit_expr(asm, expr, &var_locs)?;
            dynasm!(asm; test eax, eax; jz =>variant_exit);
        }
        emit_heads(asm, heads, rule_i, var_count, max_head_arity, pti, &var_locs)?;
        dynasm!(asm; =>variant_exit);
        return Ok(());
    }

    let p = EmitParams {
        rule_i,
        clauses,
        heads,
        rule_conds,
        use_recent: &use_recent_vec,
        is_rec: &is_rec_vec,
        var_count,
        max_head_arity,
        max_depth,
        pti,
        pdptr,
        prptr,
        pcount,
        var_locs: &var_locs,
    };

    emit_clause_level(asm, &p, 0, variant_exit, variant_exit)?;

    dynasm!(asm; =>variant_exit);
    Ok(())
}

// ─── Main entry point ─────────────────────────────────────────────────────

pub fn codegen_stratum_asm(
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    var_count: usize,
    advance_s4_addr: usize,
    packed_try_insert_addr: usize,
    packed_count_addr: usize,
    packed_data_ptr_addr: usize,
    packed_recent_ptr_addr: usize,
) -> Result<AsmStratum, String> {
    // ── Eligibility ──────────────────────────────────────────────────────────
    for (ri, (clauses, heads, conds)) in rules.iter().enumerate() {
        for head in *heads {
            if head.args.len() > 8 {
                return Err(format!("asm: rule {ri} head arity {} > 8", head.args.len()));
            }
            for arg in &head.args { check_expr(arg)?; }
        }
        for cond in *conds { check_expr(cond)?; }
        for clause in *clauses {
            for cond in &clause.conditions {
                if let CCondition::If(e) = cond { check_expr(e)?; }
            }
        }
        // clause0 must have no bound cols (full scan)
        if let Some(c0) = clauses.first() {
            if !c0.bound_cols.is_empty() {
                return Err(format!("asm: rule {ri} clause0 has bound_cols; unsupported"));
            }
        }
        // clauses 1..N must have at least one bound col for index probe
        for (ci, c) in clauses.iter().enumerate().skip(1) {
            if c.bound_cols.is_empty() {
                return Err(format!("asm: rule {ri} clause{ci} has no bound_cols; unsupported"));
            }
        }
    }

    let max_head_arity = rules
        .iter()
        .flat_map(|(_, hs, _)| hs.iter().map(|h| h.args.len()))
        .max()
        .unwrap_or(1);
    let max_head_arity = max_head_arity.max(1);

    // max_depth = maximum nesting depth (= max inner clauses count = max_clauses - 1)
    // Use at least 1 to keep the frame layout simple.
    let max_depth = rules
        .iter()
        .map(|(clauses, _, _)| clauses.len().saturating_sub(1))
        .max()
        .unwrap_or(0)
        .max(1);

    let frame_sz = frame_size(var_count, max_head_arity, max_depth);

    let mut asm = Assembler::new().map_err(|e| format!("asm alloc: {e}"))?;
    dynasm!(asm; .arch x64);
    let fn_start = asm.offset();

    // Prologue
    dynasm!(asm
        ; push rbp
        ; mov rbp, rsp
        ; push rbx
        ; push r12
        ; push r13
        ; push r14
        ; push r15
        ; sub rsp, frame_sz
        ; mov [rbp + CTX_SLOT], rdi
    );

    let exit_lbl = asm.new_dynamic_label();

    // Full variants
    for (rule_i, (clauses, heads, conds)) in rules.iter().enumerate() {
        emit_rule_variant(
            &mut asm, rule_i, clauses, heads, conds, None,
            var_count, max_head_arity, max_depth,
            packed_try_insert_addr, packed_count_addr,
            packed_data_ptr_addr, packed_recent_ptr_addr,
        )?;
    }

    // jit_stratum_advance_s4(ctx) → al
    dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
    call_abs!(asm, advance_s4_addr);
    dynasm!(asm; test al, al; jz =>exit_lbl);

    // Recent-variant loop
    let inner_hdr = asm.new_dynamic_label();
    dynasm!(asm; =>inner_hdr);

    for (rule_i, (clauses, heads, conds)) in rules.iter().enumerate() {
        for clause_seq in 0..clauses.len() {
            emit_rule_variant(
                &mut asm, rule_i, clauses, heads, conds, Some(clause_seq),
                var_count, max_head_arity, max_depth,
                packed_try_insert_addr, packed_count_addr,
                packed_data_ptr_addr, packed_recent_ptr_addr,
            )?;
        }
    }

    dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
    call_abs!(asm, advance_s4_addr);
    dynasm!(asm; test al, al; jnz =>inner_hdr);

    // Epilogue
    dynasm!(asm; =>exit_lbl);
    dynasm!(asm
        ; add rsp, frame_sz
        ; pop r15
        ; pop r14
        ; pop r13
        ; pop r12
        ; pop rbx
        ; pop rbp
        ; ret
    );

    let buffer = asm.finalize().map_err(|_| "asm finalize: reloc error".to_string())?;
    let fn_ptr: StratumStage4Fn = unsafe { std::mem::transmute(buffer.ptr(fn_start)) };
    Ok(AsmStratum { _buffer: buffer, fn_ptr })
}
