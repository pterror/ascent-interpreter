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
//!   r15 = innermost active value-scan counter j (zero-extended u32)
//!   rbx = innermost active data_ptr (or count in col-value mode)
//!
//! When descending from level L to level L+1:
//!   - level L's r15 (next_node) is saved to level_node_save_slot(L)
//!   - level L's rbx (data_ptr)  is saved to level_dptr_save_slot(L)
//!   - level L's vptr remains in level_vptr_slot(L) (written at probe setup, not clobbered)
//!
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

// ─── JitTupleSet probe ────────────────────────────────────────────────────

/// Emit an inline `JitTupleSet` probe for a fully-bound clause (all args already bound).
///
/// On match (tuple found in set): fall through.
/// On no-match (empty slot encountered): jump to `when_exhausted`.
///
/// # Constraints
/// - Supports arity 1–3 (returns Err for arity 0 or > 3).
/// - All clause args must be `Var`, `I32`, or `Bool`.
///
/// # Register protocol
/// Clobbers: rax, rcx, rdx, rdi, rsi, r8, r9, r10, r11.
///
/// # Source of JitTupleSet pointer
/// - `jit_rel_for_probe`: when `Some(jit_rel_ptr_reg)`, the `JitTupleSet` is loaded
///   from that already-loaded `*mut JitRelData` register (offset 32 = `JitRelData::tuple_set`).
///   This is the native path.
/// - When `None`, the old path: ctx->tuple_sets_buf[handle_idx] via CTX_SLOT.
///
/// # JitTupleSet layout (from storage.rs)
/// - slots @ 0, mask @ 8, len @ 16
/// - stride = arity + 1 words per slot
/// - slot[0] = hash_tag (0 = empty), slot[1..N] = tuple words
fn emit_tuple_set_probe(
    asm: &mut Assembler,
    clause: &CClause,
    handle_idx: usize,
    when_exhausted: DynamicLabel,
    var_locs: &[VarLoc],
    // When Some, load JitTupleSet from JitRelData.tuple_set (offset 32) using
    // the given flat scan-rels index into total_rels.  When None, use the old
    // ctx->tuple_sets_buf path.
    native_total_rel_flat_idx: Option<usize>,
) -> Result<(), String> {
    let arity = clause.args.len();
    if arity == 0 || arity > 3 {
        return Err(format!(
            "emit_tuple_set_probe: arity {arity} unsupported (supported: 1–3)"
        ));
    }

    // Validate arg types
    for arg in &clause.args {
        match arg {
            CClauseArg::Var(_) => {}
            CClauseArg::Expr(CExpr::Literal(crate::value::Value::I32(_))) => {}
            CClauseArg::Expr(CExpr::Literal(crate::value::Value::Bool(_))) => {}
            _ => return Err(format!("emit_tuple_set_probe: unsupported arg type: {arg:?}")),
        }
    }

    // Load JitTupleSet pointer into rdi.
    if let Some(flat_idx) = native_total_rel_flat_idx {
        // Native path: total_rels[flat_idx]->tuple_set (JitRelData offset 32)
        // total_rels is at offset 8 in StratumStage4NativeCtx.
        let flat_idx_i32 = (flat_idx * 8) as i32;
        dynasm!(asm
            ; mov rdi, [rbp + CTX_SLOT]        // native ctx (*mut StratumStage4NativeCtx)
            ; mov rdi, [rdi + 8]               // total_rels ptr (offset 8)
            ; mov rdi, [rdi + flat_idx_i32]    // *mut JitRelData
            ; add rdi, 32                      // &JitRelData.tuple_set (offset 32)
        );
    } else {
        let handle_idx_i32 = (handle_idx * 8) as i32;
        // Old path: ctx->tuple_sets_buf[handle_idx]
        dynasm!(asm
            ; mov rdi, [rbp + CTX_SLOT]        // ctx
            ; mov rdi, [rdi + 56]              // tuple_sets_buf (offset 56 in StratumStage4Ctx)
            ; mov rdi, [rdi + handle_idx_i32]  // *const JitTupleSet for this clause
        );
    }

    // Load JitTupleSet fields
    // r8 = slots ptr (offset 0), r9 = mask (offset 8, u64)
    dynasm!(asm
        ; mov r8, [rdi]      // r8 = slots ptr
        ; mov r9, [rdi + 8]  // r9 = mask
    );

    // Load args into registers: edi=arg0, esi=arg1 (arity>=2), r11d=arg2 (arity==3).
    macro_rules! load_clause_arg {
        ($asm:expr, $idx:expr, $tgt:ident) => {
            match &clause.args[$idx] {
                CClauseArg::Var(var_id) => {
                    emit_load_var($asm, var_locs, *var_id);
                    dynasm!($asm; mov $tgt, eax);
                }
                CClauseArg::Expr(CExpr::Literal(crate::value::Value::I32(n))) => {
                    dynasm!($asm; mov $tgt, *n);
                }
                CClauseArg::Expr(CExpr::Literal(crate::value::Value::Bool(b))) => {
                    let v = if *b { 1i32 } else { 0i32 };
                    dynasm!($asm; mov $tgt, v);
                }
                _ => return Err(format!("emit_tuple_set_probe: unsupported arg type")),
            }
        };
    }

    load_clause_arg!(asm, 0, edi);
    if arity >= 2 { load_clause_arg!(asm, 1, esi); }
    if arity >= 3 { load_clause_arg!(asm, 2, r11d); }

    // Compute tuple_hash(words):
    //   h = 0x9e3779b9; for each word w: h = h*0x9e3779b9 + w; if h==0: h=1
    dynasm!(asm; mov eax, 0x9e3779b9u32 as i32);
    dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, edi);
    if arity >= 2 { dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, esi); }
    if arity >= 3 { dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, r11d); }

    let hash_ok = asm.new_dynamic_label();
    dynasm!(asm; test eax, eax; jnz =>hash_ok; mov eax, 1; =>hash_ok);
    dynasm!(asm; mov edx, eax);  // edx = hash_tag

    // r10 = slot index = hash & mask
    dynasm!(asm; mov r10, rax; and r10, r9);

    // Probe loop: stride = arity+1 words.
    //   arity=1 → stride=2: rcx = r10*2 (lea [r10+r10])
    //   arity=2 → stride=3: rcx = r10*3 (lea [r10+r10*2])
    //   arity=3 → stride=4: rcx = r10*4 (lea [r10+r10*3])
    let probe_lp   = asm.new_dynamic_label();
    let probe_next = asm.new_dynamic_label();
    let probe_done = asm.new_dynamic_label();

    dynasm!(asm; =>probe_lp);
    match arity {
        1 => dynasm!(asm; lea rcx, [r10 + r10]),
        2 => dynasm!(asm; lea rcx, [r10 + r10*2]),
        _ => dynasm!(asm; lea rcx, [r10 + r10*3]),
    }

    // Check hash_tag
    dynasm!(asm
        ; mov eax, [r8 + rcx*4]   // hash_tag
        ; test eax, eax
        ; je =>when_exhausted      // empty → not found
        ; cmp eax, edx             // hash tag matches?
        ; jne =>probe_next
    );

    // Full tuple comparison
    dynasm!(asm; cmp [r8 + rcx*4 + 4], edi; jne =>probe_next);
    if arity >= 2 { dynasm!(asm; cmp [r8 + rcx*4 + 8], esi;   jne =>probe_next); }
    if arity >= 3 { dynasm!(asm; cmp [r8 + rcx*4 + 12], r11d; jne =>probe_next); }

    // Found — fall through
    dynasm!(asm; jmp =>probe_done);

    dynasm!(asm; =>probe_next);
    dynasm!(asm; inc r10; and r10, r9; jmp =>probe_lp);

    dynasm!(asm; =>probe_done);

    Ok(())
}

// ─── Expression compilation ───────────────────────────────────────────────

fn check_expr(expr: &CExpr) -> Result<(), String> {
    match expr {
        CExpr::Var(_) | CExpr::DerefVar(_) => Ok(()),
        CExpr::Literal(Value::I32(_)) | CExpr::Literal(Value::Bool(_)) => Ok(()),
        CExpr::VarBinVar(op, _, _)
        | CExpr::VarBinLit(op, _, _)
        | CExpr::LitBinVar(op, _, _) => check_binop(*op),
        CExpr::Binary(op, a, b) => {
            check_binop(*op)?;
            check_expr(a)?;
            check_expr(b)
        }
        CExpr::Unary(CUnOp::Not | CUnOp::Neg | CUnOp::Deref, i) => check_expr(i),
        _ => Err(format!("asm: unsupported expr: {expr:?}")),
    }
}

fn check_binop(op: CBinOp) -> Result<(), String> {
    use CBinOp::*;
    match op {
        Add | Sub | Mul | Eq | Ne | Lt | Le | Gt | Ge | And | Or
        | BitXor | Shl | Shr => Ok(()),
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
        CExpr::DerefVar(id) => emit_load_var(asm, var_locs, *id),
        CExpr::Unary(CUnOp::Not, i) => {
            emit_expr(asm, i, var_locs)?;
            dynasm!(asm; xor eax, 1i8);
        }
        CExpr::Unary(CUnOp::Neg, i) => {
            emit_expr(asm, i, var_locs)?;
            dynasm!(asm; neg eax);
        }
        CExpr::Unary(CUnOp::Deref, i) => emit_expr(asm, i, var_locs)?,
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
        Eq     => dynasm!(asm; cmp eax, ecx; sete al; movzx eax, al),
        Ne     => dynasm!(asm; cmp eax, ecx; setne al; movzx eax, al),
        Lt     => dynasm!(asm; cmp eax, ecx; setl al; movzx eax, al),
        Le     => dynasm!(asm; cmp eax, ecx; setle al; movzx eax, al),
        Gt     => dynasm!(asm; cmp eax, ecx; setg al; movzx eax, al),
        Ge     => dynasm!(asm; cmp eax, ecx; setge al; movzx eax, al),
        BitXor => dynasm!(asm; xor eax, ecx),
        Shl    => dynasm!(asm; shl eax, cl),
        Shr    => dynasm!(asm; sar eax, cl),
        _      => return Err(format!("asm: unsupported binop: {op:?}")),
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

/// Build and insert all heads: inline dedup probe for duplicates, then
/// `packed_try_insert` for new tuples.
///
/// The inline probe fast-paths the common duplicate case (no Rust call).
/// New tuples are routed through `packed_try_insert`, which keeps the dedup
/// table count accurate and triggers `maybe_grow` at the right time — avoiding
/// 100%-full-table infinite-probe loops that occur when count is not maintained.
#[allow(clippy::too_many_arguments)]
fn emit_heads(
    asm: &mut Assembler,
    heads: &[CHeadClause],
    rule_i: usize,
    var_count: usize,
    max_head_arity: usize,
    pti: usize,
    var_locs: &[VarLoc],
    // When true, load *mut PackedStorage from ctx->head_specs[rule_head_base+hi].rel
    // instead of from rule_ctx->head_rels[hi].  head_specs is at offset 64 in NativeCtx.
    use_jit_native: bool,
    rule_head_base: usize,
) -> Result<(), String> {
    for (hi, head) in heads.iter().enumerate() {
        let arity = head.args.len();
        let hoff: i32 = (hi as i32) * 8;
        let t0_off = head_col_slot(var_count, max_head_arity, 0);

        for (col, arg) in head.args.iter().enumerate() {
            emit_expr(asm, arg, var_locs)?;
            dynasm!(asm; mov [rbp + head_col_slot(var_count, max_head_arity, col)], eax);
        }

        if use_jit_native {
            // Native path: load *mut PackedStorage from ctx->head_specs[flat_hi].rel.
            // head_specs is at offset 64 in StratumStage4NativeCtx.
            // NativeHeadSpec is 8 bytes with rel (*mut PackedStorage) at offset 0.
            // flat_hi = rule_head_base + hi.
            let flat_hi_i32 = ((rule_head_base + hi) * 8) as i32;
            dynasm!(asm
                ; mov rdi, [rbp + CTX_SLOT]        // NativeCtx*
                ; mov rdi, [rdi + 64]              // head_specs (*const NativeHeadSpec)
                ; mov rdi, [rdi + flat_hi_i32]     // *mut PackedStorage
            );
            if arity == 0 {
                dynasm!(asm; xor rsi, rsi; xor edx, edx);
            } else {
                dynasm!(asm
                    ; lea rsi, [rbp + t0_off]
                    ; mov edx, arity as i32
                );
            }
            call_abs!(asm, pti);
            continue;
        }

        if arity == 0 {
            // Zero-arity: no dedup table in JIT; fall back to packed_try_insert.
            load_rule_ctx!(asm, rule_i);
            dynasm!(asm
                ; mov rax, [rax + 16i8]
                ; mov r8, [rax + hoff]
            );
            dynasm!(asm; mov rdi, r8; xor rsi, rsi; xor edx, edx);
            call_abs!(asm, pti);
            continue;
        }

        // Load JitDedupHandle for this head:
        //   rule_ctx->head_dedup_handles[hi] → *mut JitDedupHandle
        //   JitDedupHandle: entries @ 0, cap @ 8
        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 32i8]      // head_dedup_handles ptr
            ; mov r10, [rax + hoff]      // *mut JitDedupHandle
            ; mov r11, [r10]             // r11 = entries (*mut u32)
            ; mov r10d, [r10 + 8i8]     // r10d = cap (u32)
        );

        let call_insert = asm.new_dynamic_label();
        let after_emit = asm.new_dynamic_label();

        // If cap == 0, the dedup table is uninitialized; must insert via Rust.
        dynasm!(asm; test r10d, r10d; jz =>call_insert);

        // Compute hash (same as jit_dedup_hash): h = 0; for each word: h = h*KNUTH + word
        dynasm!(asm; xor eax, eax);
        for col in 0..arity {
            dynasm!(asm
                ; imul eax, eax, 0x9e3779b9u32 as i32
                ; add eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
            );
        }
        // Remap hash 0xFFFFFFFF (-1) → 0xFFFFFFFE (avoid sentinel)
        dynasm!(asm; cmp eax, -1i32; jne >no_remap; dec eax; no_remap:);

        // probe: mask = cap - 1 (in rcx), slot = hash & mask (in rdx)
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
            ; add rsi, r11              // rsi = &slot[0]
            ; mov edi, [rsi]            // edi = slot hash
            ; cmp edi, -1i32
            ; je =>call_insert          // empty slot → new tuple
            ; cmp edi, eax
            ; jne =>probe_nx            // hash mismatch → next slot
        );
        // Hash match: compare tuple words
        for col in 0..arity {
            dynasm!(asm
                ; mov edi, [rsi + ((1+col)*4) as i32]
                ; mov r8d, [rbp + head_col_slot(var_count, max_head_arity, col)]
                ; cmp edi, r8d
                ; jne =>probe_nx
            );
        }
        // All words match → duplicate, skip
        dynasm!(asm; jmp =>after_emit);

        dynasm!(asm; =>probe_nx);
        dynasm!(asm; inc rdx; and rdx, rcx; jmp =>probe_lp);

        // ── call_insert: new tuple found; delegate to packed_try_insert ──
        // This handles dedup table growth (maybe_grow) and storage insertion
        // correctly. The inline probe above handled the fast duplicate-skip path;
        // new tuples always go through Rust so count stays accurate and the table
        // never fills to 100% occupancy.
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
    /// Absolute starting index of this rule in the flat handles_buf / tuple_sets_buf.
    rule_handle_base: usize,
    /// Absolute starting index of this rule's heads in the flat head_specs array
    /// (used by native path to load *mut PackedStorage for packed_try_insert).
    rule_head_base: usize,
    /// When true, use the native JIT path: read data directly from JitRelData fields
    /// instead of calling packed_count/packed_data_ptr/packed_recent_ptr.
    /// CTX_SLOT holds a *mut StratumStage4NativeCtx.
    use_jit_native: bool,
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

        // flat handle index for clause 0:  rule_handle_base + 0*2 + use_recent
        let flat_idx0 = p.rule_handle_base + if use_recent { 1 } else { 0 };
        let flat_idx0_i32 = (flat_idx0 * 8) as i32;

        if p.use_jit_native {
            // ── Native path: read directly from JitRelData ──────────────────
            // scan_rels is at offset 0 in StratumStage4NativeCtx.
            // scan_rels[flat_idx0] → *mut JitRelData  (scan version: total or recent)
            //
            // r12 = count (JitRelData.len @ 8)
            // r13 = data_ptr (JitRelData.data @ 0) — needed for both total and recent
            // r14 = loop counter i

            // Load scan_rel for clause 0
            dynasm!(asm
                ; mov rdi, [rbp + CTX_SLOT]         // native ctx
                ; mov rdi, [rdi]                    // scan_rels (offset 0)
                ; mov rdi, [rdi + flat_idx0_i32]    // *mut JitRelData
            );

            // r12 = len
            dynasm!(asm
                ; mov r12, [rdi + 8]                // JitRelData.len @ 8
                ; test r12, r12
                ; jz =>outer_exit
            );

            if use_recent {
                // r13 is reserved for the data ptr (excluded from variable registers when
                // use_recent0=true by compute_var_locs).  Cache it here to avoid reloading.
                dynasm!(asm; mov r13, [rdi]);   // JitRelData.data @ 0
            }
            // (For total scan, r13 may hold a variable, so reload data ptr each iteration)

            dynasm!(asm; xor r14d, r14d);  // i = 0

            let loop_hdr = asm.new_dynamic_label();
            let outer_continue = asm.new_dynamic_label();

            dynasm!(asm; =>loop_hdr);
            dynasm!(asm; cmp r14, r12; jge =>outer_exit);

            if use_recent {
                // recent: tuple_ptr = r13 + i * stride (r13 = recent.data cached above)
                dynasm!(asm
                    ; imul rcx, r14, stride
                    ; lea rax, [r13 + rcx]          // rax = tuple_ptr
                );
            } else {
                // total: reload data_ptr from scan_rels each iteration
                dynasm!(asm
                    ; mov rax, [rbp + CTX_SLOT]         // native ctx
                    ; mov rax, [rax]                    // scan_rels (offset 0)
                    ; mov rax, [rax + flat_idx0_i32]    // *mut JitRelData
                    ; mov rax, [rax]                    // JitRelData.data (offset 0)
                    ; imul rcx, r14, stride
                    ; add rax, rcx                      // rax = data + i * stride
                );
            }

            // Bind clause0 cols (rax = tuple_ptr)
            emit_bind_cols(asm, clause, outer_continue, p.var_locs)?;

            // Clause0 conditions
            for cond in &clause.conditions {
                if let CCondition::If(expr) = cond {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>outer_continue);
                }
            }

            if p.clauses.len() == 1 {
                for expr in p.rule_conds {
                    emit_expr(asm, expr, p.var_locs)?;
                    dynasm!(asm; test eax, eax; jz =>outer_continue);
                }
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base)?;
            } else {
                emit_clause_level(asm, p, 1, outer_exit, outer_continue)?;
            }

            dynasm!(asm; =>outer_continue);
            dynasm!(asm; inc r14; jmp =>loop_hdr);
        } else {
        // ── Old path: use packed_count / packed_recent_ptr / packed_data_ptr callbacks ──

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
            emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base)?;
        } else {
            // Multi-clause: recurse into level 1.
            // The level-1 code is emitted inline here; it ends by jumping to
            // outer_continue when exhausted.
            emit_clause_level(asm, p, 1, outer_exit, outer_continue)?;
        }

        dynasm!(asm; =>outer_continue);
        dynasm!(asm; inc r14; jmp =>loop_hdr);
        } // end old path

    } else {
        // ── Level >= 1: hash probe + inner scan ────────────────────────────
        //
        // All relations use contiguous mode: values array holds tuple_idxs sequentially.
        // Inner loop: j = 0..count, load values[head + j]
        // No pointer-chasing — sequential scan enables CPU speculation/prefetch.

        let arity = clause.args.len();
        let stride = (arity * 4) as i32;
        let use_recent = p.use_recent[level];
        let is_rec = p.is_rec[level];
        // Non-recursive relations use contiguous mode (built once for EDB).
        // Recursive relations use linked-list mode (not supported in native path).
        let is_contiguous = !is_rec;

        // Native path only supports contiguous (EDB) inner clauses.
        // Reject strata where any inner-level clause is recursive (linked-list).
        if p.use_jit_native && !is_contiguous {
            return Err(format!(
                "asm native: rule {} clause {} is recursive (linked-list not supported in native path)",
                p.rule_i, level
            ));
        }

        // Save the ENCLOSING level's r15 and rbx before we clobber them with
        // the new probe.  The enclosing level is `level-1`.
        // When level == 1 this is the first probe; level-1's data (r12/r13/r14)
        // is not clobbered, so no save is needed.
        // When level >= 2 we must save level-1's r15/rbx.
        //
        // IMPORTANT: this save must happen BEFORE any fast-path early returns
        // (e.g. emit_tuple_set_probe below), because the sub_exhausted label
        // emitted by the enclosing level always restores from these slots
        // regardless of which path was taken at this level.
        if level >= 2 {
            let save_level = level - 1;
            let node_slot = level_node_save_slot(save_level, p.var_count, p.max_head_arity, p.max_depth);
            let dptr_slot = level_dptr_save_slot(save_level, p.var_count, p.max_head_arity, p.max_depth);
            dynasm!(asm
                ; mov [rbp + node_slot], r15
                ; mov [rbp + dptr_slot], rbx
            );
        }

        // flat handle index for this (level, use_recent)
        let flat_idx = p.rule_handle_base + level * 2 + if use_recent { 1 } else { 0 };
        let flat_idx_i32 = (flat_idx * 8) as i32;
        // For tuple_set probes on the native path, we use the total version (always non-recent).
        let total_flat_idx = p.rule_handle_base + level * 2; // use_recent=false
        let total_flat_idx_i32 = (total_flat_idx * 8) as i32;
        let _ = total_flat_idx_i32; // suppress unused warning in old path

        // Fast path for fully-bound existence checks: inline JitTupleSet probe.
        if clause.fresh_cols.is_empty() {
            let handle_idx_in_buf = flat_idx;
            let native_total_for_probe = if p.use_jit_native { Some(total_flat_idx) } else { None };
            if emit_tuple_set_probe(asm, clause, handle_idx_in_buf, when_exhausted, p.var_locs, native_total_for_probe).is_ok() {
                // Probe succeeded (emitted inline code). Fall through means found.
                if level + 1 == p.clauses.len() {
                    for expr in p.rule_conds {
                        emit_expr(asm, expr, p.var_locs)?;
                        dynasm!(asm; test eax, eax; jz =>when_exhausted);
                    }
                    emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base)?;
                    dynasm!(asm; jmp =>when_exhausted);
                } else {
                    emit_clause_level(asm, p, level + 1, outer_exit, when_exhausted)?;
                }
                return Ok(());
            }
            // If emit_tuple_set_probe returned Err (e.g. unsupported arity), fall through
            // to the existing col-index + scan path.
        }

        // Compute probe key into ecx (supports Var, Literal, or arbitrary expression).
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
            CClauseArg::Expr(expr) => {
                emit_expr(asm, expr, p.var_locs)?;
                dynasm!(asm; mov ecx, eax);
            }
        }

        if p.use_jit_native {
            // ── Native path: inline JitColIndex probe ────────────────────────
            //
            // JitRelData offsets:
            //   data       @ 0
            //   len        @ 8
            //   cap        @ 16
            //   col_indices @ 24  (*mut *mut JitColIndex, array of arity ptrs)
            //   tuple_set  @ 32
            //   arity      @ 56
            //
            // JitColIndex offsets:
            //   keys   @ 0   (*mut u32)
            //   ranges @ 8   (*mut u64)
            //   vals   @ 16  (*mut u32)
            //   mask   @ 24  (u32)
            //   len    @ 28  (u32)
            //
            // Hash probe: slot = (key * 0x9e3779b9) & mask   (Knuth multiplicative)
            // EMPTY_KEY = 0xFFFFFFFF
            //
            // After probe:
            //   esi = start (lo32 of ranges[slot])
            //   eax = count (hi32 of ranges[slot])
            //   r11 = vals ptr (JitColIndex.vals)
            //   rdi = *mut JitRelData for this scan slot (for data ptr later)

            // Load scan_rel *mut JitRelData
            dynasm!(asm
                ; mov rdi, [rbp + CTX_SLOT]         // native ctx
                ; mov rdi, [rdi]                    // scan_rels (offset 0)
                ; mov rdi, [rdi + flat_idx_i32]     // *mut JitRelData for this clause
            );

            // Load col_indices ptr and get JitColIndex* for primary_col
            let primary_col_i32 = (primary_col * 8) as i32;
            dynasm!(asm
                ; mov r8, [rdi + 24]                // col_indices: *mut *mut JitColIndex
                ; mov r8, [r8 + primary_col_i32]    // JitColIndex* for primary_col
            );

            // Hash: slot = (key * KNUTH32) & mask, key is in ecx
            dynasm!(asm
                ; mov eax, ecx
                ; imul eax, eax, 0x9e3779b9u32 as i32  // Knuth hash (same as col_hash in storage.rs)
                ; mov edx, [r8 + 24]                   // JitColIndex.mask (offset 24, u32)
                ; and eax, edx                         // slot = hash & mask
            );

            // Load keys ptr (JitColIndex.keys is a *mut u32 at offset 0)
            // Note: edx = mask (u32), eax = slot (u32), ecx = key, r8 = *mut JitColIndex
            dynasm!(asm
                ; mov r9, [r8]                  // r9 = keys ptr (*mut u32) at offset 0
            );

            let probe_lp = asm.new_dynamic_label();
            let probe_hit = asm.new_dynamic_label();

            // probe_lp: keys[slot] — r9 is keys ptr, eax is slot
            dynasm!(asm; =>probe_lp);
            dynasm!(asm
                ; mov esi, [r9 + rax*4]         // keys[slot]
                ; cmp esi, -1i32                // EMPTY_KEY = 0xFFFFFFFF = -1
                ; je =>when_exhausted           // not found → exhaust level
                ; cmp esi, ecx
                ; je =>probe_hit
                ; inc eax                       // slot++
                ; and eax, edx                  // slot & mask (edx still = mask)
                ; jmp =>probe_lp
            );

            dynasm!(asm; =>probe_hit);
            // Load ranges[slot]: u64 = start | (count << 32)
            // ranges is at offset 8 in JitColIndex
            dynasm!(asm
                ; mov r11, [r8 + 8]             // ranges ptr (offset 8)
                ; mov r11, [r11 + rax*8]        // ranges[slot] as u64
                ; mov esi, r11d                 // start = lo32
                ; shr r11, 32
                ; mov eax, r11d                 // count = hi32
            );
            // Load vals ptr from JitColIndex into r11 (for vals_base computation below)
            dynasm!(asm
                ; mov r11, [r8 + 16]            // vals ptr (offset 16 in JitColIndex)
            );
            // rdi still = *mut JitRelData for this scan slot
        } else {
        // ── Old path: probe JitLookupHandle (JitHashIndex) ──────────────────

        // Probe: handle_off uses (clause_seq * 2 + use_recent) * 24 (JitLookupHandle stride)
        let handle_off: i32 = ((level * 2 + if use_recent { 1 } else { 0 }) * 24) as i32;
        load_rule_ctx!(asm, p.rule_i);
        dynasm!(asm
            ; mov rax, [rax + 24i8]   // lookup_handles ptr
            ; add rax, handle_off
            ; mov r8,  [rax]          // entries_ptr (offset  0)
            ; mov r11, [rax + 8i8]   // values_ptr  (offset  8)
            ; mov r10d, [rax + 16i8] // mask        (offset 16)
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

        // probe_loop: load entries_ptr[slot*16].key (offset 0, u32)
        // JitIndexEntry: key@0, head@4, count@8, _pad@12 (16 bytes total)
        // x86 doesn't support *16 scale; use shl+add to compute slot*16.
        dynasm!(asm; =>probe_lp);
        dynasm!(asm
            ; mov rax, rdx
            ; shl rax, 4              // rax = slot * 16
            ; add rax, r8             // rax = entries_ptr + slot*16
            ; mov esi, [rax]          // esi = entry.key (offset 0)
            ; cmp esi, -1i32
            ; je =>probe_ovf
            ; cmp esi, ecx
            ; je =>probe_ok
            ; inc rdx
            ; and rdx, r10
            ; jmp =>probe_lp
        );

        // probe_ok: entry.head @ offset 4, entry.count @ offset 8
        dynasm!(asm; =>probe_ok);
        dynasm!(asm
            ; mov rax, rdx
            ; shl rax, 4
            ; add rax, r8             // rax = entries_ptr + slot*16
            ; mov esi, [rax + 4i8]   // esi = entry.head (start)
            ; mov eax, [rax + 8i8]   // eax = entry.count
            ; jmp =>after_probe
        );

        // probe_ovf: overflow slot at entries_ptr[(mask+1)*16]
        dynasm!(asm; =>probe_ovf);
        dynasm!(asm
            ; lea rdx, [r10 + 1]     // overflow slot index = mask + 1
            ; mov rax, rdx
            ; shl rax, 4
            ; add rax, r8             // rax = entries_ptr + (mask+1)*16
            ; mov esi, [rax]          // entry.key
            ; cmp esi, ecx
            ; jne =>when_exhausted   // key not found → exhaust this level
            ; mov esi, [rax + 4i8]   // entry.head (start)
            ; mov eax, [rax + 8i8]   // entry.count
        );

        dynasm!(asm; =>after_probe);
        } // end old path hash probe

        // Dispatch to contiguous scan or linked-list traversal.
        //   esi = entry.head (start index into values array, or first node index)
        //   eax = entry.count (contiguous) — unused for linked-list
        //   r11 = values_ptr
        //
        // Contiguous (EDB / non-recursive):
        //   values[head+j] = col_value (arity-2) or tuple_idx (other arity)
        //   r15d = j counter, vals_base saved to level_vptr_slot
        //
        // Linked-list (IDB / recursive):
        //   Node layout: stride 8 bytes
        //     node[0] = col_value (arity-2) or tuple_idx (arity > 2)
        //     node[4] = next node index (u32; 0xFFFFFFFF = end)
        //   r15d = current node index; advance to next BEFORE inner body
        //   so that level+1's entry-save captures the next-node value in r15.

        if is_contiguous {
            // ── Contiguous sequential scan ────────────────────────────────────
            let is_col_value = arity == 2;

            let vs = level_vptr_slot(level, p.var_count, p.max_head_arity);

            dynasm!(asm
                ; test eax, eax
                ; jz =>when_exhausted
                ; xor r15d, r15d         // j = 0
            );
            // Pre-compute vals_base = values_ptr + head*4.
            // For col_value: keep count in rbx.
            // For standard: spill count to cnt_slot, load data_ptr into rbx.
            if is_col_value {
                dynasm!(asm
                    ; mov rbx, rax                    // rbx = count
                    ; lea r11, [r11 + rsi*4]          // r11 = vals_base
                    ; mov [rbp + vs], r11             // save vals_base
                );
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov [rbp + cnt_slot], eax       // save count to stack
                    ; lea r11, [r11 + rsi*4]          // r11 = vals_base
                    ; mov [rbp + vs], r11             // save vals_base
                );
                if p.use_jit_native {
                    // Native path: load data_ptr directly from JitRelData.data (offset 0)
                    // rdi still = *mut JitRelData for this scan slot (set during hash probe)
                    dynasm!(asm; mov rbx, [rdi]);  // JitRelData.data @ 0
                } else {
                    load_rel_rdi!(asm, p.rule_i, level);
                    call_abs!(asm, p.pdptr);
                    dynasm!(asm; mov rbx, rax);
                }
            }

            let inner_hdr = asm.new_dynamic_label();
            let inner_continue = asm.new_dynamic_label();

            dynasm!(asm; =>inner_hdr);
            if is_col_value {
                dynasm!(asm; cmp r15d, ebx; jge =>when_exhausted);
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov eax, [rbp + cnt_slot]
                    ; cmp r15d, eax
                    ; jge =>when_exhausted
                );
            }

            // Load vals_base[j] into ecx.
            dynasm!(asm
                ; mov rax, [rbp + vs]                   // vals_base
                ; mov ecx, [rax + r15*4]                // vals_base[j]
                ; inc r15d                              // j++
            );

            if is_col_value {
                let primary_col = clause.bound_cols[0];
                let free_col = 1 - primary_col;
                emit_bind_col_value(asm, clause, free_col, inner_continue, p.var_locs)?;
            } else {
                dynasm!(asm
                    ; imul rcx, rcx, stride
                    ; add rcx, rbx
                    ; mov rax, rcx
                );
                emit_bind_cols(asm, clause, inner_continue, p.var_locs)?;
            }

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
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base)?;
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
            // ── Linked-list traversal (IDB / recursive) ──────────────────────
            //   esi = first node index (entry.head); 0xFFFFFFFF = not found
            //   r11 = values_ptr — CALLER-SAVED; must be saved to stack and
            //         reloaded at the top of each iteration because calls inside
            //         the body (packed_try_insert, etc.) may clobber r11.
            //   r15d = current node index (callee-saved, survives calls)
            //
            // r15 is advanced to next-node BEFORE processing each node's body.
            // When level+1 entry code saves r15, it captures next-node, so
            // sub_exhausted restores r15 = next-node and the loop resumes
            // correctly.
            let is_ll_col_value = arity == 2;

            // Save values_ptr to stack before any call that could clobber r11.
            let vs = level_vptr_slot(level, p.var_count, p.max_head_arity);
            dynasm!(asm; mov [rbp + vs], r11);

            // r15d = head (first node; r15 is callee-saved, survives pdptr call)
            dynasm!(asm; mov r15d, esi);

            // For tuple_idx mode (arity > 2), load data_ptr into rbx once.
            // (r11 already saved above so the call is safe.)
            if !is_ll_col_value {
                if p.use_jit_native {
                    // Native path: reload scan_rels[flat_idx]->data (offset 0)
                    dynasm!(asm
                        ; mov rbx, [rbp + CTX_SLOT]         // native ctx
                        ; mov rbx, [rbx]                    // scan_rels (offset 0)
                        ; mov rbx, [rbx + flat_idx_i32]     // *mut JitRelData
                        ; mov rbx, [rbx]                    // JitRelData.data @ 0
                    );
                } else {
                    load_rel_rdi!(asm, p.rule_i, level);
                    call_abs!(asm, p.pdptr);
                    dynasm!(asm; mov rbx, rax);
                }
            }

            let inner_hdr = asm.new_dynamic_label();
            let inner_continue = asm.new_dynamic_label();

            dynasm!(asm; =>inner_hdr);
            // Sentinel check: 0xFFFFFFFF = end of chain.
            dynasm!(asm; cmp r15d, -1i32; je =>when_exhausted);

            // Reload values_ptr (may have been clobbered by calls in the body).
            // Load node: node_addr = values_ptr + r15 * 8 (stride = 8 bytes)
            // node[0] = v0 (col_value or tuple_idx), node[4] = next node index
            dynasm!(asm
                ; mov r11, [rbp + vs]  // reload values_ptr
                ; mov eax, r15d
                ; shl rax, 3           // * 8
                ; add rax, r11         // rax = node_addr
                ; mov ecx, [rax]       // ecx = v0
                ; mov r15d, [rax + 4]  // r15d = next (advance BEFORE body)
            );

            if is_ll_col_value {
                let primary_col = clause.bound_cols[0];
                let free_col = 1 - primary_col;
                emit_bind_col_value(asm, clause, free_col, inner_continue, p.var_locs)?;
            } else {
                // ecx = tuple_idx → tuple_ptr = rbx + ecx * stride
                dynasm!(asm
                    ; imul rcx, rcx, stride
                    ; add rcx, rbx
                    ; mov rax, rcx
                );
                emit_bind_cols(asm, clause, inner_continue, p.var_locs)?;
            }

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
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base)?;
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
        }
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
    rule_handle_base: usize,
    rule_head_base: usize,
    use_jit_native: bool,
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
        emit_heads(asm, heads, rule_i, var_count, max_head_arity, pti, &var_locs, use_jit_native, rule_head_base)?;
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
        rule_handle_base,
        rule_head_base,
        use_jit_native,
    };

    emit_clause_level(asm, &p, 0, variant_exit, variant_exit)?;

    dynasm!(asm; =>variant_exit);
    Ok(())
}

// ─── Main entry point ─────────────────────────────────────────────────────

/// Common codegen logic for both old and native paths.
///
/// When `use_jit_native=true`:
///   - CTX_SLOT holds `*mut StratumStage4NativeCtx` instead of `*mut StratumStage4Ctx`.
///   - `advance_addr` is the address of `jit_advance_native`.
///   - Loop body reads scan data directly from `JitRelData` fields.
///
/// When `use_jit_native=false` (default):
///   - CTX_SLOT holds `*mut StratumStage4Ctx`.
///   - `advance_addr` is the address of `jit_stratum_advance_s4`.
///   - Loop body uses `packed_count` / `packed_data_ptr` / `packed_recent_ptr` callbacks.
#[allow(clippy::too_many_arguments)]
fn codegen_stratum_asm_inner(
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    var_count: usize,
    advance_addr: usize,
    packed_try_insert_addr: usize,
    packed_count_addr: usize,
    packed_data_ptr_addr: usize,
    packed_recent_ptr_addr: usize,
    use_jit_native: bool,
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
        if let Some(c0) = clauses.first()
            && !c0.bound_cols.is_empty()
        {
            return Err(format!("asm: rule {ri} clause0 has bound_cols; unsupported"));
        }
        // clauses 1..N must have at least one bound col for index probe
        for (ci, c) in clauses.iter().enumerate().skip(1) {
            if c.bound_cols.is_empty() {
                return Err(format!("asm: rule {ri} clause{ci} has no bound_cols; unsupported"));
            }
        }
        // No recursive rejection: linked-list traversal is now supported.
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

    // Compute per-rule base index into the flat handles_buf / tuple_sets_buf.
    // Each rule contributes clause_count * 2 entries (full + recent per clause).
    let rule_handle_bases: Vec<usize> = {
        let mut bases = Vec::with_capacity(rules.len());
        let mut offset = 0usize;
        for (clauses, _, _) in rules.iter() {
            bases.push(offset);
            offset += clauses.len() * 2;
        }
        bases
    };

    // Each rule contributes head_count entries to the flat head_specs / head_rels arrays.
    let rule_head_bases: Vec<usize> = {
        let mut bases = Vec::with_capacity(rules.len());
        let mut offset = 0usize;
        for (_, heads, _) in rules.iter() {
            bases.push(offset);
            offset += heads.len();
        }
        bases
    };

    // In the native path, scan_rels pointers may be stale (pointing into a JitNativeRelData
    // that was rebuilt by a previous advance_jit call between cache creation and first invocation).
    // Call advance once upfront to refresh them before the full variants run.
    if use_jit_native {
        dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
        call_abs!(asm, advance_addr);
        // Don't check al here — full variants must run regardless of whether anything changed,
        // since we need the total-scan to process all existing facts on the first evaluation.
    }

    // Full variants
    for (rule_i, (clauses, heads, conds)) in rules.iter().enumerate() {
        emit_rule_variant(
            &mut asm, rule_i, clauses, heads, conds, None,
            var_count, max_head_arity, max_depth,
            packed_try_insert_addr, packed_count_addr,
            packed_data_ptr_addr, packed_recent_ptr_addr,
            rule_handle_bases[rule_i],
            rule_head_bases[rule_i],
            use_jit_native,
        )?;
    }

    // advance(ctx) → al
    dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
    call_abs!(asm, advance_addr);
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
                rule_handle_bases[rule_i],
                rule_head_bases[rule_i],
                use_jit_native,
            )?;
        }
    }

    dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
    call_abs!(asm, advance_addr);
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

/// Old (non-native) entry point: uses `packed_count`/`packed_data_ptr`/`packed_recent_ptr`
/// callbacks and the `StratumStage4Ctx` context.
pub fn codegen_stratum_asm(
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    var_count: usize,
    advance_s4_addr: usize,
    packed_try_insert_addr: usize,
    packed_count_addr: usize,
    packed_data_ptr_addr: usize,
    packed_recent_ptr_addr: usize,
) -> Result<AsmStratum, String> {
    codegen_stratum_asm_inner(
        rules, var_count,
        advance_s4_addr,
        packed_try_insert_addr,
        packed_count_addr,
        packed_data_ptr_addr,
        packed_recent_ptr_addr,
        false,
    )
}

/// Native entry point: reads scan data directly from `JitRelData` fields (Step 4).
///
/// The generated function takes `*mut StratumStage4NativeCtx` as its argument.
/// `advance_native_addr` is the address of `jit_advance_native`.
/// `packed_try_insert_addr` is still used for head insertion (write side; Step 5 will replace this).
pub fn codegen_stratum_asm_native(
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    var_count: usize,
    advance_native_addr: usize,
    packed_try_insert_addr: usize,
) -> Result<AsmStratum, String> {
    codegen_stratum_asm_inner(
        rules, var_count,
        advance_native_addr,
        packed_try_insert_addr,
        0, // packed_count_addr unused in native path
        0, // packed_data_ptr_addr unused in native path
        0, // packed_recent_ptr_addr unused in native path
        true,
    )
}
