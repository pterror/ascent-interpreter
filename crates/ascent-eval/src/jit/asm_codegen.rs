//! x86-64 JIT backend for Stage 4 stratum functions using dynasmrt.
//!
//! Supports rules with up to 2 `CClause` body items where clause 0 is a full
//! scan (no bound cols) and clause 1 is an index scan.  Returns `Err` for any
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
//!   [rbp-52-V*4-H*4]    values_ptr spill               vptr_slot(V,H)
//! ```
//! where V = var_count, H = max_head_arity.
//!
//! # Register use inside loop bodies
//! All callee-saved registers survive `call` instructions:
//!   r12 = outer loop count
//!   r13 = recent_ptr for outer clause  (only if use_recent0)
//!   r14 = outer loop counter i
//!   r15 = inner linked-list node (current; zero-extended u32)
//!   rbx = inner data_ptr (pre-fetched before inner loop, or re-fetched if recursive)
//!
//! Caller-saved (clobbered by `call`): rax, rcx, rdx, rsi, rdi, r8-r11.

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
/// `is_two_clause`: if false (1-clause rule) both r13 and rbx are free.
///
/// Returns a Vec indexed by variable ID.
fn compute_var_locs(
    var_count: usize,
    outer_fresh_ids: &[u32],
    use_recent0: bool,
    is_two_clause: bool,
) -> Vec<VarLoc> {
    // Start with everything on the stack.
    let mut locs: Vec<VarLoc> = (0..var_count as u32).map(|id| VarLoc::Stack(var_slot(id))).collect();

    // Build the list of available registers, priority-ordered.
    let mut available: Vec<u8> = Vec::new();

    if !use_recent0 {
        // r13 is free when we don't need it for the outer recent_ptr.
        available.push(13);
    }
    if !is_two_clause {
        // For 1-clause rules, rbx is never used for tuple_idx/data_ptr in a
        // way that overlaps with variable lifetime, so it's free.
        available.push(3);
    }
    // Note: for 2-clause rules rbx is repurposed for tuple_idx then data_ptr
    // inside the outer/inner body, so assigning a variable to rbx would
    // clobber it — exclude rbx from the 2-clause register pool.

    // Assign registers to outer-stable variables in ID order.
    let mut reg_iter = available.into_iter();
    for &var_id in outer_fresh_ids {
        if let Some(reg) = reg_iter.next() {
            locs[var_id as usize] = VarLoc::Reg(reg);
        } else {
            break; // no more registers; remainder spill to stack
        }
    }

    locs
}

fn vptr_slot(var_count: usize, max_head_arity: usize) -> i32 {
    // 8 bytes below the head-col region base (= head_col_slot(vc, mha, 0) - 8)
    -52 - (var_count as i32) * 4 - (max_head_arity as i32) * 4 - 8
}

/// Compute `sub rsp, FRAME_SIZE` value.  Must be ≡ 8 (mod 16) because after
/// 5 callee-saved pushes rsp ≡ 8 (mod 16), and we need rsp ≡ 0 for calls.
fn frame_size(var_count: usize, max_head_arity: usize) -> i32 {
    // Bytes below [rbp-40] (callee-save push zone):
    //   8   = ctx slot   (at -48, occupies [-48..-41])
    //   V*4 = variable slots
    //   H*4 = head-tuple scratch (ascending; col[0] at lowest address)
    //   8   = values_ptr spill  (8 bytes below the head-col base)
    //   8   = guard / alignment slack to ensure the 8-byte vptr store fits
    //         within the frame for all (V, H) combinations.
    let raw = 8 + var_count * 4 + max_head_arity * 4 + 16;
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
/// Note: parameter renamed from `cs` to avoid conflict with the CS segment register.
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
///
/// Head cols are stored in **ascending** address order so that the contiguous
/// slice `[head_col_slot(vc, mha, 0) .. +arity*4]` can be passed directly to
/// `packed_try_insert`.  The head region lives just below the var slots and
/// above the `vptr` spill slot.
fn head_col_slot(var_count: usize, max_head_arity: usize, col: usize) -> i32 {
    // base = lowest address in head region (= col 0)
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
            // Evaluate b, push to stack; evaluate a; pop b into ecx.
            // push/pop are balanced; no calls between them, so alignment OK.
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
    // Check literal equality constraints
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
    // Check variable-bound cols
    for &col in &clause.bound_cols {
        if let CClauseArg::Var(var_id) = &clause.args[col] {
            dynasm!(asm; mov edx, [rax + (col as i32)*4]);
            emit_load_var_ecx(asm, var_locs, *var_id);
            dynasm!(asm; cmp edx, ecx; jne =>skip);
        }
    }
    // Bind fresh cols
    for &(col, var_id) in &clause.fresh_cols {
        dynasm!(asm; mov edx, [rax + (col as i32)*4]);
        emit_store_var(asm, var_locs, var_id);
    }
    Ok(())
}

// ─── Head emission ────────────────────────────────────────────────────────

/// Build and insert all heads: inline dedup probe + `packed_try_insert`.
/// Clobbers all caller-saved regs.
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

        // Build candidate tuple on stack (ascending addresses: col0 at base, col1 at base+4…)
        for (col, arg) in head.args.iter().enumerate() {
            emit_expr(asm, arg, var_locs)?;
            dynasm!(asm; mov [rbp + head_col_slot(var_count, max_head_arity, col)], eax);
        }

        // Load head_rel from head_rels[hi]
        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 16i8]     // head_rels ptr
            ; mov r8, [rax + hoff]      // r8 = head_rel (*mut PackedStorage)
        );

        if arity == 0 {
            dynasm!(asm; mov rdi, r8; xor rsi, rsi; xor edx, edx);
            call_abs!(asm, pti);
            continue;
        }

        // t0_off = col 0 base (ascending layout → packed_try_insert reads correctly)
        let t0_off = head_col_slot(var_count, max_head_arity, 0);

        // Load dedup handle: head_dedup_handles[hi] → r10 = *mut JitDedupHandle
        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 32i8]     // head_dedup_handles ptr
            ; mov r10, [rax + hoff]     // r10 = *mut JitDedupHandle
            ; mov r11, [r10]            // r11 = entries ptr
            ; mov r10d, [r10 + 8i8]    // r10d = cap (u32; zero-extends to r10)
        );

        let call_insert = asm.new_dynamic_label();
        let after_emit = asm.new_dynamic_label();

        // If cap==0 (empty table): skip probe, call insert directly
        dynasm!(asm; test r10d, r10d; jz =>call_insert);

        // Compute polynomial hash: h=0; for each col: h = h*0x9e3779b9 + col_val
        dynasm!(asm; xor eax, eax);
        for col in 0..arity {
            dynasm!(asm
                ; imul eax, eax, 0x9e3779b9u32 as i32
                ; add eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
            );
        }
        // Remap 0xFFFFFFFF → 0xFFFFFFFE
        dynasm!(asm; cmp eax, -1i32; jne >no_remap; dec eax; no_remap:);

        // mask = cap-1; slot = hash & mask
        dynasm!(asm
            ; lea rcx, [r10 - 1]   // rcx = mask (64-bit, zero-extended from r10d-1)
            ; mov edx, eax         // rdx = hash (zero-extended via 32-bit write)
            ; and rdx, rcx         // rdx = slot
        );
        // r11=entries, rdx=slot, eax=hash, rcx=mask

        let stride_d = ((arity + 1) * 4) as i32;
        let probe_lp = asm.new_dynamic_label();
        let probe_nx = asm.new_dynamic_label();

        dynasm!(asm; =>probe_lp);
        dynasm!(asm
            ; imul rsi, rdx, stride_d  // rsi = slot*stride
            ; add rsi, r11             // rsi = entry_ptr
            ; mov edi, [rsi]           // entry_hash
            ; cmp edi, -1i32          // JITDEDUP_EMPTY?
            ; je =>call_insert
            ; cmp edi, eax            // hash match?
            ; jne =>probe_nx
        );
        // Verify all data fields
        for col in 0..arity {
            dynasm!(asm
                ; mov edi, [rsi + ((1+col)*4) as i32]
                ; mov r8d, [rbp + head_col_slot(var_count, max_head_arity, col)]
                ; cmp edi, r8d
                ; jne =>probe_nx
            );
        }
        dynasm!(asm; jmp =>after_emit);  // duplicate: skip insert

        dynasm!(asm; =>probe_nx);
        dynasm!(asm; inc rdx; and rdx, rcx; jmp =>probe_lp);

        // call_insert: packed_try_insert(head_rel, tuple_ptr, arity)
        dynasm!(asm; =>call_insert);
        // Reload head_rel (r8 may have been overwritten by field-verify loop)
        load_rule_ctx!(asm, rule_i);
        dynasm!(asm
            ; mov rax, [rax + 16i8]
            ; mov rdi, [rax + hoff]   // head_rel
            ; lea rsi, [rbp + t0_off] // tuple_ptr (re-derive; probe loop doesn't clobber rbp)
            ; mov edx, arity as i32
        );
        call_abs!(asm, pti);

        dynasm!(asm; =>after_emit);
    }
    Ok(())
}

// ─── Hash probe + linked-list traversal ──────────────────────────────────

/// Emit an inline index probe and linked-list traversal for `clause` at
/// `clause_seq`.  Precondition: `ecx` holds the probe key (u32).
///
/// After the probe, sets:
///   r15 = initial linked-list node (SENTINEL = 0xFFFF_FFFF = no results)
///   rbx = inner data_ptr (if !is_recursive; else undefined — caller re-fetches)
///
/// Saves values_ptr to the vptr stack slot for use inside the inner loop.
/// Jumps to `no_match` if the probe finds no entries.
#[allow(clippy::too_many_arguments)]
fn emit_probe_setup(
    asm: &mut Assembler,
    rule_i: usize,
    clause_seq: usize,
    use_recent: bool,
    is_recursive: bool,
    var_count: usize,
    max_head_arity: usize,
    pdptr_addr: usize,
    no_match: DynamicLabel,
) -> Result<(), String> {
    // Load lookup handle: entries_ptr → r8, values_ptr → r9, mask → r10
    let handle_off: i32 = ((clause_seq * 2 + if use_recent { 1 } else { 0 }) * 24) as i32;
    load_rule_ctx!(asm, rule_i);
    dynasm!(asm
        ; mov rax, [rax + 24i8]          // lookup_handles ptr
        ; add rax, handle_off
        ; mov r8,  [rax]                 // entries_ptr
        ; mov r9,  [rax + 8i8]          // values_ptr
        ; mov r10d, [rax + 16i8]        // mask (u32 → r10)
    );
    // ecx = key (u32)

    // Hash: slot = (key * KNUTH32) & mask
    dynasm!(asm
        ; mov eax, ecx
        ; imul eax, eax, 0x9e3779b1u32 as i32   // u32 wrapping multiply
        ; mov edx, eax                            // rdx = hash (zero-extended via 32-bit write)
        ; and rdx, r10                            // rdx = slot
    );

    // Open-addressed probe loop
    let probe_lp = asm.new_dynamic_label();
    let probe_ok = asm.new_dynamic_label();
    let probe_ovf = asm.new_dynamic_label();
    let after_probe = asm.new_dynamic_label();

    dynasm!(asm; =>probe_lp);
    // entry_ptr = entries + slot*16 (JitIndexEntry = 16 bytes)
    dynasm!(asm
        ; lea rax, [rdx + rdx]    // rdx*2
        ; shl rax, 3              // rdx*16
        ; add rax, r8             // entry_ptr
        ; mov esi, [rax]          // entry_key
        ; cmp esi, -1i32          // EMPTY_KEY?
        ; je =>probe_ovf
        ; cmp esi, ecx
        ; je =>probe_ok
        ; inc rdx
        ; and rdx, r10
        ; jmp =>probe_lp
    );

    dynasm!(asm; =>probe_ok);
    dynasm!(asm; mov esi, [rax + 4i8]; jmp =>after_probe);  // head @ offset 4

    dynasm!(asm; =>probe_ovf);
    // Overflow slot: entries[mask+1]
    dynasm!(asm
        ; lea rax, [r10 + 1]
        ; lea rax, [rax + rax]
        ; shl rax, 3
        ; add rax, r8
        ; mov esi, [rax]          // ovf_key
        ; cmp esi, ecx
        ; jne =>no_match          // no entry for this key
        ; mov esi, [rax + 4i8]   // ovf head
    );

    dynasm!(asm; =>after_probe);
    // esi = head node; check SENTINEL
    dynasm!(asm; cmp esi, -1i32; je =>no_match; mov r15d, esi);

    // Save values_ptr to vptr_slot (r9 will be clobbered by calls inside the loop)
    let vs = vptr_slot(var_count, max_head_arity);
    dynasm!(asm; mov [rbp + vs], r9);

    // Pre-fetch inner data_ptr into rbx (if non-recursive; otherwise caller re-fetches)
    if !is_recursive {
        // r8/r9/r10 are set; loading data_ptr clobbers them (caller-saved).
        // We only need r9 (values_ptr) saved to stack already — so clobbering r9 now is fine.
        load_rel_rdi!(asm, rule_i, clause_seq);
        call_abs!(asm, pdptr_addr);
        dynasm!(asm; mov rbx, rax);
    }

    Ok(())
}

/// Emit the linked-list traversal loop body once per node.
/// Preconditions:
///   r15 = current node (u32, zero-extended)
///   rbx = data_ptr for this clause (if !is_recursive; else re-fetched)
///   vptr_slot holds values_ptr
///
/// After the body: jumps to `inner_continue` (which then loads next node from r15 and loops).
/// When node == SENTINEL: jumps to `no_match` (same as outer_continue for 2-clause rules).
#[allow(clippy::too_many_arguments)]
fn emit_inner_loop(
    asm: &mut Assembler,
    rule_i: usize,
    clause_seq: usize,
    clause: &CClause,
    heads: &[CHeadClause],
    rule_conds: &[CExpr],
    use_recent: bool,
    _is_recursive: bool,
    var_count: usize,
    max_head_arity: usize,
    pdptr_addr: usize,
    pti_addr: usize,
    no_match: DynamicLabel,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    let _ = use_recent; // only used in emit_probe_setup label selection, not here
    let arity = clause.args.len();
    let stride = (arity * 4) as i32;
    let vs = vptr_slot(var_count, max_head_arity);

    let inner_hdr = asm.new_dynamic_label();
    let inner_continue = asm.new_dynamic_label();

    dynasm!(asm; =>inner_hdr);
    dynasm!(asm; cmp r15d, -1i32; je =>no_match);

    // Load tuple_idx (eax) and next_node (r15) from values_ptr[r15*8]
    // values_ptr is in the vptr stack slot (r9 may be clobbered elsewhere)
    dynasm!(asm
        ; mov rax, [rbp + vs]           // rax = values_ptr
        ; mov ecx, [rax + r15*8]        // tuple_idx (u32 → ecx; zero-extends to rcx)
        ; mov esi, [rax + r15*8 + 4i8]  // next_node (u32)
        ; mov r15d, esi                  // r15 = next_node (zero-extended via 32-bit write)
        ; mov eax, ecx                   // rax = tuple_idx (zero-extended via 32-bit write)
    );

    // For recursive: re-fetch data_ptr.  tuple_idx in rax must survive.
    // Use aligned push: push rax + sub rsp,8 then call, add rsp,8 + pop rax.
    if _is_recursive {
        dynasm!(asm
            ; push rax      // save tuple_idx; rsp was ≡0, now ≡8
            ; sub rsp, 8    // realign to 0 for the upcoming call
        );
        load_rel_rdi!(asm, rule_i, clause_seq);
        call_abs!(asm, pdptr_addr);
        dynasm!(asm
            ; mov rbx, rax  // rbx = data_ptr
            ; add rsp, 8    // undo realign; rsp back to ≡8
            ; pop rax       // restore tuple_idx; rsp back to ≡0
        );
    }
    // rax = tuple_idx, rbx = data_ptr

    // tuple_ptr = data_ptr + tuple_idx * stride
    dynasm!(asm
        ; imul rax, rax, stride
        ; add rax, rbx          // rax = tuple_ptr
    );

    // Bind cols (rax = tuple_ptr); skip inner_continue on failure
    emit_bind_cols(asm, clause, inner_continue, var_locs)?;

    // Per-clause conditions
    for cond in &clause.conditions {
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, var_locs)?;
            dynasm!(asm; test eax, eax; jz =>inner_continue);
        }
    }

    // Rule-level conditions
    for expr in rule_conds {
        emit_expr(asm, expr, var_locs)?;
        dynasm!(asm; test eax, eax; jz =>inner_continue);
    }

    // Emit heads
    emit_heads(asm, heads, rule_i, var_count, max_head_arity, pti_addr, var_locs)?;

    dynasm!(asm; =>inner_continue);
    // r15 already holds next_node (loaded above); loop back
    dynasm!(asm; jmp =>inner_hdr);

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
    pti: usize,
    pcount: usize,
    pdptr: usize,
    prptr: usize,
) -> Result<(), String> {
    let variant_exit = asm.new_dynamic_label();

    // Compute variable locations for this variant.
    // Outer-stable variables are those first bound by clauses[0].fresh_cols.
    let use_recent0 = recent_idx == Some(0);
    let is_two_clause = clauses.len() == 2;
    let outer_fresh_ids: Vec<u32> = if clauses.is_empty() {
        vec![]
    } else {
        let mut ids: Vec<u32> = clauses[0].fresh_cols.iter().map(|&(_, id)| id).collect();
        ids.sort_unstable();
        ids
    };
    let var_locs = compute_var_locs(var_count, &outer_fresh_ids, use_recent0, is_two_clause);

    match clauses.len() {
        0 => {
            // No scan: emit conditions + heads
            for expr in rule_conds {
                emit_expr(asm, expr, &var_locs)?;
                dynasm!(asm; test eax, eax; jz =>variant_exit);
            }
            emit_heads(asm, heads, rule_i, var_count, max_head_arity, pti, &var_locs)?;
        }

        1 => {
            let clause = &clauses[0];
            let use_recent = recent_idx == Some(0);
            let is_rec = heads.iter().any(|h| h.relation == clause.relation);

            emit_one_clause_scan(
                asm, rule_i, clause, heads, rule_conds,
                use_recent, is_rec, var_count, max_head_arity,
                pti, pcount, pdptr, prptr, variant_exit, &var_locs,
            )?;
        }

        2 => {
            let use_recent1 = recent_idx == Some(1);
            let is_rec0 = heads.iter().any(|h| h.relation == clauses[0].relation);
            let is_rec1 = heads.iter().any(|h| h.relation == clauses[1].relation);

            emit_two_clause_scan(
                asm, rule_i, &clauses[0], &clauses[1], heads, rule_conds,
                use_recent0, use_recent1, is_rec0, is_rec1,
                var_count, max_head_arity,
                pti, pcount, pdptr, prptr, variant_exit, &var_locs,
            )?;
        }

        _ => unreachable!(),
    }

    dynasm!(asm; =>variant_exit);
    Ok(())
}

// ─── One-clause scan ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn emit_one_clause_scan(
    asm: &mut Assembler,
    rule_i: usize,
    clause: &CClause,
    heads: &[CHeadClause],
    rule_conds: &[CExpr],
    use_recent: bool,
    _is_recursive: bool,
    var_count: usize,
    max_head_arity: usize,
    pti: usize,
    pcount: usize,
    pdptr: usize,
    prptr: usize,
    outer_exit: DynamicLabel,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    if !clause.bound_cols.is_empty() {
        return Err("asm: 1-clause rule with bound_cols unsupported".into());
    }
    let arity = clause.args.len();
    let stride = (arity * 4) as i32;
    let uri32 = if use_recent { 1i32 } else { 0i32 };

    // ── Count ─────────────────────────────────────────────────────────────────
    load_rel_rdi!(asm, rule_i, 0usize);
    dynasm!(asm; mov esi, uri32);
    call_abs!(asm, pcount);
    dynasm!(asm; mov r12, rax; test r12, r12; jz =>outer_exit);

    // ── Data/recent ptr ───────────────────────────────────────────────────────
    if use_recent {
        load_rel_rdi!(asm, rule_i, 0usize);
        call_abs!(asm, prptr);
        dynasm!(asm; mov r13, rax); // r13 = recent_ptr (stable)
    }
    // For non-recursive non-recent: could cache data_ptr, but we re-fetch per
    // iteration for uniformity (in L1 cache, cost is negligible).

    // ── Loop ──────────────────────────────────────────────────────────────────
    dynasm!(asm; xor r14d, r14d); // r14 = i = 0

    let loop_hdr = asm.new_dynamic_label();
    let loop_continue = asm.new_dynamic_label();

    dynasm!(asm; =>loop_hdr);
    dynasm!(asm; cmp r14, r12; jge =>outer_exit);

    // Compute tuple_idx: recent → r13[r14]; full → r14
    // Use r15 as temp for tuple_idx (it's callee-saved; inner loop uses it for node
    // but we have no inner loop here)
    if use_recent {
        dynasm!(asm; mov r15, [r13 + r14*8]); // r15 = tuple_idx
    } else {
        dynasm!(asm; mov r15, r14); // r15 = tuple_idx = i
    }

    // Get data_ptr
    load_rel_rdi!(asm, rule_i, 0usize);
    call_abs!(asm, pdptr);
    // rax = data_ptr

    // tuple_ptr = data_ptr + tuple_idx * stride
    dynasm!(asm
        ; imul rcx, r15, stride  // rcx = tuple_idx * stride (r15 is 64-bit)
        ; add rax, rcx           // rax = tuple_ptr
    );

    // Bind cols
    emit_bind_cols(asm, clause, loop_continue, var_locs)?;

    // Clause conditions
    for cond in &clause.conditions {
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, var_locs)?;
            dynasm!(asm; test eax, eax; jz =>loop_continue);
        }
    }

    // Rule conditions
    for expr in rule_conds {
        emit_expr(asm, expr, var_locs)?;
        dynasm!(asm; test eax, eax; jz =>loop_continue);
    }

    // Heads
    emit_heads(asm, heads, rule_i, var_count, max_head_arity, pti, var_locs)?;

    dynasm!(asm; =>loop_continue);
    dynasm!(asm; inc r14; jmp =>loop_hdr);

    // outer_exit is defined by caller (=>variant_exit)
    Ok(())
}

// ─── Two-clause scan ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn emit_two_clause_scan(
    asm: &mut Assembler,
    rule_i: usize,
    clause0: &CClause,
    clause1: &CClause,
    heads: &[CHeadClause],
    rule_conds: &[CExpr],
    use_recent0: bool,
    use_recent1: bool,
    _is_rec0: bool,
    is_rec1: bool,
    var_count: usize,
    max_head_arity: usize,
    pti: usize,
    pcount: usize,
    pdptr: usize,
    prptr: usize,
    outer_exit: DynamicLabel,
    var_locs: &[VarLoc],
) -> Result<(), String> {
    if !clause0.bound_cols.is_empty() {
        return Err("asm: clause0 has bound_cols; unsupported in 2-clause rules".into());
    }
    if clause1.bound_cols.is_empty() {
        return Err("asm: clause1 has no bound_cols; unsupported".into());
    }

    let arity0 = clause0.args.len();
    let stride0 = (arity0 * 4) as i32;
    let uri32 = if use_recent0 { 1i32 } else { 0i32 };

    // ── Outer count ───────────────────────────────────────────────────────────
    load_rel_rdi!(asm, rule_i, 0usize);
    dynasm!(asm; mov esi, uri32);
    call_abs!(asm, pcount);
    dynasm!(asm; mov r12, rax; test r12, r12; jz =>outer_exit);

    // ── Outer recent_ptr ──────────────────────────────────────────────────────
    if use_recent0 {
        load_rel_rdi!(asm, rule_i, 0usize);
        call_abs!(asm, prptr);
        dynasm!(asm; mov r13, rax); // r13 = recent_ptr (stable)
    }

    // ── Outer loop ────────────────────────────────────────────────────────────
    dynasm!(asm; xor r14d, r14d); // r14 = i = 0

    let outer_hdr = asm.new_dynamic_label();
    let outer_continue = asm.new_dynamic_label();

    dynasm!(asm; =>outer_hdr);
    dynasm!(asm; cmp r14, r12; jge =>outer_exit);

    // Compute tuple_idx for clause0 → rbx (temp; callee-saved, so survives the data_ptr call)
    if use_recent0 {
        dynasm!(asm; mov rbx, [r13 + r14*8]); // rbx = tuple_idx from recent
    } else {
        dynasm!(asm; mov rbx, r14); // rbx = tuple_idx = i
    }

    // Get data_ptr for clause0 (re-fetch each outer iteration)
    load_rel_rdi!(asm, rule_i, 0usize);
    call_abs!(asm, pdptr);
    // rax = data_ptr0

    // tuple_ptr0 = data_ptr0 + tuple_idx * stride0
    // rbx = tuple_idx (callee-saved → survived the call ✓)
    dynasm!(asm
        ; imul rcx, rbx, stride0
        ; add rax, rcx   // rax = tuple_ptr0
    );

    // Bind clause0 cols
    emit_bind_cols(asm, clause0, outer_continue, var_locs)?;

    // Clause0 conditions
    for cond in &clause0.conditions {
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, var_locs)?;
            dynasm!(asm; test eax, eax; jz =>outer_continue);
        }
    }

    // ── Inner: compute key for clause1 index probe ────────────────────────────
    let primary_col1 = clause1.bound_cols[0];
    match &clause1.args[primary_col1] {
        CClauseArg::Var(var_id) => {
            emit_load_var_ecx(asm, var_locs, *var_id);
        }
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            dynasm!(asm; mov ecx, *n);
        }
        _ => return Err("asm: clause1 primary bound col has unsupported arg".into()),
    }
    // ecx = key

    // ── Inner: hash probe + setup ─────────────────────────────────────────────
    emit_probe_setup(
        asm, rule_i, 1, use_recent1, is_rec1,
        var_count, max_head_arity, pdptr, outer_continue,
    )?;
    // r15 = initial node; rbx = inner data_ptr (if !is_rec1; else unchanged/garbage)
    // vptr_slot holds values_ptr for clause1

    // ── Inner loop ────────────────────────────────────────────────────────────
    emit_inner_loop(
        asm, rule_i, 1, clause1, heads, rule_conds,
        use_recent1, is_rec1,
        var_count, max_head_arity, pdptr, pti, outer_continue,
        var_locs,
    )?;

    // ── Outer loop advance ────────────────────────────────────────────────────
    dynasm!(asm; =>outer_continue);
    dynasm!(asm; inc r14; jmp =>outer_hdr);

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
        if clauses.len() > 2 {
            return Err(format!("asm: rule {ri} has {} clauses (max 2)", clauses.len()));
        }
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
    }

    let max_head_arity = rules
        .iter()
        .flat_map(|(_, hs, _)| hs.iter().map(|h| h.args.len()))
        .max()
        .unwrap_or(1);
    let max_head_arity = max_head_arity.max(1);

    let frame_sz = frame_size(var_count, max_head_arity);

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
            var_count, max_head_arity,
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
                var_count, max_head_arity,
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
