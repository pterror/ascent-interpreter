//! x86-64 JIT backend for Stage 4 stratum functions using dynasmrt.
//!
//! Supports rules with N `CClause` body items where clause 0 is a full scan
//! (no bound cols) and clauses 1..N-1 are index scans.  Returns `Err` for any
//! unsupported pattern; the caller falls back to the interpreted path.
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
//!   [base - 3*MAX_DEPTH*8 - level*8]      level_count_slot(level) for level 1..MAX_DEPTH (contiguous)
//!   [base - 4*MAX_DEPTH*8 - 8]            data_ptr0_slot  = cached JitRelData.data for clause-0
//!                                          total-scan (avoids 4-load pointer chain per outer iter)
//!   [base - 4*MAX_DEPTH*8 - 16]           native_total_rels_slot   (NativeCtx.total_rels)
//!   [base - 4*MAX_DEPTH*8 - 24]           native_head_rels_slot    (NativeCtx.head_rels)
//!   [base - 4*MAX_DEPTH*8 - 32]           native_head_total_rels_slot (NativeCtx.head_total_rels)
//!   [base - 4*MAX_DEPTH*8 - 36]           simd_mask_slot (4 bytes, SIMD batch element mask)
//!   [base - 4*MAX_DEPTH*8 - 68]           simd_vals_slot (32 bytes, SIMD batch u32 values)
//! ```
//! where V = var_count, H = max_head_arity, MAX_DEPTH = max nesting depth.
//!
//! # Register use inside loop bodies
//! All callee-saved registers survive `call` instructions:
//!   r12 = outer loop end_ptr (use_recent0=true) or count (use_recent0=false)
//!   r13 = outer loop current_ptr (advances by stride; use_recent0=true only)
//!   r14 = outer loop counter i  (use_recent0=false only; free for vars when use_recent0=true)
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

use crate::compiled::{CAggArg, CAggregation, CBinOp, CClause, CClauseArg, CCondition, CExpr, CHeadClause, CUnOp};
use crate::jit::packed_helpers::StratumStage4Fn;
use crate::jit::storage;
use crate::value::Value;

/// 5-tuple representing one rule for asm codegen:
/// `(clauses, heads, conditions, not_clauses, agg_clauses)`.
/// `not_clauses`: negation anti-joins (`aggregator_name == "not"`).
/// `agg_clauses`: real aggregations (count/sum/min/max, pure-aggregation rules only).
pub(crate) type AsmRuleRef<'a> = (&'a [CClause], &'a [CHeadClause], &'a [CExpr], &'a [CAggregation], &'a [CAggregation]);

/// A compiled stratum function produced by the asm backend.
pub struct AsmStratum {
    _buffer: dynasmrt::ExecutableBuffer,
    pub fn_ptr: StratumStage4Fn,
}

// SAFETY: Raw pointers in AsmStratum (fn_ptr, _buffer) are owned and not aliased.
// The struct is created, used, and dropped within a single Engine::run() call,
// and only accessed from the JIT execution thread.
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
/// may appear here (r12 is always loop-machinery; r13/r14 availability depends
/// on use_recent0; r15 is clobbered during inner probes).
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
/// `use_recent0`: if true, the outer loop is pointer-based (r13 = current_ptr,
///   r12 = end_ptr) and r14 is freed for variable assignment.  r13 is NOT
///   available (occupied by the advancing loop pointer).
///   If false, r14 is the loop counter (not available) and r13 is free.
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

    if use_recent0 {
        // Outer loop is pointer-based: r13 = advancing current_ptr (not free),
        // but r14 is freed (no longer used as counter).
        available.push(14);
    } else {
        // r13 is free when we don't need it for the outer current_ptr.
        // r14 is occupied as the counter for the index-based loop.
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
    //   8      = data_ptr0 cache slot (for total-scan outer loop; avoids 4-load chain)
    //   24     = native ctx array-ptr cache: total_rels + head_rels + head_total_rels (3×8)
    //   8      = guard / alignment slack
    let level_bytes = max_depth * 4 * 8;
    // 36 = 4 (simd_mask_slot) + 32 (simd_vals_slot)
    let raw = 8 + var_count * 4 + max_head_arity * 4 + level_bytes + 8 + 24 + 36 + 8;
    // Round up to next value ≡ 8 (mod 16)
    let rem = raw % 16;
    let pad = if rem <= 8 { 8 - rem } else { 24 - rem };
    (raw + pad) as i32
}

/// Stack slot for caching the clause-0 data pointer (JitRelData.data) before the outer
/// loop in the native total-scan path.  Avoids a 4-load pointer chain on every iteration.
/// Placed just below the per-level count-slot region.
fn data_ptr0_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    level_slots_base(var_count, max_head_arity) - 4 * (max_depth as i32) * 8 - 8
}

/// Cached NativeCtx.total_rels pointer (offset 8 in StratumStage4NativeCtx).
/// Precomputed once per rule variant; used by existence-check probes in inner clauses.
fn native_total_rels_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    data_ptr0_slot(var_count, max_head_arity, max_depth) - 8
}

/// Cached NativeCtx.head_rels pointer (offset 24 in StratumStage4NativeCtx).
/// Precomputed once per rule variant; used by head new-buffer writes.
fn native_head_rels_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    data_ptr0_slot(var_count, max_head_arity, max_depth) - 16
}

/// Cached NativeCtx.head_total_rels pointer (offset 56 in StratumStage4NativeCtx).
/// Precomputed once per rule variant; used by cross-iteration dedup probes in head emission.
fn native_head_total_rels_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    data_ptr0_slot(var_count, max_head_arity, max_depth) - 24
}

/// SIMD batch element mask (4 bytes): 8-bit bitmask of elements passing the vectorized filter.
/// Used by emit_col_value_simd_prefix to iterate over passing elements in a batch.
fn simd_mask_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    native_head_total_rels_slot(var_count, max_head_arity, max_depth) - 4
}

/// SIMD batch values (32 bytes): one batch of 8 u32 column values saved to stack.
/// Indexed as [simd_vals_slot + i*4] where i is the element index (0–7).
fn simd_vals_slot(var_count: usize, max_head_arity: usize, max_depth: usize) -> i32 {
    simd_mask_slot(var_count, max_head_arity, max_depth) - 32
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
        VarLoc::Reg(14)     => dynasm!(asm; mov eax, r14d),
        VarLoc::Reg(r)      => panic!("emit_load_var: unexpected reg {r}"),
    }
}

/// Load variable `id` into `ecx` (zero-extends to rcx).
fn emit_load_var_ecx(asm: &mut Assembler, var_locs: &[VarLoc], id: u32) {
    match var_locs[id as usize] {
        VarLoc::Stack(slot) => dynasm!(asm; mov ecx, [rbp + slot]),
        VarLoc::Reg(3)      => dynasm!(asm; mov ecx, ebx),
        VarLoc::Reg(13)     => dynasm!(asm; mov ecx, r13d),
        VarLoc::Reg(14)     => dynasm!(asm; mov ecx, r14d),
        VarLoc::Reg(r)      => panic!("emit_load_var_ecx: unexpected reg {r}"),
    }
}

/// Store `edx` into variable `id`.
fn emit_store_var(asm: &mut Assembler, var_locs: &[VarLoc], id: u32) {
    match var_locs[id as usize] {
        VarLoc::Stack(slot) => dynasm!(asm; mov [rbp + slot], edx),
        VarLoc::Reg(3)      => dynasm!(asm; mov ebx, edx),
        VarLoc::Reg(13)     => dynasm!(asm; mov r13d, edx),
        VarLoc::Reg(14)     => dynasm!(asm; mov r14d, edx),
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
/// - Native path: `Some((flat_idx, total_rels_slot))` — load `total_rels[flat_idx]->tuple_set`
///   where `total_rels` is read from the precomputed stack slot `total_rels_slot`.
/// - Old path: `None` — ctx->tuple_sets_buf[handle_idx] via CTX_SLOT.
///
/// # JitTupleSet layout (from storage.rs)
/// - slots @ 0, mask @ 8, len @ 16
/// - stride = arity + 1 words per slot
/// - slot[0] = hash_tag (0 = empty), slot[1..N] = tuple words
///
/// Scalar (non-SIMD) tuple-set probe against `JitTupleSet` (open-addressed,
/// stride = arity+1).  Used for the non-native path and arity-3 fall-back.
///
/// When `native_probe` is `Some((flat_idx, tr_slot))`, loads the `JitTupleSet`
/// from `total_rels[flat_idx].tuple_set` (offset 32).  When `None`, loads from
/// the old `ctx->tuple_sets_buf[handle_idx]` path.
///
/// Caller must have already loaded `edi=arg0`, `esi=arg1` (arity≥2),
/// `r11d=arg2` (arity=3) before calling.
fn emit_tuple_set_probe_scalar(
    asm: &mut Assembler,
    clause: &CClause,
    handle_idx: usize,
    when_exhausted: DynamicLabel,
    var_locs: &[VarLoc],
    native_probe: Option<(usize, i32)>,
) -> Result<(), String> {
    let arity = clause.args.len();

    macro_rules! load_arg {
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

    // Load JitTupleSet pointer into rdi FIRST (uses rdi which aliases edi).
    if let Some((flat_idx, tr_slot)) = native_probe {
        let flat_idx_i32 = (flat_idx * 8) as i32;
        dynasm!(asm
            ; mov rdi, [rbp + tr_slot]
            ; mov rdi, [rdi + flat_idx_i32]
            ; add rdi, 32              // &JitRelData.tuple_set (offset 32)
        );
    } else {
        let handle_idx_i32 = (handle_idx * 8) as i32;
        dynasm!(asm
            ; mov rdi, [rbp + CTX_SLOT]
            ; mov rdi, [rdi + 56]      // tuple_sets_buf
            ; mov rdi, [rdi + handle_idx_i32]
        );
    }

    // r8 = slots ptr, r9 = mask
    dynasm!(asm; mov r8, [rdi]; mov r9, [rdi + 8]);

    // Load args AFTER rdi (safe: rdi no longer needed as pointer).
    load_arg!(asm, 0, edi);
    if arity >= 2 { load_arg!(asm, 1, esi); }
    if arity >= 3 { load_arg!(asm, 2, r11d); }

    // Compute tuple_hash
    dynasm!(asm; mov eax, 0x9e3779b9u32 as i32);
    dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, edi);
    if arity >= 2 { dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, esi); }
    if arity >= 3 { dynasm!(asm; imul eax, eax, 0x9e3779b9u32 as i32; add eax, r11d); }
    let hash_ok = asm.new_dynamic_label();
    dynasm!(asm; test eax, eax; jnz =>hash_ok; mov eax, 1; =>hash_ok);
    dynasm!(asm; mov edx, eax); // edx = hash_tag
    dynasm!(asm; mov r10, rax; and r10, r9); // r10 = slot

    let probe_lp   = asm.new_dynamic_label();
    let probe_next = asm.new_dynamic_label();
    let probe_done = asm.new_dynamic_label();

    dynasm!(asm; =>probe_lp);
    match arity {
        1 => dynasm!(asm; lea rcx, [r10 + r10]),
        2 => dynasm!(asm; lea rcx, [r10 + r10*2]),
        _ => dynasm!(asm; lea rcx, [r10 + r10*3]),
    }
    dynasm!(asm
        ; mov eax, [r8 + rcx*4]
        ; test eax, eax
        ; je =>when_exhausted
        ; cmp eax, edx
        ; jne =>probe_next
    );
    dynasm!(asm; cmp [r8 + rcx*4 + 4], edi; jne =>probe_next);
    if arity >= 2 { dynasm!(asm; cmp [r8 + rcx*4 + 8], esi;   jne =>probe_next); }
    if arity >= 3 { dynasm!(asm; cmp [r8 + rcx*4 + 12], r11d; jne =>probe_next); }
    dynasm!(asm; jmp =>probe_done);
    dynasm!(asm; =>probe_next; inc r10; and r10, r9; jmp =>probe_lp);
    dynasm!(asm; =>probe_done);
    Ok(())
}

fn emit_tuple_set_probe(
    asm: &mut Assembler,
    clause: &CClause,
    handle_idx: usize,
    when_exhausted: DynamicLabel,
    var_locs: &[VarLoc],
    // When Some((flat_idx, total_rels_slot)), native path: use cached total_rels slot.
    // When None, old ctx->tuple_sets_buf path.
    native_probe: Option<(usize, i32)>,
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

    // For both native and non-native paths, use the scalar JitTupleSet probe.
    // The SIMD Swiss-table path (JitSwissTable, commit 3392df1) was measured to add
    // ~15 instructions of setup overhead vs no benefit for small hot tables that fit
    // in L1/L2 cache.  The scalar probe is faster in the common case.
    emit_tuple_set_probe_scalar(asm, clause, handle_idx, when_exhausted, var_locs, native_probe)
}

// ─── EDB full-probe via JitColIndex scan ───────────────────────────────────
//
// For an arity-2 clause where ALL args are bound (existence check), the
// JitTupleSet hash table is a poor fit at large n: the table is 12 bytes/slot
// and grows to ~96 KB for 5000 tuples, causing L2/L3 cache misses on every
// random probe.
//
// Instead: look up the bound primary-column value in the JitColIndex (which
// holds a compact hash of ~99 unique keys, ~1 KB, always in L1), get the
// start/count of the corresponding sorted col-1 values, then scan linearly
// through that contiguous range (≤50 u32s = ≤200 bytes, fits in L1).
//
// Preconditions:
//   - p.use_jit_native && !is_rec (EDB relation, native path)
//   - clause.fresh_cols.is_empty() (all args already bound)
//   - clause.args.len() == 2
//   - level >= 1
//
// On miss: jumps to when_not_found.
// On hit:  falls through (caller emits conditions/heads).
//
// Registers used (caller-saved or saved by level entry code):
//   rdi  = JitRelData*   (scratch)
//   r8   = JitColIndex*  (scratch)
//   r9   = keys ptr      (scratch)
//   eax, edx = hash slot / count (scratch)
//   ecx  = primary key   (scratch)
//   esi  = start         (scratch)
//   r11  = vals ptr      (scratch)
//   r10d = secondary key (scratch)
//   r15, rbx = scan ptr / count (saved by level >= 2 entry; free at level 1)
fn emit_native_edb_full_probe(
    asm: &mut Assembler,
    p: &EmitParams<'_>,
    level: usize,
    clause: &CClause,
    when_not_found: DynamicLabel,
) -> Result<(), String> {
    let arity = clause.args.len();
    debug_assert_eq!(arity, 2);

    // flat handle index for total (non-recent) version of this clause's relation
    let total_flat_idx = p.rule_handle_base + level * 2;
    let total_flat_idx_i32 = (total_flat_idx * 8) as i32;

    // Load JitRelData* for this clause → rdi
    dynasm!(asm
        ; mov rdi, [rbp + CTX_SLOT]
        ; mov rdi, [rdi]                       // scan_rels ptr
        ; mov rdi, [rdi + total_flat_idx_i32]  // *mut JitRelData
    );

    // Use bound_cols[0] as the primary lookup column.
    let primary_col = clause.bound_cols[0];
    let secondary_col = 1 - primary_col;
    let primary_col_i32 = (primary_col * 8) as i32;

    // Load JitColIndex* for primary_col → r8
    dynasm!(asm
        ; mov r8, [rdi + 24]              // col_indices: *mut *mut JitColIndex
        ; mov r8, [r8 + primary_col_i32]  // JitColIndex* for primary_col
    );

    // Load primary key → ecx
    match &clause.args[primary_col] {
        CClauseArg::Var(var_id) => emit_load_var_ecx(asm, p.var_locs, *var_id),
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            dynasm!(asm; mov ecx, *n);
        }
        _ => {
            return Err(
                "emit_native_edb_full_probe: unsupported primary arg type".to_string(),
            )
        }
    }

    // Hash-probe JitColIndex for primary key → start (esi), count (eax), vals (r11)
    // Knuth multiplicative hash: slot = (key * 0x9e3779b9) & mask
    dynasm!(asm
        ; mov eax, ecx
        ; imul eax, eax, 0x9e3779b9u32 as i32
        ; mov edx, [r8 + 24]   // JitColIndex.mask
        ; and eax, edx          // slot
        ; mov r9, [r8]          // keys ptr
    );

    let probe_lp  = asm.new_dynamic_label();
    let probe_hit = asm.new_dynamic_label();

    dynasm!(asm; =>probe_lp);
    dynasm!(asm
        ; mov esi, [r9 + rax*4]
        ; cmp esi, -1i32           // EMPTY_KEY → key absent
        ; je =>when_not_found
        ; cmp esi, ecx
        ; je =>probe_hit
        ; inc eax
        ; and eax, edx
        ; jmp =>probe_lp
    );

    dynasm!(asm; =>probe_hit);
    // ranges[slot] = start(lo32) | count(hi32)
    dynasm!(asm
        ; mov r11, [r8 + 8]       // ranges ptr
        ; mov r11, [r11 + rax*8]  // ranges[slot]
        ; mov esi, r11d            // start = lo32
        ; shr r11, 32
        ; mov eax, r11d            // count = hi32
        ; mov r11, [r8 + 16]      // vals ptr
    );

    // Check count is non-zero and compute hi before loading secondary key
    // (secondary key load may clobber eax which holds count).
    dynasm!(asm
        ; test eax, eax
        ; jz =>when_not_found               // count == 0: not in relation
        ; lea edx, [rsi + rax - 1]          // edx = hi = start + count - 1
    );

    // Load secondary key → r10d  (emit_load_var writes eax; move to r10d)
    match &clause.args[secondary_col] {
        CClauseArg::Var(var_id) => {
            emit_load_var(asm, p.var_locs, *var_id);
            dynasm!(asm; mov r10d, eax);
        }
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            dynasm!(asm; mov r10d, *n);
        }
        _ => {
            return Err(
                "emit_native_edb_full_probe: unsupported secondary arg type".to_string(),
            )
        }
    }

    // Binary search: vals[lo..=hi] for secondary key (target = r10d).
    //
    // Registers:
    //   esi  = lo  (start index, updated each iteration)
    //   edx  = hi  (end index, updated each iteration)
    //   ecx  = mid (= (lo+hi)/2 each iteration)
    //   eax  = vals[mid]  (scratch)
    //   r10d = target (secondary key)
    //   r11  = vals base ptr
    //
    // vals is sorted ascending (JitColIndex invariant), so binary search
    // terminates in ceil(log₂(count)) ≤ 7 iterations for count ≤ 99.
    // The ~400-byte slice for a fixed primary key is L1-resident after the
    // first probe, making each subsequent probe essentially free.
    let bsearch_loop  = asm.new_dynamic_label();
    let bsearch_right = asm.new_dynamic_label();
    let bsearch_found = asm.new_dynamic_label();

    dynasm!(asm
        ; =>bsearch_loop
        ; cmp esi, edx
        ; jg =>when_not_found               // lo > hi: not found
        ; lea ecx, [rsi + rdx]
        ; shr ecx, 1                        // mid = (lo + hi) / 2
        ; mov eax, [r11 + rcx*4]            // val = vals[mid]
        ; cmp eax, r10d
        ; je =>bsearch_found                // hit: fall through
        ; jl =>bsearch_right                // val < target → search right half
        ; lea edx, [rcx - 1]               // hi = mid - 1
        ; jmp =>bsearch_loop
        ; =>bsearch_right
        ; lea esi, [rcx + 1]               // lo = mid + 1
        ; jmp =>bsearch_loop
        ; =>bsearch_found
    );

    Ok(())
}

// ─── Merge intersection ───────────────────────────────────────────────────

/// Detected pattern for a two-pointer sorted-merge intersection.
///
/// Fuses a col-scan level L (arity-2 EDB, one fresh col) with a fully-bound
/// existence-check level L+1 (arity-2 EDB, no fresh cols) into a single
/// two-pointer sorted-merge loop that eliminates the per-element hash probe.
///
/// Both relations must be native-path EDB (JitColIndex sorted vals arrays).
struct MergePattern {
    /// Flat handle index for the scan relation at level L (includes use_recent).
    scan_flat_idx: usize,
    /// Primary column for JitColIndex probe on the scan relation.
    scan_primary_col: usize,
    /// Flat handle index for the existence relation at level L+1 (always total).
    exist_flat_idx: usize,
    /// Primary column for JitColIndex probe on the existence relation.
    exist_primary_col: usize,
    /// Variable ID of the key for the existence probe (bound before level L).
    exist_key_var: u32,
    /// Variable ID bound as the fresh column from the scan level.
    fresh_var: u32,
    /// Index of level L+1 (the existence check level).
    exist_level: usize,
}

/// Detect whether levels [level, level+1] form a merge-intersectable pattern.
fn detect_merge_pattern(p: &EmitParams<'_>, level: usize) -> Option<MergePattern> {
    if !p.use_jit_native {
        return None;
    }
    let clause_l = &p.clauses[level];
    if clause_l.args.len() != 2 || p.is_rec[level] {
        return None;
    }
    // Scan level: exactly one fresh col (the col we iterate over).
    if clause_l.fresh_cols.len() != 1 || clause_l.bound_cols.is_empty() {
        return None;
    }
    let exist_level = level + 1;
    if exist_level >= p.clauses.len() {
        return None;
    }
    let clause_l1 = &p.clauses[exist_level];
    if clause_l1.args.len() != 2 || p.is_rec[exist_level] {
        return None;
    }
    // Existence check: fully-bound, total version.
    if !clause_l1.fresh_cols.is_empty() || p.use_recent[exist_level] {
        return None;
    }
    let (_, fresh_var_l) = clause_l.fresh_cols[0];
    let scan_primary_col = clause_l.bound_cols[0];
    // Classify clause_l1's args: one must be fresh_var_l (the merge key),
    // the other must be a different Var (the key for the existence probe).
    let (exist_primary_col, exist_key_var) = match (&clause_l1.args[0], &clause_l1.args[1]) {
        (CClauseArg::Var(v0), CClauseArg::Var(v1)) => {
            if *v1 == fresh_var_l && *v0 != fresh_var_l {
                (0usize, *v0)
            } else if *v0 == fresh_var_l && *v1 != fresh_var_l {
                (1usize, *v1)
            } else {
                return None;
            }
        }
        _ => return None,
    };
    let scan_flat_idx =
        p.rule_handle_base + level * 2 + if p.use_recent[level] { 1 } else { 0 };
    let exist_flat_idx = p.rule_handle_base + exist_level * 2; // total (use_recent=false)
    Some(MergePattern {
        scan_flat_idx,
        scan_primary_col,
        exist_flat_idx,
        exist_primary_col,
        exist_key_var,
        fresh_var: fresh_var_l,
        exist_level,
    })
}

/// Emit a two-pointer sorted-merge intersection loop.
///
/// Replaces the col-scan at `level` and the existence check at `level+1`.
/// For each (key_b, key_a) pair already established, probes both JitColIndex
/// sorted slices once and walks them with two pointers in O(|A|+|B|) steps,
/// binding the matched value as `fresh_var` and emitting heads (or deeper levels)
/// for each hit.
///
/// Register contract during the merge loop:
///   r15 = ptr_b: advancing pointer into scan vals slice
///   rbx = end_b: end of scan vals slice (fixed)
///   [rbp + ptr_a_slot] = ptr_a: advancing pointer into exist vals slice
///   [rbp + end_a_slot] = end_a: end of exist vals slice (fixed)
#[allow(clippy::too_many_arguments)]
fn emit_merge_scan_exist(
    asm: &mut Assembler,
    p: &EmitParams<'_>,
    level: usize,
    outer_exit: DynamicLabel,
    when_exhausted: DynamicLabel,
    mp: &MergePattern,
) -> Result<(), String> {
    // Reuse per-level stack slots for ptr_a and end_a.
    // level_vptr_slot  → ptr_a (current position in exist vals slice)
    // level_count_slot → end_a (fixed end of exist vals slice)
    let ptr_a_slot = level_vptr_slot(level, p.var_count, p.max_head_arity);
    let end_a_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
    let exist_flat_idx_i32 = (mp.exist_flat_idx * 8) as i32;
    let scan_flat_idx_i32 = (mp.scan_flat_idx * 8) as i32;

    // ── Step 1: Probe existence relation for key_a → load vals_a slice ──────
    // Load JitRelData* for existence rel → rdi
    dynasm!(asm
        ; mov rdi, [rbp + CTX_SLOT]
        ; mov rdi, [rdi]                              // scan_rels ptr (NativeCtx offset 0)
        ; mov rdi, [rdi + exist_flat_idx_i32]         // *mut JitRelData for exist rel
    );
    let exist_pcol_off = (mp.exist_primary_col * 8) as i32;
    dynasm!(asm
        ; mov r8, [rdi + 24]                          // col_indices: *mut *mut JitColIndex
        ; mov r8, [r8 + exist_pcol_off]               // JitColIndex* for exist_primary_col
    );
    // Load exist key into ecx
    emit_load_var_ecx(asm, p.var_locs, mp.exist_key_var);
    // Hash probe JitColIndex: slot = (key * KNUTH32) & mask
    dynasm!(asm
        ; mov eax, ecx
        ; imul eax, eax, 0x9e3779b9u32 as i32
        ; mov edx, [r8 + 24]                          // JitColIndex.mask
        ; and eax, edx                                 // slot = hash & mask
        ; mov r9, [r8]                                 // keys ptr
    );
    let probe_a_lp  = asm.new_dynamic_label();
    let probe_a_hit = asm.new_dynamic_label();
    dynasm!(asm; =>probe_a_lp);
    dynasm!(asm
        ; mov esi, [r9 + rax*4]
        ; cmp esi, -1i32
        ; je =>when_exhausted          // key absent: no intersection possible
        ; cmp esi, ecx
        ; je =>probe_a_hit
        ; inc eax
        ; and eax, edx
        ; jmp =>probe_a_lp
    );
    dynasm!(asm; =>probe_a_hit);
    // ranges[slot] = start(lo32) | count(hi32)
    dynasm!(asm
        ; mov r11, [r8 + 8]            // ranges ptr
        ; mov r11, [r11 + rax*8]       // ranges[slot]
        ; mov esi, r11d                // start_a = lo32
        ; shr r11, 32
        ; mov eax, r11d                // count_a = hi32
        ; mov r11, [r8 + 16]           // vals_a ptr
    );
    dynasm!(asm; test eax, eax; jz =>when_exhausted);
    // ptr_a = vals_a + start_a*4;  end_a = ptr_a + count_a*4
    // (rsi = start_a u32 zero-extended; rax = count_a u32 zero-extended)
    dynasm!(asm
        ; lea rcx, [r11 + rsi*4]       // ptr_a
        ; mov [rbp + ptr_a_slot], rcx
        ; lea rax, [rcx + rax*4]       // end_a = ptr_a + count_a*4
        ; mov [rbp + end_a_slot], rax
    );

    // ── Step 2: Probe scan relation for key_b → r15=ptr_b, rbx=end_b ────────
    dynasm!(asm
        ; mov rdi, [rbp + CTX_SLOT]
        ; mov rdi, [rdi]
        ; mov rdi, [rdi + scan_flat_idx_i32]          // *mut JitRelData for scan rel
    );
    let scan_pcol_off = (mp.scan_primary_col * 8) as i32;
    dynasm!(asm
        ; mov r8, [rdi + 24]
        ; mov r8, [r8 + scan_pcol_off]                // JitColIndex* for scan_primary_col
    );
    // Load scan key into ecx
    let scan_clause = &p.clauses[level];
    match &scan_clause.args[mp.scan_primary_col] {
        CClauseArg::Var(var_id) => emit_load_var_ecx(asm, p.var_locs, *var_id),
        CClauseArg::Expr(CExpr::Literal(Value::I32(n))) => {
            let n = *n;
            dynasm!(asm; mov ecx, n);
        }
        _ => return Err("emit_merge: unsupported scan key arg type".to_string()),
    }
    // Hash probe
    dynasm!(asm
        ; mov eax, ecx
        ; imul eax, eax, 0x9e3779b9u32 as i32
        ; mov edx, [r8 + 24]
        ; and eax, edx
        ; mov r9, [r8]
    );
    let probe_b_lp  = asm.new_dynamic_label();
    let probe_b_hit = asm.new_dynamic_label();
    dynasm!(asm; =>probe_b_lp);
    dynasm!(asm
        ; mov esi, [r9 + rax*4]
        ; cmp esi, -1i32
        ; je =>when_exhausted
        ; cmp esi, ecx
        ; je =>probe_b_hit
        ; inc eax
        ; and eax, edx
        ; jmp =>probe_b_lp
    );
    dynasm!(asm; =>probe_b_hit);
    dynasm!(asm
        ; mov r11, [r8 + 8]
        ; mov r11, [r11 + rax*8]
        ; mov esi, r11d
        ; shr r11, 32
        ; mov eax, r11d
        ; mov r11, [r8 + 16]
    );
    dynasm!(asm; test eax, eax; jz =>when_exhausted);
    // r15 = ptr_b = vals_b + start_b*4;  rbx = end_b = ptr_b + count_b*4
    dynasm!(asm
        ; lea r15, [r11 + rsi*4]
        ; lea rbx, [r15 + rax*4]
    );

    // ── Step 3: Two-pointer sorted merge loop ────────────────────────────────
    // Each iteration: compare *ptr_b vs *ptr_a
    //   equal   → match: bind fresh_var, emit body, advance both
    //   b < a   → advance ptr_b (scan pointer)
    //   b > a   → advance ptr_a (exist pointer, in [rbp + ptr_a_slot])
    let merge_loop  = asm.new_dynamic_label();
    let merge_adv_b = asm.new_dynamic_label();
    let merge_match = asm.new_dynamic_label();
    let merge_cont  = asm.new_dynamic_label();

    dynasm!(asm; =>merge_loop);
    // Bounds checks
    dynasm!(asm; cmp r15, rbx; jge =>when_exhausted);        // ptr_b >= end_b
    dynasm!(asm
        ; mov rax, [rbp + ptr_a_slot]
        ; cmp rax, [rbp + end_a_slot]
        ; jge =>when_exhausted                                // ptr_a >= end_a
    );
    // Compare elements (rax still = ptr_a)
    dynasm!(asm
        ; mov ecx, [r15]               // b_val = *ptr_b
        ; mov r10d, [rax]              // a_val = *ptr_a
        ; cmp ecx, r10d
        ; je =>merge_match
        ; jl =>merge_adv_b
        // b_val > a_val: advance ptr_a
        ; add rax, 4
        ; mov [rbp + ptr_a_slot], rax
        ; jmp =>merge_loop
    );
    dynasm!(asm; =>merge_adv_b);
    dynasm!(asm; add r15, 4; jmp =>merge_loop);

    dynasm!(asm; =>merge_match);
    // Bind fresh_var = ecx (the matched value)
    dynasm!(asm; mov edx, ecx);
    emit_store_var(asm, p.var_locs, mp.fresh_var);

    // Clause L conditions
    for cond in &scan_clause.conditions {
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, p.var_locs)?;
            dynasm!(asm; test eax, eax; jz =>merge_cont);
        }
    }
    // Clause L+1 conditions (existence check may carry additional filters)
    let exist_clause = &p.clauses[mp.exist_level];
    for cond in &exist_clause.conditions {
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, p.var_locs)?;
            dynasm!(asm; test eax, eax; jz =>merge_cont);
        }
    }

    if mp.exist_level + 1 == p.clauses.len() {
        // Terminal: emit rule-level conditions, anti-joins, and heads.
        for expr in p.rule_conds {
            emit_expr(asm, expr, p.var_locs)?;
            dynasm!(asm; test eax, eax; jz =>merge_cont);
        }
        emit_not_probes(
            asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, merge_cont,
        )?;
        emit_heads(
            asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth,
            p.pti, p.var_locs, p.use_jit_native, p.rule_head_base,
            p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr,
        )?;
    } else {
        // Non-terminal: recurse into deeper level (exist_level+1).
        // emit_clause_level(exist_level+1) will save r15/rbx at its top
        // (since exist_level+1 >= 2 when level >= 1) and sub_exhausted restores them.
        let sub_exhausted = asm.new_dynamic_label();
        emit_clause_level(asm, p, mp.exist_level + 1, outer_exit, sub_exhausted)?;
        dynasm!(asm; =>sub_exhausted);
        let node_slot =
            level_node_save_slot(mp.exist_level, p.var_count, p.max_head_arity, p.max_depth);
        let dptr_slot =
            level_dptr_save_slot(mp.exist_level, p.var_count, p.max_head_arity, p.max_depth);
        dynasm!(asm
            ; mov r15, [rbp + node_slot]
            ; mov rbx, [rbp + dptr_slot]
        );
    }

    // merge_cont: advance both pointers, then loop.
    dynasm!(asm; =>merge_cont);
    dynasm!(asm; add r15, 4);                    // advance ptr_b
    dynasm!(asm
        ; mov rax, [rbp + ptr_a_slot]
        ; add rax, 4
        ; mov [rbp + ptr_a_slot], rax            // advance ptr_a
        ; jmp =>merge_loop
    );
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
        Add | Sub | Mul | Div | Rem | Eq | Ne | Lt | Le | Gt | Ge
        | And | Or | BitAnd | BitOr | BitXor | Shl | Shr => Ok(()),
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
        CExpr::VarBinLit(op, a, lit) => {
            emit_load_var(asm, var_locs, *a);
            match lit {
                Value::I32(n) => dynasm!(asm; mov ecx, *n),
                Value::Bool(b) => dynasm!(asm; mov ecx, if *b { 1i32 } else { 0i32 }),
                _ => return Err(format!("asm: VarBinLit unsupported literal: {lit:?}")),
            }
            emit_binop(asm, *op)?;
        }
        CExpr::LitBinVar(op, lit, b) => {
            match lit {
                Value::I32(n) => dynasm!(asm; mov eax, *n),
                Value::Bool(bval) => dynasm!(asm; mov eax, if *bval { 1i32 } else { 0i32 }),
                _ => return Err(format!("asm: LitBinVar unsupported literal: {lit:?}")),
            }
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
        BitAnd => dynasm!(asm; and eax, ecx),
        BitOr  => dynasm!(asm; or eax, ecx),
        BitXor => dynasm!(asm; xor eax, ecx),
        Shl    => dynasm!(asm; shl eax, cl),
        Shr    => dynasm!(asm; sar eax, cl),
        // cdq sign-extends eax into edx:eax; idiv ecx: quotient→eax, remainder→edx.
        // rdx is documented as clobbered by emit_expr, so this is safe.
        Div    => dynasm!(asm; cdq; idiv ecx),
        Rem    => dynasm!(asm; cdq; idiv ecx; mov eax, edx),
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

/// Build and insert all heads.
///
/// Non-native path: inline dedup probe for duplicates, then `packed_try_insert`.
///
/// Native path (Step 5): fully inline — JitTupleSet probe for dedup, then
/// direct write to `JitRelData.data` + JitTupleSet insert.  Growth callbacks
/// (`jit_rel_data_grow`, `jit_tuple_set_grow`) are called only on overflow.
#[allow(clippy::too_many_arguments)]
fn emit_heads(
    asm: &mut Assembler,
    heads: &[CHeadClause],
    rule_i: usize,
    var_count: usize,
    max_head_arity: usize,
    max_depth: usize,
    pti: usize,
    var_locs: &[VarLoc],
    // When true, write directly to JitRelData (head_rels[flat_hi]) without any Rust
    // call in the common case.  CTX_SLOT holds a *mut StratumStage4NativeCtx.
    use_jit_native: bool,
    rule_head_base: usize,
    jit_rel_data_grow_addr: usize,
    jit_tuple_set_grow_addr: usize,
) -> Result<(), String> {
    // Precompute native ctx array-ptr cache slots (used throughout native write path).
    let hr_slot  = if use_jit_native { native_head_rels_slot(var_count, max_head_arity, max_depth) } else { 0 };
    let htr_slot = if use_jit_native { native_head_total_rels_slot(var_count, max_head_arity, max_depth) } else { 0 };

    for (hi, head) in heads.iter().enumerate() {
        let arity = head.args.len();
        let hoff: i32 = (hi as i32) * 8;
        let t0_off = head_col_slot(var_count, max_head_arity, 0);

        for (col, arg) in head.args.iter().enumerate() {
            emit_expr(asm, arg, var_locs)?;
            dynasm!(asm; mov [rbp + head_col_slot(var_count, max_head_arity, col)], eax);
        }

        if use_jit_native {
            // ── Step 5: fully inline native write path ──────────────────────────
            //
            // Layout refs:
            //   StratumStage4NativeCtx:
            //     head_rels @ 24  (*mut *mut JitRelData)
            //   JitRelData:
            //     data      @  0  (*mut u32)
            //     len       @  8  (u64)
            //     cap       @ 16  (u64)
            //     tuple_set @ 32  (JitTupleSet embedded, 24 bytes)
            //     arity     @ 56  (u32)
            //   JitTupleSet (at JitRelData+32):
            //     slots     @  0  (*mut u32)   → absolute: JitRelData+32
            //     mask      @  8  (u64)        → absolute: JitRelData+40
            //     len       @ 16  (u64)        → absolute: JitRelData+48
            //
            // flat_hi = rule_head_base + hi; head_rels[flat_hi] = *mut JitRelData (.new buf)
            //
            // Register protocol:
            //   rdi = *mut JitRelData (reloaded after any call)
            //   r8  = tuple_set.slots (*mut u32)
            //   r9  = tuple_set.mask (u64)
            //   r10 = probe slot index (u64)
            //   rdx = hash_tag (u32; non-zero)
            //   rsi, rcx, rax = scratch

            let flat_hi = rule_head_base + hi;
            let flat_hi_i32 = (flat_hi * 8) as i32;

            if arity == 0 {
                // Zero-arity head: set new.len to 1 if not already 1 (idempotent).
                // No tuple_set probe needed — arity-0 has exactly one possible fact.
                // JitRelData.len @ 8; if already 1, skip write.
                let head_done_0 = asm.new_dynamic_label();
                dynasm!(asm
                    ; mov rdi, [rbp + hr_slot]         // head_rels (pre-cached)
                    ; mov rdi, [rdi + flat_hi_i32]     // *mut JitRelData
                    ; mov rax, [rdi + 8]               // len
                    ; test rax, rax
                    ; jnz =>head_done_0                // if len != 0 → already inserted
                    ; mov rax, 1i32
                    ; mov [rdi + 8], rax               // set len = 1
                    ; =>head_done_0
                );
                continue;
            }

            let head_done = asm.new_dynamic_label();
            let write_path = asm.new_dynamic_label();
            let stride = (arity + 1) as i32;

            // ── 1. Compute tuple hash (shared for both probes) ───────────────────
            //   h = 0x9e3779b9; for each word w: h = h*0x9e3779b9 + w; if h==0: h=1
            dynasm!(asm; mov edx, 0x9e3779b9u32 as i32);
            for col in 0..arity {
                dynasm!(asm
                    ; imul edx, edx, 0x9e3779b9u32 as i32
                    ; add edx, [rbp + head_col_slot(var_count, max_head_arity, col)]
                );
            }
            {
                let hash_nonzero = asm.new_dynamic_label();
                dynasm!(asm; test edx, edx; jnz =>hash_nonzero; mov edx, 1; =>hash_nonzero);
            }
            // rdx = hash_tag (non-zero)

            // ── 2a. Cross-iteration dedup: probe total.tuple_set ─────────────────
            //   head_total_rels is pre-cached in htr_slot.
            //   total.tuple_set.slots @ JitRelData+32, mask @ JitRelData+40
            {
                dynasm!(asm
                    ; mov rdi, [rbp + htr_slot]        // head_total_rels (pre-cached)
                    ; mov rdi, [rdi + flat_hi_i32]     // *mut JitRelData (total)
                    ; mov r8, [rdi + 32]               // total.tuple_set.slots
                    ; mov r9, [rdi + 40]               // total.tuple_set.mask
                    ; mov r10, rdx                     // slot = hash & mask
                    ; and r10, r9
                );
                let total_probe_lp   = asm.new_dynamic_label();
                let total_probe_next = asm.new_dynamic_label();
                let total_probe_miss = asm.new_dynamic_label();
                dynasm!(asm; =>total_probe_lp);
                dynasm!(asm; imul rcx, r10, stride);
                dynasm!(asm
                    ; mov eax, [r8 + rcx*4]
                    ; test eax, eax
                    ; jz =>total_probe_miss    // empty → not in total
                    ; cmp eax, edx
                    ; jne =>total_probe_next   // hash mismatch
                );
                for col in 0..arity {
                    let field_off = ((1 + col) as i32) * 4;
                    dynasm!(asm
                        ; mov eax, [r8 + rcx*4 + field_off]
                        ; cmp eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
                        ; jne =>total_probe_next
                    );
                }
                // Found in total → cross-iteration duplicate, skip
                dynasm!(asm; jmp =>head_done);
                dynasm!(asm; =>total_probe_next);
                dynasm!(asm; inc r10; and r10, r9; jmp =>total_probe_lp);
                dynasm!(asm; =>total_probe_miss);
                // Not in total → check new (within-iteration dedup)
            }

            // ── 2b. Within-iteration dedup: probe new.tuple_set ──────────────────
            //   head_rels is pre-cached in hr_slot.
            //   Reuse r10 as slot (recompute from hash & mask of new.tuple_set).
            {
                dynasm!(asm
                    ; mov rdi, [rbp + hr_slot]         // head_rels (pre-cached)
                    ; mov rdi, [rdi + flat_hi_i32]     // *mut JitRelData (new)
                    ; mov r8, [rdi + 32]               // new.tuple_set.slots
                    ; mov r9, [rdi + 40]               // new.tuple_set.mask
                    ; mov r10, rdx                     // slot = hash & mask
                    ; and r10, r9
                );
                let probe_lp   = asm.new_dynamic_label();
                let probe_next = asm.new_dynamic_label();
                dynasm!(asm; =>probe_lp);
                dynasm!(asm; imul rcx, r10, stride);
                dynasm!(asm
                    ; mov eax, [r8 + rcx*4]
                    ; test eax, eax
                    ; jz =>write_path          // empty → not found, proceed to write
                    ; cmp eax, edx
                    ; jne =>probe_next
                );
                for col in 0..arity {
                    let field_off = ((1 + col) as i32) * 4;
                    dynasm!(asm
                        ; mov eax, [r8 + rcx*4 + field_off]
                        ; cmp eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
                        ; jne =>probe_next
                    );
                }
                // All match → within-iteration duplicate, skip
                dynasm!(asm; jmp =>head_done);
                dynasm!(asm; =>probe_next);
                dynasm!(asm; inc r10; and r10, r9; jmp =>probe_lp);
                // (write_path follows below — rdi still = *mut JitRelData (new))
            }

            // ── 6. write_path: new tuple — write to JitRelData.data + tuple_set ──
            dynasm!(asm; =>write_path);
            // r10 = empty slot index (save across grow calls by using callee-saved pattern)
            // rdx = hash_tag
            // rdi = *mut JitRelData
            // r8  = slots ptr (may need reload after jit_rel_data_grow if data changed)
            // r9  = mask

            // 6a. Check capacity; call jit_rel_data_grow if full
            //   len @ rdi+8, cap @ rdi+16
            {
                let skip_grow = asm.new_dynamic_label();
                dynasm!(asm
                    ; mov rsi, [rdi + 8]    // len
                    ; mov rcx, [rdi + 16]   // cap
                    ; cmp rsi, rcx
                    ; jne =>skip_grow
                    // Save r10 (slot) and rdx (hash) across the grow call.
                    // r10 and rdx are caller-saved; push/pop around call.
                    ; push r10
                    ; push rdx
                    // rdi already = *mut JitRelData (first arg)
                    ; mov rax, QWORD jit_rel_data_grow_addr as _
                    ; call rax
                    // Restore
                    ; pop rdx
                    ; pop r10
                    // Reload rdi (clobbered by call); use pre-cached head_rels slot
                    ; mov rdi, [rbp + hr_slot]
                    ; mov rdi, [rdi + flat_hi_i32]
                    ; =>skip_grow
                );
            }

            // 6b. Write arity words to data[len * arity]
            //   data @ rdi+0, len @ rdi+8 (reload after possible grow)
            dynasm!(asm
                ; mov rsi, [rdi + 8]         // len (reload)
                ; mov rcx, [rdi]             // data ptr
                ; imul rax, rsi, (arity * 4) as i32  // byte offset = len * arity * 4
                ; add rcx, rax               // write_ptr = data + len*arity*4
            );
            for col in 0..arity {
                dynasm!(asm
                    ; mov eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
                    ; mov [rcx + (col * 4) as i32], eax
                );
            }
            // Increment len (rdi = *mut JitRelData, len @ offset 8)
            dynasm!(asm
                ; mov rax, [rdi + 8]
                ; inc rax
                ; mov [rdi + 8], rax
            );

            // 6c. Insert into JitTupleSet at the empty slot (r10 = slot, rdx = hash_tag)
            //   slots ptr may be stale if jit_rel_data_grow moved data — but JitRelData
            //   itself didn't move, so reload r8/r9 from rdi+32/rdi+40.
            dynasm!(asm
                ; mov r8, [rdi + 32]   // reload slots ptr (in case grow ran)
                ; mov r9, [rdi + 40]   // reload mask
            );
            // Write slot: slots[r10*stride + 0] = hash_tag; slots[r10*stride + k+1] = col[k]
            dynasm!(asm; imul rcx, r10, stride);  // rcx = r10 * stride (word index)
            dynasm!(asm; mov [r8 + rcx*4], edx);  // write hash_tag
            for col in 0..arity {
                let field_off = ((1 + col) as i32) * 4;
                dynasm!(asm
                    ; mov eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
                    ; mov [r8 + rcx*4 + field_off], eax
                );
            }
            // Increment tuple_set.len (JitRelData+48 = JitRelData.tuple_set.len)
            dynasm!(asm
                ; mov rax, [rdi + 48]
                ; inc rax
                ; mov [rdi + 48], rax
            );

            // 6d. Load-factor check: if (new_len)*10 > (mask+1)*7 → call jit_tuple_set_grow
            //   new_len = tuple_set.len (just incremented), mask = r9
            //   (mask+1) is a power of two → cap_in_slots
            //   condition: len*10 > cap*7  →  len*10 > (mask+1)*7
            {
                let no_ts_grow = asm.new_dynamic_label();
                dynasm!(asm
                    ; mov rax, [rdi + 48]         // tuple_set.len (JitRelData+48)
                    ; imul rax, rax, 10i8
                    ; lea rcx, [r9 + 1]           // cap = mask + 1
                    ; imul rcx, rcx, 7i8
                    ; cmp rax, rcx
                    ; jle =>no_ts_grow
                    // call jit_tuple_set_grow(ts: *mut JitTupleSet, arity: u32)
                    // ts = rdi + 32, arity = compile-time constant
                    ; push r10
                    ; push rdx
                    ; lea rdi, [rdi + 32]          // rdi = &JitRelData.tuple_set
                    ; mov esi, arity as i32
                    ; mov rax, QWORD jit_tuple_set_grow_addr as _
                    ; call rax
                    ; pop rdx
                    ; pop r10
                    ; =>no_ts_grow
                );
                // Note: after jit_tuple_set_grow, r8/r9 are stale, but we don't need them
                // again (insert is already done; the grow only rehashes existing entries).
            }

            dynasm!(asm; =>head_done);
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

        // Compute hash (same as jit_dedup_hash / tuple_hash): h = 0x9e3779b9; for each word: h = h*KNUTH + word
        dynasm!(asm; mov eax, 0x9e3779b9u32 as i32);
        for col in 0..arity {
            dynasm!(asm
                ; imul eax, eax, 0x9e3779b9u32 as i32
                ; add eax, [rbp + head_col_slot(var_count, max_head_arity, col)]
            );
        }
        // Remap hash 0 → 1 (avoid sentinel JITDEDUP_EMPTY = 0)
        dynasm!(asm; test eax, eax; jnz >hash_nonzero_d; mov eax, 1; hash_nonzero_d:);

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
            ; test edi, edi
            ; jz =>call_insert          // empty slot (JITDEDUP_EMPTY=0) → new tuple
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
    /// Address of `jit_rel_data_grow` (native path only).
    jit_rel_data_grow_addr: usize,
    /// Address of `jit_tuple_set_grow` (native path only).
    jit_tuple_set_grow_addr: usize,
    /// Negation (anti-join) clauses for this rule.
    /// `rels[clauses.len() + i]` points to the i-th negated relation.
    negations: &'a [CAggregation],
}

/// Emit anti-join probes for all negation clauses of a rule.
///
/// For each `not rel(v0, v1, ...)` clause, emits:
///   1. Load argument variable values into arg registers (esi/edx/ecx via eax).
///   2. Load the negated relation pointer into rdi via `load_rel_rdi!`.
///   3. Call `check_not_packed_N(rdi, esi, ...)`.
///   4. `test eax, eax; jz =>skip_label` — skip if found in the relation.
///
/// `num_pos_clauses` = number of regular (positive) body clauses, used to compute
/// the negation rel index (`rels[num_pos_clauses + neg_i]`).
///
/// Only supported in the non-native asm path (i.e. `use_jit_native = false`).
fn emit_not_probes(
    asm: &mut Assembler,
    negations: &[CAggregation],
    num_pos_clauses: usize,
    rule_i: usize,
    var_locs: &[VarLoc],
    skip_label: DynamicLabel,
) -> Result<(), String> {
    use crate::jit::packed_helpers::{check_not_packed_1, check_not_packed_2, check_not_packed_3};
    for (neg_i, neg) in negations.iter().enumerate() {
        let arity = neg.args.len();
        // Extract var ids (eligibility already checked — all args are Var).
        let var_ids: Vec<u32> = neg.args.iter().map(|arg| match arg {
            CAggArg::Var(id) => Ok(*id),
            CAggArg::Expr(_) => Err(format!("not-clause '{}' has non-var arg", neg.relation)),
        }).collect::<Result<Vec<_>, _>>()?;

        let check_fn: usize = match arity {
            1 => check_not_packed_1 as usize,
            2 => check_not_packed_2 as usize,
            3 => check_not_packed_3 as usize,
            _ => return Err(format!("not-clause '{}' arity {arity} > 3", neg.relation)),
        };

        // Load arg values into argument registers via eax intermediary.
        // load_rel_rdi! only clobbers rdi, so pre-loading esi/edx/ecx is safe.
        for (i, &var_id) in var_ids.iter().enumerate() {
            emit_load_var(asm, var_locs, var_id); // → eax
            match i {
                0 => dynasm!(asm; mov esi, eax),
                1 => dynasm!(asm; mov edx, eax),
                2 => dynasm!(asm; mov ecx, eax),
                _ => unreachable!(),
            }
        }

        // Load negated relation pointer into rdi.
        load_rel_rdi!(asm, rule_i, num_pos_clauses + neg_i);

        // Call: returns 1 if NOT found (proceed), 0 if found (skip).
        call_abs!(asm, check_fn);
        dynasm!(asm; test eax, eax; jz =>skip_label);
    }
    Ok(())
}

// ─── SIMD column filter ───────────────────────────────────────────────────

/// Return the stack slot for `var_id` if it lives on the stack, or None if in a register.
fn var_loc_stack(var_locs: &[VarLoc], var_id: u32) -> Option<i32> {
    match var_locs.get(var_id as usize)? {
        VarLoc::Stack(s) => Some(*s),
        VarLoc::Reg(_) => None,
    }
}

/// How the SIMD comparand is supplied.
enum SimdComparand {
    /// Stack slot of an already-bound variable (loaded with `mov r8d, [rbp + slot]`).
    Var(i32),
    /// Literal i32 value (loaded with `mov r8d, DWORD n`).
    Lit(i32),
}

/// An AVX2-vectorizable `>` or `<` filter on the free column in an `is_col_value` scan.
///
/// The SIMD comparison is `vpcmpgtd ymm2, lhs, rhs` where:
///   `elements_gt = true`  → `lhs = ymm0 (elements)`, `rhs = ymm1 (comparand)` → element > comparand
///   `elements_gt = false` → `lhs = ymm1 (comparand)`, `rhs = ymm0 (elements)` → element < comparand
struct SimdFilter {
    comparand: SimdComparand,
    elements_gt: bool,
    /// Index in `clause.conditions` to skip inside the SIMD element body (already filtered).
    skip_cond: usize,
}

/// Try to detect a SIMD-filterable `>` or `<` condition on the free column.
///
/// Returns `Some(SimdFilter)` when:
///   - `clause.args[free_col]` is a fresh variable (being bound by this scan).
///   - One of `clause.conditions` is `CCondition::If(VarBinVar(Lt|Gt, ...) | VarBinLit(...) | LitBinVar(...))`.
///   - The comparand is a bound variable on the stack or an i32 literal.
fn detect_simd_filter(clause: &CClause, free_col: usize, var_locs: &[VarLoc]) -> Option<SimdFilter> {
    // Only applicable when the free column arg is a fresh variable.
    let free_var = match &clause.args[free_col] {
        CClauseArg::Var(vid) if clause.fresh_cols.iter().any(|&(c, _)| c == free_col) => *vid,
        _ => return None,
    };

    for (idx, cond) in clause.conditions.iter().enumerate() {
        let CCondition::If(expr) = cond else { continue };
        let result = match expr {
            // free_var < v2 → element < bound → elements_gt = false
            CExpr::VarBinVar(CBinOp::Lt, v1, v2) if *v1 == free_var => {
                var_loc_stack(var_locs, *v2).map(|s| (SimdComparand::Var(s), false))
            }
            // v1 < free_var → free_var > v1 → elements_gt = true
            CExpr::VarBinVar(CBinOp::Lt, v1, v2) if *v2 == free_var => {
                var_loc_stack(var_locs, *v1).map(|s| (SimdComparand::Var(s), true))
            }
            // free_var > v2 → elements_gt = true
            CExpr::VarBinVar(CBinOp::Gt, v1, v2) if *v1 == free_var => {
                var_loc_stack(var_locs, *v2).map(|s| (SimdComparand::Var(s), true))
            }
            // v1 > free_var → free_var < v1 → elements_gt = false
            CExpr::VarBinVar(CBinOp::Gt, v1, v2) if *v2 == free_var => {
                var_loc_stack(var_locs, *v1).map(|s| (SimdComparand::Var(s), false))
            }
            // free_var < n
            CExpr::VarBinLit(CBinOp::Lt, v, Value::I32(n)) if *v == free_var => {
                Some((SimdComparand::Lit(*n), false))
            }
            // free_var > n
            CExpr::VarBinLit(CBinOp::Gt, v, Value::I32(n)) if *v == free_var => {
                Some((SimdComparand::Lit(*n), true))
            }
            // n < free_var → free_var > n
            CExpr::LitBinVar(CBinOp::Lt, Value::I32(n), v) if *v == free_var => {
                Some((SimdComparand::Lit(*n), true))
            }
            // n > free_var → free_var < n
            CExpr::LitBinVar(CBinOp::Gt, Value::I32(n), v) if *v == free_var => {
                Some((SimdComparand::Lit(*n), false))
            }
            _ => None,
        };
        if let Some((comparand, elements_gt)) = result {
            return Some(SimdFilter { comparand, elements_gt, skip_cond: idx });
        }
    }
    None
}

/// Emit an AVX2 SIMD prefix loop for an `is_col_value` contiguous scan.
///
/// Entry state: r15 = elem_ptr (u32*), rbx = end_ptr (exclusive).
/// Processes batches of 8 elements at a time using `vpcmpgtd` to filter.
/// Falls through to `scalar_hdr` when fewer than 8 elements remain.
///
/// Element body: for each passing element, binds the free column variable and
/// recurses into clause level+1 (or emits heads if at the last clause).
/// Conditions in `clause.conditions[filter.skip_cond]` are skipped (already filtered).
#[allow(clippy::too_many_arguments)]
fn emit_col_value_simd_prefix(
    asm: &mut Assembler,
    p: &EmitParams<'_>,
    level: usize,
    outer_exit: DynamicLabel,
    scalar_hdr: DynamicLabel,
    filter: &SimdFilter,
) -> Result<(), String> {
    let clause = &p.clauses[level];
    let free_col = 1 - clause.bound_cols[0];
    let mask_slot = simd_mask_slot(p.var_count, p.max_head_arity, p.max_depth);
    let vals_slot = simd_vals_slot(p.var_count, p.max_head_arity, p.max_depth);
    let simd_hdr = asm.new_dynamic_label();
    let simd_elem_hdr = asm.new_dynamic_label();
    let simd_elem_continue = asm.new_dynamic_label();

    // ── SIMD batch loop ───────────────────────────────────────────────────
    // Precondition: r15 = elem_ptr, rbx = end_ptr.
    // Jump to scalar_hdr when fewer than 8 elements remain.
    dynasm!(asm; =>simd_hdr);
    dynasm!(asm
        ; lea rdx, [r15 + 32]         // rdx = r15 + 32 (one past last batch element)
        ; cmp rdx, rbx
        ; jg =>scalar_hdr             // fewer than 8 elements → fall to scalar tail
        ; vmovdqu ymm0, [r15]         // ymm0 = 8 u32 column values
    );
    // Broadcast the comparand value into ymm1.
    // Store comparand to simd_mask_slot temporarily (will be overwritten with the actual mask).
    // vpbroadcastd with mem32 source avoids the vmovd r32→xmm→ymm sequence.
    match &filter.comparand {
        SimdComparand::Var(slot) => dynasm!(asm
            ; mov eax, [rbp + *slot]
            ; mov [rbp + mask_slot], eax
        ),
        SimdComparand::Lit(n) => dynasm!(asm
            ; mov eax, DWORD *n
            ; mov [rbp + mask_slot], eax
        ),
    }
    dynasm!(asm
        ; vmovdqu ymm0, [r15]                      // load 8 u32 column values
        ; vpbroadcastd ymm1, [rbp + mask_slot]     // broadcast comparand (32-bit mem source)
    );
    // Signed compare: vpcmpgtd sets lane to 0xFFFFFFFF where lhs > rhs, else 0.
    if filter.elements_gt {
        dynasm!(asm; vpcmpgtd ymm2, ymm0, ymm1); // element > comparand
    } else {
        dynasm!(asm; vpcmpgtd ymm2, ymm1, ymm0); // comparand > element ⟺ element < comparand
    }
    // Extract the high bit of each 32-bit float lane as an 8-bit mask.
    // vmovmskps eax, ymm2 encoded as raw bytes: VEX.256.0F 50 /r
    //   C4 E1 7C 50 C2  →  vmovmskps eax, ymm2
    dynasm!(asm; .bytes [0xC4u8, 0xE1, 0x7C, 0x50, 0xC2]);
    dynasm!(asm
        ; vmovdqu [rbp + vals_slot], ymm0  // save batch values to stack
        ; vzeroupper                        // clear YMM upper halves before any Rust calls
        ; mov [rbp + mask_slot], eax        // overwrite temp comparand with actual mask
        ; add r15, 32                       // advance elem_ptr to next batch
    );

    // ── SIMD element body loop ────────────────────────────────────────────
    // r15 = next-batch ptr (callee-saved, survives Rust calls).
    // rbx = end_ptr (callee-saved, survives Rust calls).
    // [rbp + mask_slot]: 8-bit mask of remaining passing elements for this batch.
    // [rbp + vals_slot]: saved u32 values for this batch.
    dynasm!(asm; =>simd_elem_hdr);
    dynasm!(asm
        ; mov eax, [rbp + mask_slot]
        ; test eax, eax
        ; jz =>simd_hdr               // batch exhausted → fetch next
        ; bsf r10d, eax               // r10d = index of lowest set bit (0–7)
        ; btr eax, r10d               // clear that bit
        ; mov [rbp + mask_slot], eax
        ; lea r9, [rbp + vals_slot]
        ; mov ecx, [r9 + r10*4]       // ecx = element value at index r10
    );

    // Bind the free column variable from ecx (fresh → store; bound → compare+skip).
    emit_bind_col_value(asm, clause, free_col, simd_elem_continue, p.var_locs)?;

    // Emit conditions, skipping the one already handled by SIMD.
    for (idx, cond) in clause.conditions.iter().enumerate() {
        if idx == filter.skip_cond { continue; }
        if let CCondition::If(expr) = cond {
            emit_expr(asm, expr, p.var_locs)?;
            dynasm!(asm; test eax, eax; jz =>simd_elem_continue);
        }
    }

    // Emit level+1 body or (if last clause) rule conditions and heads.
    if level + 1 == p.clauses.len() {
        for expr in p.rule_conds {
            emit_expr(asm, expr, p.var_locs)?;
            dynasm!(asm; test eax, eax; jz =>simd_elem_continue);
        }
        emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, simd_elem_continue)?;
        emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
    } else {
        let simd_sub_exhausted = asm.new_dynamic_label();
        emit_clause_level(asm, p, level + 1, outer_exit, simd_sub_exhausted)?;

        // When the level+1 body exhausts, restore r15/rbx from the slots written
        // by level+1's entry code (lines that save r15/rbx when level >= 2).
        // At entry to the element body: r15 = next_batch_ptr, rbx = end_ptr.
        // Level+1 saves these and simd_sub_exhausted restores them.
        dynasm!(asm; =>simd_sub_exhausted);
        let node_slot = level_node_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
        let dptr_slot = level_dptr_save_slot(level, p.var_count, p.max_head_arity, p.max_depth);
        dynasm!(asm
            ; mov r15, [rbp + node_slot]
            ; mov rbx, [rbp + dptr_slot]
        );
    }

    dynasm!(asm; =>simd_elem_continue);
    dynasm!(asm; jmp =>simd_elem_hdr);

    Ok(())
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
            // use_recent=true  (pointer-based loop, r14 freed for var assignment):
            //   r13 = current tuple_ptr (advances by stride each iteration)
            //   r12 = end_ptr (= data + len*stride; loop terminates when r13 >= r12)
            //   r14 = available for variable assignment
            //
            // use_recent=false (index-based loop, r13 freed for var assignment):
            //   r12 = count (JitRelData.len @ 8)
            //   r13 = available for variable assignment
            //   r14 = loop counter i

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
                // Pointer-based outer loop: r13 = current_ptr, r12 = end_ptr.
                // r14 is freed — available for variable assignment in compute_var_locs.
                dynasm!(asm; mov r13, [rdi]);               // r13 = data (start_ptr)
                dynasm!(asm; imul r12, r12, stride);        // r12 = len * stride
                dynasm!(asm; add r12, r13);                 // r12 = end_ptr
            } else {
                // Index-based outer loop: r12 = count, r14 = counter.
                // r13 may hold a variable — cache data_ptr in a stack slot instead.
                let dp0 = data_ptr0_slot(p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov rax, [rdi]                    // JitRelData.data @ 0
                    ; mov [rbp + dp0], rax              // cache data_ptr
                );
                dynasm!(asm; xor r14d, r14d);           // i = 0
            }

            let loop_hdr = asm.new_dynamic_label();
            let outer_continue = asm.new_dynamic_label();

            dynasm!(asm; =>loop_hdr);

            if use_recent {
                // Pointer-based: compare current_ptr against end_ptr.
                dynasm!(asm; cmp r13, r12; jge =>outer_exit);
                dynasm!(asm; mov rax, r13);             // rax = current tuple_ptr
            } else {
                // Index-based: compare counter against count.
                dynasm!(asm; cmp r14, r12; jge =>outer_exit);
                let dp0 = data_ptr0_slot(p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov rax, [rbp + dp0]              // data_ptr (cached before loop)
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
                emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, outer_continue)?;
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
            } else {
                emit_clause_level(asm, p, 1, outer_exit, outer_continue)?;
            }

            dynasm!(asm; =>outer_continue);
            if use_recent {
                dynasm!(asm; add r13, stride; jmp =>loop_hdr);
            } else {
                dynasm!(asm; inc r14; jmp =>loop_hdr);
            }
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
            emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, outer_continue)?;
            emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
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
        // Native path only supports EDB (contiguous JitColIndex) inner clauses.
        // IDB inner clauses require full JitColIndex rebuild per iteration (O(n log n));
        // the non-native asm path uses O(1) linked-list inserts and handles IDB correctly.
        if p.use_jit_native && is_rec {
            return Err(format!(
                "asm native: clause {level} is IDB \
                 (per-iteration JitColIndex rebuild would be slower than linked-list path)"
            ));
        }
        let is_contiguous = !is_rec;

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

        // Fast path for fully-bound existence checks.
        if clause.fresh_cols.is_empty() {
            // Native EDB path (arity 2): use JitColIndex binary search instead of
            // JitTupleSet.  Binary search on the sorted vals slice has O(log n) ≈ 7
            // iterations vs O(1) for JitTupleSet, but the ~400-byte slice for a fixed
            // primary key is L1-resident across all inner iterations.  JitTupleSet at
            // large n (96KB for n=100) spills to L2 (~10 cycles/miss), making the
            // binary search faster despite more instructions.
            let use_col_scan = p.use_jit_native && !is_rec && arity == 2
                && !clause.bound_cols.is_empty();
            if use_col_scan {
                emit_native_edb_full_probe(asm, p, level, clause, when_exhausted)?;
                if level + 1 == p.clauses.len() {
                    for expr in p.rule_conds {
                        emit_expr(asm, expr, p.var_locs)?;
                        dynasm!(asm; test eax, eax; jz =>when_exhausted);
                    }
                    emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, when_exhausted)?;
                    emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
                    dynasm!(asm; jmp =>when_exhausted);
                } else {
                    emit_clause_level(asm, p, level + 1, outer_exit, when_exhausted)?;
                }
                return Ok(());
            }
            // Non-native or IDB or arity != 2: fall back to JitTupleSet hash probe.
            let handle_idx_in_buf = flat_idx;
            let native_total_for_probe = if p.use_jit_native {
                Some((total_flat_idx, native_total_rels_slot(p.var_count, p.max_head_arity, p.max_depth)))
            } else { None };
            if emit_tuple_set_probe(asm, clause, handle_idx_in_buf, when_exhausted, p.var_locs, native_total_for_probe).is_ok() {
                // Probe succeeded (emitted inline code). Fall through means found.
                if level + 1 == p.clauses.len() {
                    for expr in p.rule_conds {
                        emit_expr(asm, expr, p.var_locs)?;
                        dynasm!(asm; test eax, eax; jz =>when_exhausted);
                    }
                    emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, when_exhausted)?;
                    emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
                    dynasm!(asm; jmp =>when_exhausted);
                } else {
                    emit_clause_level(asm, p, level + 1, outer_exit, when_exhausted)?;
                }
                return Ok(());
            }
            // If emit_tuple_set_probe returned Err (e.g. unsupported arity), fall through
            // to the existing col-index + scan path.
        }

        // ── Merge intersection fast path ─────────────────────────────────────
        // When level L is a col-scan and level L+1 is a fully-bound existence
        // check, fuse them into a two-pointer sorted merge (O(|A|+|B|) per outer
        // pair instead of O(|B|×hash_probe)).
        if let Some(mp) = detect_merge_pattern(p, level) {
            emit_merge_scan_exist(asm, p, level, outer_exit, when_exhausted, &mp)?;
            return Ok(());
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
            );
            // Pre-compute vals_base = values_ptr + head*4.
            // For col_value: pointer-increment scan (r15 = elem_ptr, rbx = end_ptr).
            // For standard: j = 0 in r15d, count on stack, data_ptr in rbx.
            if is_col_value {
                dynasm!(asm
                    ; lea r15, [r11 + rsi*4]          // r15 = elem_ptr = vals_base + start*4
                    ; lea rbx, [r15 + rax*4]          // rbx = end_ptr = elem_ptr + count*4
                );
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; xor r15d, r15d                  // j = 0
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

            // SIMD prefix: emit before scalar tail when a filterable condition exists
            // on the free column.  Falls through to inner_hdr when <8 elements remain.
            if is_col_value {
                let free_col = 1 - clause.bound_cols[0];
                if let Some(ref sf) = detect_simd_filter(clause, free_col, p.var_locs) {
                    emit_col_value_simd_prefix(asm, p, level, outer_exit, inner_hdr, sf)?;
                }
            }

            dynasm!(asm; =>inner_hdr);
            if is_col_value {
                dynasm!(asm; cmp r15, rbx; jge =>when_exhausted);
            } else {
                let cnt_slot = level_count_slot(level, p.var_count, p.max_head_arity, p.max_depth);
                dynasm!(asm
                    ; mov eax, [rbp + cnt_slot]
                    ; cmp r15d, eax
                    ; jge =>when_exhausted
                );
            }

            // Load next element into ecx and advance position.
            if is_col_value {
                dynasm!(asm
                    ; mov ecx, [r15]                    // c = *elem_ptr (no stack load)
                    ; add r15, 4                        // elem_ptr++
                );
            } else {
                dynasm!(asm
                    ; mov rax, [rbp + vs]               // vals_base
                    ; mov ecx, [rax + r15*4]            // vals_base[j]
                    ; inc r15d                          // j++
                );
            }

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
                emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, inner_continue)?;
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
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
                emit_not_probes(asm, p.negations, p.clauses.len(), p.rule_i, p.var_locs, inner_continue)?;
                emit_heads(asm, p.heads, p.rule_i, p.var_count, p.max_head_arity, p.max_depth, p.pti, p.var_locs, p.use_jit_native, p.rule_head_base, p.jit_rel_data_grow_addr, p.jit_tuple_set_grow_addr)?;
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

/// Emit aggregation computations for a pure-aggregation rule (0 positive clauses).
///
/// For each aggregation, loads the relation from `rels[num_pos_clauses + num_nots + agg_i]`,
/// calls the appropriate helper, stores the result to `var_slot(result_var)`, and jumps
/// to `variant_exit` if the relation is empty (for sum/min/max only; count always continues).
fn emit_aggregations(
    asm: &mut Assembler,
    aggregations: &[CAggregation],
    num_pos_clauses: usize,
    num_nots: usize,
    rule_i: usize,
    variant_exit: DynamicLabel,
) -> Result<(), String> {
    use crate::jit::packed_helpers::{AGG_EMPTY, packed_agg_count, packed_agg_max_i32, packed_agg_min_i32, packed_agg_sum_i32};
    for (agg_i, agg) in aggregations.iter().enumerate() {
        let rel_seq = num_pos_clauses + num_nots + agg_i;
        let result_var = *agg.result_vars.first()
            .ok_or_else(|| format!("agg '{}': no result vars", agg.aggregator_name))?;
        match agg.aggregator_name.as_str() {
            "count" => {
                load_rel_rdi!(asm, rule_i, rel_seq);
                call_abs!(asm, packed_agg_count as usize);
                // eax = u32 count; store to var_slot (always emit, count=0 is valid)
                dynasm!(asm; mov [rbp + var_slot(result_var)], eax);
            }
            name @ ("sum" | "min" | "max") => {
                let bv = *agg.bound_vars.first()
                    .ok_or_else(|| format!("agg '{name}': no bound vars"))?;
                let col = agg.args.iter().position(|a| matches!(a, CAggArg::Var(v) if *v == bv))
                    .ok_or_else(|| format!("agg '{name}': bound var not found in args"))?;
                let fn_addr = match name {
                    "sum" => packed_agg_sum_i32 as usize,
                    "max" => packed_agg_max_i32 as usize,
                    "min" => packed_agg_min_i32 as usize,
                    _ => unreachable!(),
                };
                load_rel_rdi!(asm, rule_i, rel_seq);
                dynasm!(asm; mov esi, col as i32);
                call_abs!(asm, fn_addr);
                // rax = i64 result; skip if AGG_EMPTY (empty relation)
                dynasm!(asm
                    ; mov rcx, QWORD AGG_EMPTY
                    ; cmp rax, rcx
                    ; je =>variant_exit
                );
                // Store low 32 bits (bit-cast i32→u32) to var_slot
                dynasm!(asm; mov [rbp + var_slot(result_var)], eax);
            }
            other => return Err(format!("unsupported aggregator '{other}' in asm backend")),
        }
    }
    Ok(())
}

/// Emit one rule variant (full or recent).
#[allow(clippy::too_many_arguments)]
fn emit_rule_variant(
    asm: &mut Assembler,
    rule_i: usize,
    clauses: &[CClause],
    heads: &[CHeadClause],
    rule_conds: &[CExpr],
    negations: &[CAggregation],
    aggs: &[CAggregation],
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
    jit_rel_data_grow_addr: usize,
    jit_tuple_set_grow_addr: usize,
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
        emit_not_probes(asm, negations, 0, rule_i, &var_locs, variant_exit)?;
        emit_aggregations(asm, aggs, 0, negations.len(), rule_i, variant_exit)?;
        emit_heads(asm, heads, rule_i, var_count, max_head_arity, max_depth, pti, &var_locs, use_jit_native, rule_head_base, jit_rel_data_grow_addr, jit_tuple_set_grow_addr)?;
        dynasm!(asm; =>variant_exit);
        return Ok(());
    }

    // Precompute NativeCtx array pointer cache for native path.
    // Caches total_rels, head_rels, head_total_rels from NativeCtx into stack slots
    // so inner loops use 1 load per pointer chain instead of 3 (ctx → array → rel).
    if use_jit_native {
        let tr_slot  = native_total_rels_slot(var_count, max_head_arity, max_depth);
        let hr_slot  = native_head_rels_slot(var_count, max_head_arity, max_depth);
        let htr_slot = native_head_total_rels_slot(var_count, max_head_arity, max_depth);
        dynasm!(asm
            ; mov r8, [rbp + CTX_SLOT]
            ; mov rax, [r8 +  8]   // NativeCtx.total_rels (offset 8)
            ; mov [rbp + tr_slot], rax
            ; mov rax, [r8 + 24]   // NativeCtx.head_rels (offset 24)
            ; mov [rbp + hr_slot], rax
            ; mov rax, [r8 + 56]   // NativeCtx.head_total_rels (offset 56)
            ; mov [rbp + htr_slot], rax
        );
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
        jit_rel_data_grow_addr,
        jit_tuple_set_grow_addr,
        negations,
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
    rules: &[AsmRuleRef<'_>],
    var_count: usize,
    advance_addr: usize,
    packed_try_insert_addr: usize,
    packed_count_addr: usize,
    packed_data_ptr_addr: usize,
    packed_recent_ptr_addr: usize,
    use_jit_native: bool,
    jit_rel_data_grow_addr: usize,
    jit_tuple_set_grow_addr: usize,
) -> Result<AsmStratum, String> {
    // ── Eligibility ──────────────────────────────────────────────────────────
    if var_count > 10_000 {
        return Err(format!("asm: var_count {var_count} exceeds limit of 10000; stack frame would be too large"));
    }
    for (ri, (clauses, heads, conds, nots, aggs)) in rules.iter().enumerate() {
        // Native path does not support negation or aggregation.
        if use_jit_native && (!nots.is_empty() || !aggs.is_empty()) {
            return Err(format!("asm native: rule {ri} has negation clauses (unsupported)"));
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
        // IDB inner clauses are handled correctly by the non-native asm linked-list path.
        // The rejection that existed here was removed after confirming that commit f2901fc
        // (support arbitrary expressions in bound clause arg positions) fixed the prior
        // incorrect behavior (hangs, extreme slowness) that motivated the restriction.
    }

    let max_head_arity = rules
        .iter()
        .flat_map(|(_, hs, _, _, _)| hs.iter().map(|h| h.args.len()))
        .max()
        .unwrap_or(1);
    let max_head_arity = max_head_arity.max(1);

    // max_depth = maximum nesting depth (= max inner clauses count = max_clauses - 1)
    // Use at least 1 to keep the frame layout simple.
    let max_depth = rules
        .iter()
        .map(|(clauses, _, _, _, _)| clauses.len().saturating_sub(1))
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
        for (clauses, _, _, _, _) in rules.iter() {
            bases.push(offset);
            offset += clauses.len() * 2;
        }
        bases
    };

    // Each rule contributes head_count entries to the flat head_specs / head_rels arrays.
    let rule_head_bases: Vec<usize> = {
        let mut bases = Vec::with_capacity(rules.len());
        let mut offset = 0usize;
        for (_, heads, _, _, _) in rules.iter() {
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
    for (rule_i, (clauses, heads, conds, nots, aggs)) in rules.iter().enumerate() {
        emit_rule_variant(
            &mut asm, rule_i, clauses, heads, conds, nots, aggs, None,
            var_count, max_head_arity, max_depth,
            packed_try_insert_addr, packed_count_addr,
            packed_data_ptr_addr, packed_recent_ptr_addr,
            rule_handle_bases[rule_i],
            rule_head_bases[rule_i],
            use_jit_native,
            jit_rel_data_grow_addr,
            jit_tuple_set_grow_addr,
        )?;
    }

    // advance(ctx) → al
    dynasm!(asm; mov rdi, [rbp + CTX_SLOT]);
    call_abs!(asm, advance_addr);
    dynasm!(asm; test al, al; jz =>exit_lbl);

    // Recent-variant loop
    let inner_hdr = asm.new_dynamic_label();
    dynasm!(asm; =>inner_hdr);

    for (rule_i, (clauses, heads, conds, nots, aggs)) in rules.iter().enumerate() {
        for clause_seq in 0..clauses.len() {
            emit_rule_variant(
                &mut asm, rule_i, clauses, heads, conds, nots, aggs, Some(clause_seq),
                var_count, max_head_arity, max_depth,
                packed_try_insert_addr, packed_count_addr,
                packed_data_ptr_addr, packed_recent_ptr_addr,
                rule_handle_bases[rule_i],
                rule_head_bases[rule_i],
                use_jit_native,
                jit_rel_data_grow_addr,
                jit_tuple_set_grow_addr,
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
///
/// `rules` is a 4-tuple per rule: `(clauses, heads, conditions, not_clauses)`.
/// Not-clauses are anti-join checks emitted before head insertion.
pub fn codegen_stratum_asm(
    rules: &[AsmRuleRef<'_>],
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
        0, // jit_rel_data_grow_addr unused in non-native path
        0, // jit_tuple_set_grow_addr unused in non-native path
    )
}

/// Native entry point: reads scan data and writes heads directly via `JitRelData` fields
/// (Steps 4 + 5).
///
/// The generated function takes `*mut StratumStage4NativeCtx` as its argument.
/// `advance_native_addr` is the address of `jit_advance_native`.
/// Head writes are fully inline: `jit_rel_data_grow` and `jit_tuple_set_grow` are called
/// only on capacity overflow; no Rust call occurs in the common case.
///
/// Note: negation clauses are NOT supported in the native path. Rules with negation
/// cause the native compile to fail, falling back to the non-native asm path.
pub fn codegen_stratum_asm_native(
    rules: &[(&[CClause], &[CHeadClause], &[CExpr])],
    var_count: usize,
    advance_native_addr: usize,
) -> Result<AsmStratum, String> {
    // Native path has no negation support; wrap 3-tuples with empty not-slices.
    let rules_with_nots: Vec<AsmRuleRef<'_>> = rules
        .iter()
        .map(|(c, h, e)| (*c, *h, *e, &[] as &[CAggregation], &[] as &[CAggregation]))
        .collect();
    codegen_stratum_asm_inner(
        &rules_with_nots, var_count,
        advance_native_addr,
        0, // packed_try_insert_addr unused in native path (Step 5)
        0, // packed_count_addr unused in native path
        0, // packed_data_ptr_addr unused in native path
        0, // packed_recent_ptr_addr unused in native path
        true,
        storage::jit_rel_data_grow as usize,
        storage::jit_tuple_set_grow as usize,
    )
}
