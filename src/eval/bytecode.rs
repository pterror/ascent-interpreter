//! Flat bytecode compiler and VM for expression evaluation.
//!
//! Compiles `CExpr` trees into a linear sequence of stack-machine instructions,
//! eliminating recursive function calls, Box indirection, and branch misprediction
//! from tree-walking. The bytecode VM is a tight loop over a flat `Vec<Op>`.
//!
//! Only "computational" expressions (arithmetic, comparisons, logic, variables,
//! literals) are bytecoded. Complex expressions (Range, Tuple, Call, MethodCall,
//! Cast, Array, Dynamic) fall back to the tree-walk evaluator.

use crate::eval::compiled::{CBinOp, CExpr, CUnOp, eval_binary_op};
use crate::eval::engine::{Bindings, VarId};
use crate::eval::value::Value;

// ─── Instruction set ────────────────────────────────────────────────

/// Bytecode instruction for the expression stack machine.
#[derive(Debug, Clone)]
pub(crate) enum Op {
    /// Push a constant from the constant pool.
    LoadConst(u16),
    /// Push a variable value from bindings.
    LoadVar(VarId),
    /// Pop two values, apply binary op, push result.
    Binary(CBinOp),
    /// Pop one value, apply unary op, push result.
    Unary(CUnOp),
    /// Pop top, convert to bool. If false, jump to absolute target; else fall through.
    JumpIfFalse(u16),
    /// Pop top, convert to bool. If true, jump to absolute target; else fall through.
    JumpIfTrue(u16),
    /// Unconditional jump to absolute target.
    Jump(u16),
    /// Push a boolean literal (used for short-circuit results).
    PushBool(bool),
}

// ─── Program ────────────────────────────────────────────────────────

/// A compiled bytecode program for expression evaluation.
#[derive(Debug, Clone)]
pub(crate) struct BytecodeProgram {
    pub ops: Vec<Op>,
    pub constants: Vec<Value>,
    /// All variable IDs referenced by this program (for optimization passes).
    pub referenced_vars: Vec<VarId>,
}

impl BytecodeProgram {
    fn new() -> Self {
        Self {
            ops: Vec::new(),
            constants: Vec::new(),
            referenced_vars: Vec::new(),
        }
    }

    fn add_constant(&mut self, val: Value) -> u16 {
        // Reuse existing constant if identical
        for (i, c) in self.constants.iter().enumerate() {
            if c == &val {
                return i as u16;
            }
        }
        let idx = self.constants.len();
        if idx > u16::MAX as usize {
            panic!(
                "bytecode constant pool overflow: more than {} constants",
                u16::MAX
            );
        }
        self.constants.push(val);
        idx as u16
    }

    fn add_var(&mut self, var_id: VarId) {
        if !self.referenced_vars.contains(&var_id) {
            self.referenced_vars.push(var_id);
        }
    }
}

// ─── Compilation ────────────────────────────────────────────────────

/// Try to compile a CExpr tree into bytecode.
///
/// Returns `Some(program)` if the expression can be fully represented in bytecode
/// (no Range/Tuple/Call/MethodCall/Cast/Array/Dynamic nodes). Returns `None` if
/// the expression contains unsupported nodes or is too simple to benefit from
/// bytecoding.
pub(crate) fn try_compile_to_bytecode(expr: &CExpr) -> Option<BytecodeProgram> {
    // Only worth bytecoding if there's actual tree structure to flatten
    if !has_recursive_structure(expr) {
        return None;
    }
    let mut program = BytecodeProgram::new();
    emit(expr, &mut program)?;
    Some(program)
}

/// Check if the expression has recursive structure that benefits from bytecoding.
/// Flat expressions (Literal, Var, VarBinVar, etc.) are already optimally handled
/// by the existing match-based dispatch.
fn has_recursive_structure(expr: &CExpr) -> bool {
    matches!(expr, CExpr::Binary(..) | CExpr::Unary(..))
}

/// Emit bytecode for an expression. Returns None if the expression contains
/// unsupported nodes.
fn emit(expr: &CExpr, program: &mut BytecodeProgram) -> Option<()> {
    match expr {
        CExpr::Literal(val) => {
            let idx = program.add_constant(val.clone());
            program.ops.push(Op::LoadConst(idx));
        }
        CExpr::Var(id) | CExpr::DerefVar(id) => {
            program.add_var(*id);
            program.ops.push(Op::LoadVar(*id));
        }
        CExpr::VarBinVar(op, a, b) => {
            program.add_var(*a);
            program.add_var(*b);
            program.ops.push(Op::LoadVar(*a));
            program.ops.push(Op::LoadVar(*b));
            program.ops.push(Op::Binary(*op));
        }
        CExpr::VarBinLit(op, a, v) => {
            program.add_var(*a);
            program.ops.push(Op::LoadVar(*a));
            let idx = program.add_constant(v.clone());
            program.ops.push(Op::LoadConst(idx));
            program.ops.push(Op::Binary(*op));
        }
        CExpr::LitBinVar(op, v, b) => {
            program.add_var(*b);
            let idx = program.add_constant(v.clone());
            program.ops.push(Op::LoadConst(idx));
            program.ops.push(Op::LoadVar(*b));
            program.ops.push(Op::Binary(*op));
        }
        CExpr::Binary(op, left, right) => {
            if matches!(op, CBinOp::And | CBinOp::Or) {
                return emit_short_circuit(*op, left, right, program);
            }
            emit(left, program)?;
            emit(right, program)?;
            program.ops.push(Op::Binary(*op));
        }
        CExpr::Unary(op, inner) => {
            emit(inner, program)?;
            program.ops.push(Op::Unary(*op));
        }
        // Unsupported: fall back to tree-walk
        CExpr::Range { .. }
        | CExpr::Tuple(_)
        | CExpr::Call(..)
        | CExpr::MethodCall(..)
        | CExpr::Cast(..)
        | CExpr::Array(_)
        | CExpr::Dynamic(_)
        | CExpr::Bytecode(_) => return None,
    }
    Some(())
}

/// Emit bytecode for short-circuit && / || with jump instructions.
///
/// For `a && b`:
///   [eval a]  JumpIfFalse(L1)  [eval b]  Jump(L2)  L1: PushBool(false)  L2:
///
/// For `a || b`:
///   [eval a]  JumpIfTrue(L1)  [eval b]  Jump(L2)  L1: PushBool(true)  L2:
fn emit_short_circuit(
    op: CBinOp,
    left: &CExpr,
    right: &CExpr,
    program: &mut BytecodeProgram,
) -> Option<()> {
    // Evaluate left operand
    emit(left, program)?;

    // Conditional jump (placeholder target)
    let jump_idx = program.ops.len();
    match op {
        CBinOp::And => program.ops.push(Op::JumpIfFalse(0)),
        CBinOp::Or => program.ops.push(Op::JumpIfTrue(0)),
        _ => unreachable!(),
    }

    // Evaluate right operand (only reached if left didn't short-circuit)
    emit(right, program)?;

    // Skip over the short-circuit push
    let skip_idx = program.ops.len();
    program.ops.push(Op::Jump(0));

    // Short-circuit target: push the short-circuit result
    let short_circuit_target = program.ops.len() as u16;
    match op {
        CBinOp::And => program.ops.push(Op::PushBool(false)),
        CBinOp::Or => program.ops.push(Op::PushBool(true)),
        _ => unreachable!(),
    }

    let end = program.ops.len() as u16;

    // Patch jump targets
    match &mut program.ops[jump_idx] {
        Op::JumpIfFalse(target) | Op::JumpIfTrue(target) => *target = short_circuit_target,
        _ => unreachable!(),
    }
    match &mut program.ops[skip_idx] {
        Op::Jump(target) => *target = end,
        _ => unreachable!(),
    }

    Some(())
}

// ─── VM ─────────────────────────────────────────────────────────────

/// Evaluate a bytecode program against the given variable bindings.
#[inline]
pub(crate) fn eval_bytecode(program: &BytecodeProgram, bindings: &Bindings) -> Option<Value> {
    let ops = &program.ops;
    let constants = &program.constants;
    let mut stack: Vec<Value> = Vec::with_capacity(8);
    let mut ip: usize = 0;

    while ip < ops.len() {
        match &ops[ip] {
            Op::LoadConst(idx) => {
                stack.push(constants[*idx as usize].clone());
            }
            Op::LoadVar(var_id) => {
                stack.push(bindings.get(var_id)?.clone());
            }
            Op::Binary(op) => {
                let right = stack.pop().expect("bytecode stack underflow");
                let left = stack.pop().expect("bytecode stack underflow");
                stack.push(eval_binary_op(*op, &left, &right)?);
            }
            Op::Unary(op) => {
                let val = stack.pop().expect("bytecode stack underflow");
                let result = match op {
                    CUnOp::Neg => val.neg()?,
                    CUnOp::Not => val.not()?,
                    CUnOp::Deref => val,
                };
                stack.push(result);
            }
            Op::JumpIfFalse(target) => {
                let val = stack.pop().expect("bytecode stack underflow");
                if !val.as_bool()? {
                    ip = *target as usize;
                    continue;
                }
            }
            Op::JumpIfTrue(target) => {
                let val = stack.pop().expect("bytecode stack underflow");
                if val.as_bool()? {
                    ip = *target as usize;
                    continue;
                }
            }
            Op::Jump(target) => {
                ip = *target as usize;
                continue;
            }
            Op::PushBool(val) => {
                stack.push(Value::Bool(*val));
            }
        }
        ip += 1;
    }

    stack.pop()
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::engine::VarInterner;

    fn make_bindings(interner: &VarInterner, vars: &[(&str, Value)]) -> Bindings {
        let mut bindings = Bindings::new(vars.len());
        for (name, val) in vars {
            let id = interner.intern(name);
            bindings.insert(id, val.clone());
        }
        bindings
    }

    #[test]
    fn test_simple_addition() {
        // x + 1
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let expr = CExpr::Binary(
            CBinOp::Add,
            Box::new(CExpr::Var(x)),
            Box::new(CExpr::Literal(Value::I32(1))),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();
        let bindings = make_bindings(&interner, &[("x", Value::I32(10))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::I32(11)));
    }

    #[test]
    fn test_nested_arithmetic() {
        // (x + 1) * (y - 2)
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let y = interner.intern("y");
        let expr = CExpr::Binary(
            CBinOp::Mul,
            Box::new(CExpr::Binary(
                CBinOp::Add,
                Box::new(CExpr::Var(x)),
                Box::new(CExpr::Literal(Value::I32(1))),
            )),
            Box::new(CExpr::Binary(
                CBinOp::Sub,
                Box::new(CExpr::Var(y)),
                Box::new(CExpr::Literal(Value::I32(2))),
            )),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();
        let bindings = make_bindings(&interner, &[("x", Value::I32(5)), ("y", Value::I32(10))]);
        // (5 + 1) * (10 - 2) = 6 * 8 = 48
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::I32(48)));
    }

    #[test]
    fn test_short_circuit_and() {
        // x > 0 && y > 0
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let y = interner.intern("y");
        let expr = CExpr::Binary(
            CBinOp::And,
            Box::new(CExpr::VarBinLit(CBinOp::Gt, x, Value::I32(0))),
            Box::new(CExpr::VarBinLit(CBinOp::Gt, y, Value::I32(0))),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();

        // Both true
        let bindings = make_bindings(&interner, &[("x", Value::I32(5)), ("y", Value::I32(3))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(true)));

        // Left false (short-circuit)
        let bindings = make_bindings(&interner, &[("x", Value::I32(-1)), ("y", Value::I32(3))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(false)));

        // Right false
        let bindings = make_bindings(&interner, &[("x", Value::I32(5)), ("y", Value::I32(-1))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(false)));
    }

    #[test]
    fn test_short_circuit_or() {
        // x > 0 || y > 0
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let y = interner.intern("y");
        let expr = CExpr::Binary(
            CBinOp::Or,
            Box::new(CExpr::VarBinLit(CBinOp::Gt, x, Value::I32(0))),
            Box::new(CExpr::VarBinLit(CBinOp::Gt, y, Value::I32(0))),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();

        // Left true (short-circuit)
        let bindings = make_bindings(&interner, &[("x", Value::I32(5)), ("y", Value::I32(-1))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(true)));

        // Right true
        let bindings = make_bindings(&interner, &[("x", Value::I32(-1)), ("y", Value::I32(3))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(true)));

        // Both false
        let bindings = make_bindings(&interner, &[("x", Value::I32(-1)), ("y", Value::I32(-1))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(false)));
    }

    #[test]
    fn test_unary_neg() {
        // -(x + 1)
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let expr = CExpr::Unary(
            CUnOp::Neg,
            Box::new(CExpr::Binary(
                CBinOp::Add,
                Box::new(CExpr::Var(x)),
                Box::new(CExpr::Literal(Value::I32(1))),
            )),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();
        let bindings = make_bindings(&interner, &[("x", Value::I32(5))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::I32(-6)));
    }

    #[test]
    fn test_not_compiled_for_simple_expr() {
        // Literal, Var, VarBinVar should NOT be bytecoded
        let interner = VarInterner::default();
        let x = interner.intern("x");

        assert!(try_compile_to_bytecode(&CExpr::Literal(Value::I32(42))).is_none());
        assert!(try_compile_to_bytecode(&CExpr::Var(x)).is_none());
        assert!(try_compile_to_bytecode(&CExpr::VarBinVar(CBinOp::Add, x, x)).is_none());
    }

    #[test]
    fn test_constant_dedup() {
        // x + 1 > 1 — the constant 1 should be stored once
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let expr = CExpr::Binary(
            CBinOp::Gt,
            Box::new(CExpr::VarBinLit(CBinOp::Add, x, Value::I32(1))),
            Box::new(CExpr::Literal(Value::I32(1))),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();
        assert_eq!(bc.constants.len(), 1); // only one constant: 1
    }

    #[test]
    fn test_chained_and() {
        // x > 0 && x < 10 && y > 0
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let y = interner.intern("y");
        let expr = CExpr::Binary(
            CBinOp::And,
            Box::new(CExpr::Binary(
                CBinOp::And,
                Box::new(CExpr::VarBinLit(CBinOp::Gt, x, Value::I32(0))),
                Box::new(CExpr::VarBinLit(CBinOp::Lt, x, Value::I32(10))),
            )),
            Box::new(CExpr::VarBinLit(CBinOp::Gt, y, Value::I32(0))),
        );
        let bc = try_compile_to_bytecode(&expr).unwrap();

        let bindings = make_bindings(&interner, &[("x", Value::I32(5)), ("y", Value::I32(3))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(true)));

        // x out of range
        let bindings = make_bindings(&interner, &[("x", Value::I32(15)), ("y", Value::I32(3))]);
        assert_eq!(eval_bytecode(&bc, &bindings), Some(Value::Bool(false)));
    }

    #[test]
    fn test_unsupported_falls_back() {
        // Call expression — should return None
        let expr = CExpr::Call("foo".to_string(), vec![CExpr::Literal(Value::I32(1))]);
        assert!(try_compile_to_bytecode(&expr).is_none());

        // Binary with a Call child — should return None
        let interner = VarInterner::default();
        let x = interner.intern("x");
        let expr = CExpr::Binary(
            CBinOp::Add,
            Box::new(CExpr::Var(x)),
            Box::new(CExpr::Call("foo".to_string(), vec![])),
        );
        assert!(try_compile_to_bytecode(&expr).is_none());
    }
}
