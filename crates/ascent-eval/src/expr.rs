//! Tree-walking evaluator for Rust expressions (syn::Expr).
//!
//! Evaluates the subset of Rust expressions that appear in Ascent programs:
//! literals, variables, binary/unary ops, ranges, tuples, method calls.

use syn::Expr;

use crate::eval::{Bindings, TypeRegistry, VarInterner};
use crate::value::Value;

/// Evaluate a syn expression with the given variable bindings.
pub fn eval_expr(expr: &Expr, bindings: &Bindings, interner: &VarInterner) -> Option<Value> {
    eval_expr_inner(expr, bindings, None, interner)
}

/// Evaluate a syn expression with bindings and a type registry for custom constructors.
pub fn eval_expr_with_registry(
    expr: &Expr,
    bindings: &Bindings,
    registry: &TypeRegistry,
    interner: &VarInterner,
) -> Option<Value> {
    eval_expr_inner(expr, bindings, Some(registry), interner)
}

fn eval_expr_inner(
    expr: &Expr,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    match expr {
        Expr::Lit(lit) => eval_lit(&lit.lit),
        Expr::Path(p) => {
            if let Some(ident) = p.path.get_ident() {
                let name = ident.to_string();
                // Check for boolean literals that parse as paths
                match name.as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => {
                        let var_id = interner.intern(&name);
                        bindings.get(&var_id).cloned()
                    }
                }
            } else {
                None
            }
        }
        Expr::Binary(bin) => eval_binary(bin, bindings, registry, interner),
        Expr::Unary(unary) => eval_unary(unary, bindings, registry, interner),
        Expr::Paren(paren) => eval_expr_inner(&paren.expr, bindings, registry, interner),
        Expr::Range(range) => eval_range(range, bindings, registry, interner),
        Expr::Tuple(tuple) => eval_tuple(tuple, bindings, registry, interner),
        Expr::Call(call) => eval_call(call, bindings, registry, interner),
        Expr::MethodCall(mc) => eval_method_call(mc, bindings, registry, interner),
        Expr::Reference(r) => {
            // For &x in Datalog context, just evaluate x
            eval_expr_inner(&r.expr, bindings, registry, interner)
        }
        Expr::Cast(cast) => eval_cast(cast, bindings, registry, interner),
        Expr::If(if_expr) => eval_if(if_expr, bindings, registry, interner),
        Expr::Block(block) => eval_block(block, bindings, registry, interner),
        Expr::Array(arr) => {
            let values: Option<Vec<Value>> = arr
                .elems
                .iter()
                .map(|e| eval_expr_inner(e, bindings, registry, interner))
                .collect();
            values.map(Value::tuple)
        }
        _ => None,
    }
}

/// Evaluate a literal.
pub(crate) fn eval_lit(lit: &syn::Lit) -> Option<Value> {
    match lit {
        syn::Lit::Int(i) => {
            let suffix = i.suffix();
            match suffix {
                "i8" => i.base10_parse::<i8>().ok().map(Value::I8),
                "i16" => i.base10_parse::<i16>().ok().map(Value::I16),
                "i32" => i.base10_parse::<i32>().ok().map(Value::I32),
                "i64" => i.base10_parse::<i64>().ok().map(Value::I64),
                "i128" => i.base10_parse::<i128>().ok().map(Value::I128),
                "isize" => i.base10_parse::<isize>().ok().map(Value::Isize),
                "u8" => i.base10_parse::<u8>().ok().map(Value::U8),
                "u16" => i.base10_parse::<u16>().ok().map(Value::U16),
                "u32" => i.base10_parse::<u32>().ok().map(Value::U32),
                "u64" => i.base10_parse::<u64>().ok().map(Value::U64),
                "u128" => i.base10_parse::<u128>().ok().map(Value::U128),
                "usize" => i.base10_parse::<usize>().ok().map(Value::Usize),
                // No suffix: default to i32 (Rust's default integer type)
                "" => i.base10_parse::<i32>().ok().map(Value::I32),
                _ => None,
            }
        }
        syn::Lit::Float(f) => {
            let suffix = f.suffix();
            match suffix {
                "f32" => f
                    .base10_parse::<f32>()
                    .ok()
                    .map(|v| Value::F32(crate::value::OrderedFloat(v))),
                _ => f
                    .base10_parse::<f64>()
                    .ok()
                    .map(|v| Value::F64(crate::value::OrderedFloat(v))),
            }
        }
        syn::Lit::Bool(b) => Some(Value::Bool(b.value)),
        syn::Lit::Str(s) => Some(Value::string(s.value())),
        syn::Lit::Char(c) => Some(Value::Char(c.value())),
        _ => None,
    }
}

/// Evaluate a binary expression.
fn eval_binary(
    bin: &syn::ExprBinary,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    // Short-circuit for && and ||
    match bin.op {
        syn::BinOp::And(_) => {
            let left = eval_expr_inner(&bin.left, bindings, registry, interner)?;
            if !left.as_bool()? {
                return Some(Value::Bool(false));
            }
            let right = eval_expr_inner(&bin.right, bindings, registry, interner)?;
            return Some(Value::Bool(right.as_bool()?));
        }
        syn::BinOp::Or(_) => {
            let left = eval_expr_inner(&bin.left, bindings, registry, interner)?;
            if left.as_bool()? {
                return Some(Value::Bool(true));
            }
            let right = eval_expr_inner(&bin.right, bindings, registry, interner)?;
            return Some(Value::Bool(right.as_bool()?));
        }
        _ => {}
    }

    let left = eval_expr_inner(&bin.left, bindings, registry, interner)?;
    let right = eval_expr_inner(&bin.right, bindings, registry, interner)?;

    match bin.op {
        syn::BinOp::Add(_) => left.add(&right),
        syn::BinOp::Sub(_) => left.sub(&right),
        syn::BinOp::Mul(_) => left.mul(&right),
        syn::BinOp::Div(_) => left.div(&right),
        syn::BinOp::Rem(_) => left.rem(&right),
        syn::BinOp::Eq(_) => Some(Value::Bool(left == right)),
        syn::BinOp::Ne(_) => Some(Value::Bool(left != right)),
        syn::BinOp::Lt(_) => left.partial_cmp_val(&right).map(|o| Value::Bool(o.is_lt())),
        syn::BinOp::Le(_) => left
            .partial_cmp_val(&right)
            .map(|o| Value::Bool(!o.is_gt())),
        syn::BinOp::Gt(_) => left.partial_cmp_val(&right).map(|o| Value::Bool(o.is_gt())),
        syn::BinOp::Ge(_) => left
            .partial_cmp_val(&right)
            .map(|o| Value::Bool(!o.is_lt())),
        syn::BinOp::BitAnd(_) => left.bitand(&right),
        syn::BinOp::BitOr(_) => left.bitor(&right),
        syn::BinOp::BitXor(_) => left.bitxor(&right),
        syn::BinOp::Shl(_) => left.shl(&right),
        syn::BinOp::Shr(_) => left.shr(&right),
        _ => None,
    }
}

/// Evaluate a unary expression.
fn eval_unary(
    unary: &syn::ExprUnary,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let val = eval_expr_inner(&unary.expr, bindings, registry, interner)?;
    match unary.op {
        syn::UnOp::Neg(_) => val.neg(),
        syn::UnOp::Not(_) => val.not(),
        syn::UnOp::Deref(_) => Some(val), // In Datalog context, deref is identity
        _ => None,
    }
}

/// Evaluate a range expression.
fn eval_range(
    range: &syn::ExprRange,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let start = range
        .start
        .as_ref()
        .and_then(|e| eval_expr_inner(e, bindings, registry, interner));
    let end = range
        .end
        .as_ref()
        .and_then(|e| eval_expr_inner(e, bindings, registry, interner));
    let inclusive = matches!(range.limits, syn::RangeLimits::Closed(_));

    match (start, end) {
        (Some(s), Some(e)) => Some(Value::Range {
            start: Box::new(s),
            end: Box::new(e),
            inclusive,
        }),
        _ => None,
    }
}

/// Evaluate a tuple expression.
fn eval_tuple(
    tuple: &syn::ExprTuple,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let values: Option<Vec<_>> = tuple
        .elems
        .iter()
        .map(|e| eval_expr_inner(e, bindings, registry, interner))
        .collect();
    values.map(Value::tuple)
}

/// Evaluate a function call (limited support).
fn eval_call(
    call: &syn::ExprCall,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    if let Expr::Path(p) = &*call.func
        && let Some(ident) = p.path.get_ident()
    {
        let name = ident.to_string();

        // Built-in single-arg constructors
        if call.args.len() == 1 {
            let val = eval_expr_inner(&call.args[0], bindings, registry, interner)?;
            match name.as_str() {
                "Some" => return Some(Value::Option(Some(Box::new(val)))),
                "Dual" => return Some(Value::Dual(Box::new(val))),
                _ => {}
            }
        }

        // Check type registry for custom constructors
        if let Some(reg) = registry
            && let Some(constructor) = reg.get(&name)
        {
            let args: Option<Vec<Value>> = call
                .args
                .iter()
                .map(|e| eval_expr_inner(e, bindings, registry, interner))
                .collect();
            return args.and_then(|a| constructor(&a));
        }
    }
    None
}

/// Evaluate a method call (limited support).
fn eval_method_call(
    mc: &syn::ExprMethodCall,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let receiver = eval_expr_inner(&mc.receiver, bindings, registry, interner)?;
    let method = mc.method.to_string();

    match method.as_str() {
        "clone" => Some(receiver),
        "abs" => receiver.abs(),
        "eq" if mc.args.len() == 1 => {
            let arg = eval_expr_inner(&mc.args[0], bindings, registry, interner)?;
            Some(Value::Bool(receiver == arg))
        }
        _ => None,
    }
}

/// Evaluate a cast expression (e.g., `x as i32`).
fn eval_cast(
    cast: &syn::ExprCast,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let val = eval_expr_inner(&cast.expr, bindings, registry, interner)?;

    if let syn::Type::Path(tp) = &*cast.ty
        && let Some(ident) = tp.path.get_ident()
    {
        return val.cast_to(&ident.to_string());
    }
    None
}

/// Evaluate an if expression.
fn eval_if(
    if_expr: &syn::ExprIf,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    let cond = eval_expr_inner(&if_expr.cond, bindings, registry, interner)?;
    if cond.as_bool()? {
        eval_block_stmts(&if_expr.then_branch, bindings, registry, interner)
    } else if let Some((_, else_branch)) = &if_expr.else_branch {
        eval_expr_inner(else_branch, bindings, registry, interner)
    } else {
        Some(Value::Unit)
    }
}

/// Evaluate a block expression.
fn eval_block(
    block: &syn::ExprBlock,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    eval_block_stmts(&block.block, bindings, registry, interner)
}

/// Evaluate the last expression in a block.
fn eval_block_stmts(
    block: &syn::Block,
    bindings: &Bindings,
    registry: Option<&TypeRegistry>,
    interner: &VarInterner,
) -> Option<Value> {
    // Only handle blocks with a single trailing expression
    if block.stmts.len() == 1
        && let syn::Stmt::Expr(expr, _) = &block.stmts[0]
    {
        return eval_expr_inner(expr, bindings, registry, interner);
    }
    None
}

/// Expand a range Value into individual values.
pub fn expand_range(range: &Value) -> Option<Vec<Value>> {
    match range {
        Value::Range {
            start,
            end,
            inclusive,
        } => {
            let s = start.as_i64()?;
            let e = end.as_i64()?;
            let values: Vec<Value> = if *inclusive {
                (s..=e).map(|v| coerce_i64(v, start)).collect()
            } else {
                (s..e).map(|v| coerce_i64(v, start)).collect()
            };
            Some(values)
        }
        // Array/tuple literals expand to their elements
        Value::Tuple(elements) => Some(elements.as_ref().clone()),
        _ => None,
    }
}

/// Coerce an i64 back to the same type as the reference value.
fn coerce_i64(v: i64, reference: &Value) -> Value {
    match reference {
        Value::I32(_) => Value::I32(v as i32),
        Value::I8(_) => Value::I8(v as i8),
        Value::I16(_) => Value::I16(v as i16),
        Value::I64(_) => Value::I64(v),
        Value::U8(_) => Value::U8(v as u8),
        Value::U16(_) => Value::U16(v as u16),
        Value::U32(_) => Value::U32(v as u32),
        Value::U64(_) => Value::U64(v as u64),
        Value::Usize(_) => Value::Usize(v as usize),
        Value::Isize(_) => Value::Isize(v as isize),
        _ => Value::I32(v as i32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(input: &str) -> Value {
        let interner = VarInterner::default();
        let expr: syn::Expr = syn::parse_str(input).unwrap();
        eval_expr(&expr, &Bindings::default(), &interner).unwrap()
    }

    #[test]
    fn test_literals() {
        assert_eq!(eval("42"), Value::I32(42));
        assert_eq!(eval("42i64"), Value::I64(42));
        assert_eq!(eval("true"), Value::Bool(true));
        assert_eq!(eval("false"), Value::Bool(false));
        assert_eq!(eval("'a'"), Value::Char('a'));
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(eval("1 + 2"), Value::I32(3));
        assert_eq!(eval("10 - 3"), Value::I32(7));
        assert_eq!(eval("4 * 5"), Value::I32(20));
        assert_eq!(eval("10 / 3"), Value::I32(3));
        assert_eq!(eval("10 % 3"), Value::I32(1));
    }

    #[test]
    fn test_comparison() {
        assert_eq!(eval("1 == 1"), Value::Bool(true));
        assert_eq!(eval("1 != 2"), Value::Bool(true));
        assert_eq!(eval("1 < 2"), Value::Bool(true));
        assert_eq!(eval("2 > 1"), Value::Bool(true));
        assert_eq!(eval("1 <= 1"), Value::Bool(true));
        assert_eq!(eval("1 >= 2"), Value::Bool(false));
    }

    #[test]
    fn test_logical() {
        assert_eq!(eval("true && false"), Value::Bool(false));
        assert_eq!(eval("true || false"), Value::Bool(true));
        assert_eq!(eval("!true"), Value::Bool(false));
    }

    #[test]
    fn test_variables() {
        let interner = VarInterner::default();
        let mut bindings = Bindings::default();
        bindings.insert(interner.intern("x"), Value::I32(10));
        bindings.insert(interner.intern("y"), Value::I32(20));

        let expr_add: syn::Expr = syn::parse_str("x + y").unwrap();
        assert_eq!(
            eval_expr(&expr_add, &bindings, &interner).unwrap(),
            Value::I32(30)
        );
        let expr_mul: syn::Expr = syn::parse_str("x * 2").unwrap();
        assert_eq!(
            eval_expr(&expr_mul, &bindings, &interner).unwrap(),
            Value::I32(20)
        );
        let expr_lt: syn::Expr = syn::parse_str("x < y").unwrap();
        assert_eq!(
            eval_expr(&expr_lt, &bindings, &interner).unwrap(),
            Value::Bool(true)
        );
    }

    #[test]
    fn test_unary() {
        assert_eq!(eval("-5"), Value::I32(-5));
        assert_eq!(eval("!false"), Value::Bool(true));
    }

    #[test]
    fn test_range() {
        let range = eval("0..10");
        let values = expand_range(&range).unwrap();
        assert_eq!(values.len(), 10);
        assert_eq!(values[0], Value::I32(0));
        assert_eq!(values[9], Value::I32(9));
    }

    #[test]
    fn test_range_inclusive() {
        let range = eval("0..=9");
        let values = expand_range(&range).unwrap();
        assert_eq!(values.len(), 10);
        assert_eq!(values[9], Value::I32(9));
    }

    #[test]
    fn test_tuple() {
        assert_eq!(
            eval("(1, 2, 3)"),
            Value::tuple(vec![Value::I32(1), Value::I32(2), Value::I32(3)])
        );
    }

    #[test]
    fn test_nested_expr() {
        assert_eq!(eval("(1 + 2) * 3"), Value::I32(9));
        assert_eq!(eval("10 % 3 == 1"), Value::Bool(true));
    }

    #[test]
    fn test_cast() {
        assert_eq!(eval("42i64 as i32"), Value::I32(42));
    }
}
