//! Built-in aggregators for the interpreter.
//!
//! Aggregators consume an iterator of bound-variable slices and return
//! zero or more result tuples. Streaming: no intermediate Vec is needed.

use crate::value::Value;

/// Result of an aggregation: zero or more result tuples.
pub type AggResult = Vec<Vec<Value>>;

/// Apply a named aggregator to a stream of bound-variable tuples.
///
/// Unknown aggregator names produce an empty result (no tuples) and emit a
/// warning to stderr. The return type is `AggResult` (not `Result`) so callers
/// cannot distinguish "unknown aggregator" from "aggregator matched zero tuples"
/// at the type level.
pub fn apply_aggregator<'a>(name: &str, values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    match name {
        "min" => agg_min(values),
        "max" => agg_max(values),
        "sum" => agg_sum(values),
        "count" => agg_count(values),
        "mean" => agg_mean(values),
        "not" => agg_not(values),
        // TODO: return Result<AggResult, EvalError> to use EvalError::UnknownAggregator
        _ => {
            eprintln!("warning: unknown aggregator '{name}', producing no results");
            vec![]
        }
    }
}

/// `min(x)` - returns the minimum value.
fn agg_min<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut min_val: Option<&Value> = None;
    for tuple in values {
        if let Some(val) = tuple.first() {
            if let Some(current_min) = min_val {
                if val.try_cmp(current_min).is_some_and(|o| o.is_lt()) {
                    min_val = Some(val);
                }
            } else {
                min_val = Some(val);
            }
        }
    }

    min_val.map(|v| vec![vec![v.clone()]]).unwrap_or_default()
}

/// `max(x)` - returns the maximum value.
fn agg_max<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut max_val: Option<&Value> = None;
    for tuple in values {
        if let Some(val) = tuple.first() {
            if let Some(current_max) = max_val {
                if val.try_cmp(current_max).is_some_and(|o| o.is_gt()) {
                    max_val = Some(val);
                }
            } else {
                max_val = Some(val);
            }
        }
    }

    max_val.map(|v| vec![vec![v.clone()]]).unwrap_or_default()
}

/// `sum(x)` - returns the sum of values.
fn agg_sum<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut acc: Option<Value> = None;
    for tuple in values {
        if let Some(val) = tuple.first() {
            acc = Some(match acc {
                Some(a) => a.add(val).unwrap_or(a),
                None => val.clone(),
            });
        }
    }

    acc.map(|v| vec![vec![v]]).unwrap_or_default()
}

/// `count()` - returns the number of tuples.
fn agg_count(values: impl Iterator<Item = impl Sized>) -> AggResult {
    vec![vec![Value::I64(values.count() as i64)]]
}

/// `mean(x)` - returns the average of numeric values.
///
/// Returns `Value::I64` when all inputs are integers and the mean is exact,
/// `Value::F64` when any input is a float or the integer mean has a remainder.
/// Non-numeric values are counted but contribute zero to the sum.
fn agg_mean<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut count = 0usize;
    let mut int_sum = 0i64;
    let mut has_float = false;
    let mut float_sum = 0.0f64;
    for tuple in values {
        count += 1;
        if let Some(val) = tuple.first() {
            if let Some(n) = val.as_i64() {
                int_sum += n;
                float_sum += n as f64;
            } else if let Some(f) = val.as_f64() {
                has_float = true;
                float_sum += f;
            }
        }
    }

    if count == 0 {
        return vec![];
    }

    if has_float {
        vec![vec![Value::F64(crate::value::OrderedFloat(
            float_sum / count as f64,
        ))]]
    } else if int_sum % count as i64 == 0 {
        vec![vec![Value::I64(int_sum / count as i64)]]
    } else {
        vec![vec![Value::F64(crate::value::OrderedFloat(
            float_sum / count as f64,
        ))]]
    }
}

/// `not()` - negation aggregator. Returns `()` if no matching tuples.
fn agg_not<'a>(mut values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    if values.next().is_none() {
        vec![vec![Value::Unit]]
    } else {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(3)],
            vec![Value::I32(1)],
            vec![Value::I32(2)],
        ];
        let result = agg_min(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I32(1)]]);
    }

    #[test]
    fn test_min_empty() {
        let values: Vec<Vec<Value>> = vec![];
        let result = agg_min(values.iter().map(|v| v.as_slice()));
        assert!(result.is_empty());
    }

    #[test]
    fn test_max() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(3)],
            vec![Value::I32(1)],
            vec![Value::I32(2)],
        ];
        let result = agg_max(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I32(3)]]);
    }

    #[test]
    fn test_sum() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(1)],
            vec![Value::I32(2)],
            vec![Value::I32(3)],
        ];
        let result = agg_sum(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I32(6)]]);
    }

    #[test]
    fn test_count() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(10)],
            vec![Value::I32(20)],
            vec![Value::I32(30)],
        ];
        let result = agg_count(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I64(3)]]);
    }

    #[test]
    fn test_count_empty() {
        let values: Vec<Vec<Value>> = vec![];
        let result = agg_count(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I64(0)]]);
    }

    #[test]
    fn test_mean_exact_int() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(2)],
            vec![Value::I32(4)],
            vec![Value::I32(6)],
        ];
        let result = agg_mean(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I64(4)]]);
    }

    #[test]
    fn test_mean_inexact_int() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(1)],
            vec![Value::I32(2)],
        ];
        let result = agg_mean(values.iter().map(|v| v.as_slice()));
        assert_eq!(result.len(), 1);
        if let Value::F64(crate::value::OrderedFloat(v)) = result[0][0] {
            assert!((v - 1.5).abs() < f64::EPSILON);
        } else {
            panic!("expected f64 for inexact integer mean");
        }
    }

    #[test]
    fn test_mean_with_floats() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(2)],
            vec![Value::F64(crate::value::OrderedFloat(4.0))],
            vec![Value::I32(6)],
        ];
        let result = agg_mean(values.iter().map(|v| v.as_slice()));
        assert_eq!(result.len(), 1);
        if let Value::F64(crate::value::OrderedFloat(v)) = result[0][0] {
            assert!((v - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected f64 when floats are present");
        }
    }

    #[test]
    fn test_not_empty() {
        let values: Vec<Vec<Value>> = vec![];
        let result = agg_not(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::Unit]]);
    }

    #[test]
    fn test_not_nonempty() {
        let values: Vec<Vec<Value>> = vec![vec![Value::I32(1)]];
        let result = agg_not(values.iter().map(|v| v.as_slice()));
        assert!(result.is_empty());
    }
}
