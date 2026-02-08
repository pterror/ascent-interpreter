//! Built-in aggregators for the interpreter.
//!
//! Aggregators consume an iterator of bound-variable slices and return
//! zero or more result tuples. Streaming: no intermediate Vec is needed.

use crate::value::Value;

/// Result of an aggregation: zero or more result tuples.
pub type AggResult = Vec<Vec<Value>>;

/// Apply a named aggregator to a stream of bound-variable tuples.
pub fn apply_aggregator<'a>(name: &str, values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    match name {
        "min" => agg_min(values),
        "max" => agg_max(values),
        "sum" => agg_sum(values),
        "count" => agg_count(values),
        "mean" => agg_mean(values),
        "not" => agg_not(values),
        _ => vec![], // Unknown aggregator
    }
}

/// `min(x)` - returns the minimum value.
fn agg_min<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut min_val: Option<&Value> = None;
    for tuple in values {
        if let Some(val) = tuple.first() {
            if let Some(current_min) = min_val {
                if val.partial_cmp_val(current_min).is_some_and(|o| o.is_lt()) {
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
                if val.partial_cmp_val(current_max).is_some_and(|o| o.is_gt()) {
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
    vec![vec![Value::I32(values.count() as i32)]]
}

/// `mean(x)` - returns the average as f64.
fn agg_mean<'a>(values: impl Iterator<Item = &'a [Value]>) -> AggResult {
    let mut count = 0usize;
    let mut sum = 0.0f64;
    for tuple in values {
        count += 1;
        if let Some(val) = tuple.first()
            && let Some(n) = val.as_i64()
        {
            sum += n as f64;
        }
    }

    if count == 0 {
        return vec![];
    }

    vec![vec![Value::F64(crate::value::OrderedFloat(
        sum / count as f64,
    ))]]
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
        assert_eq!(result, vec![vec![Value::I32(3)]]);
    }

    #[test]
    fn test_count_empty() {
        let values: Vec<Vec<Value>> = vec![];
        let result = agg_count(values.iter().map(|v| v.as_slice()));
        assert_eq!(result, vec![vec![Value::I32(0)]]);
    }

    #[test]
    fn test_mean() {
        let values: Vec<Vec<Value>> = vec![
            vec![Value::I32(2)],
            vec![Value::I32(4)],
            vec![Value::I32(6)],
        ];
        let result = agg_mean(values.iter().map(|v| v.as_slice()));
        assert_eq!(result.len(), 1);
        if let Value::F64(crate::value::OrderedFloat(v)) = result[0][0] {
            assert!((v - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected f64");
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
