//! Built-in aggregators for the interpreter.
//!
//! Each aggregator takes a list of bound-variable tuples and returns
//! zero or more result tuples.

use crate::value::Value;

/// Result of an aggregation: zero or more result tuples.
pub type AggResult = Vec<Vec<Value>>;

/// Apply a named aggregator to collected values.
pub fn apply_aggregator(name: &str, values: Vec<Vec<Value>>) -> AggResult {
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
fn agg_min(values: Vec<Vec<Value>>) -> AggResult {
    if values.is_empty() {
        return vec![];
    }

    let mut min_val: Option<&Value> = None;
    for tuple in &values {
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
fn agg_max(values: Vec<Vec<Value>>) -> AggResult {
    if values.is_empty() {
        return vec![];
    }

    let mut max_val: Option<&Value> = None;
    for tuple in &values {
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
fn agg_sum(values: Vec<Vec<Value>>) -> AggResult {
    if values.is_empty() {
        return vec![];
    }

    let mut acc: Option<Value> = None;
    for tuple in &values {
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
fn agg_count(values: Vec<Vec<Value>>) -> AggResult {
    vec![vec![Value::I32(values.len() as i32)]]
}

/// `mean(x)` - returns the average as f64.
fn agg_mean(values: Vec<Vec<Value>>) -> AggResult {
    if values.is_empty() {
        return vec![];
    }

    let count = values.len() as f64;
    let mut sum = 0.0f64;
    for tuple in &values {
        if let Some(val) = tuple.first()
            && let Some(n) = val.as_i64()
        {
            sum += n as f64;
        }
    }

    vec![vec![Value::F64(crate::value::OrderedFloat(sum / count))]]
}

/// `not()` - negation aggregator. Returns `()` if no matching tuples.
fn agg_not(values: Vec<Vec<Value>>) -> AggResult {
    if values.is_empty() {
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
        let values = vec![
            vec![Value::I32(3)],
            vec![Value::I32(1)],
            vec![Value::I32(2)],
        ];
        let result = agg_min(values);
        assert_eq!(result, vec![vec![Value::I32(1)]]);
    }

    #[test]
    fn test_min_empty() {
        let result = agg_min(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_max() {
        let values = vec![
            vec![Value::I32(3)],
            vec![Value::I32(1)],
            vec![Value::I32(2)],
        ];
        let result = agg_max(values);
        assert_eq!(result, vec![vec![Value::I32(3)]]);
    }

    #[test]
    fn test_sum() {
        let values = vec![
            vec![Value::I32(1)],
            vec![Value::I32(2)],
            vec![Value::I32(3)],
        ];
        let result = agg_sum(values);
        assert_eq!(result, vec![vec![Value::I32(6)]]);
    }

    #[test]
    fn test_count() {
        let values = vec![
            vec![Value::I32(10)],
            vec![Value::I32(20)],
            vec![Value::I32(30)],
        ];
        let result = agg_count(values);
        assert_eq!(result, vec![vec![Value::I32(3)]]);
    }

    #[test]
    fn test_count_empty() {
        let result = agg_count(vec![]);
        assert_eq!(result, vec![vec![Value::I32(0)]]);
    }

    #[test]
    fn test_mean() {
        let values = vec![
            vec![Value::I32(2)],
            vec![Value::I32(4)],
            vec![Value::I32(6)],
        ];
        let result = agg_mean(values);
        assert_eq!(result.len(), 1);
        if let Value::F64(crate::value::OrderedFloat(v)) = result[0][0] {
            assert!((v - 4.0).abs() < f64::EPSILON);
        } else {
            panic!("expected f64");
        }
    }

    #[test]
    fn test_not_empty() {
        let result = agg_not(vec![]);
        assert_eq!(result, vec![vec![Value::Unit]]);
    }

    #[test]
    fn test_not_nonempty() {
        let result = agg_not(vec![vec![Value::I32(1)]]);
        assert!(result.is_empty());
    }
}
