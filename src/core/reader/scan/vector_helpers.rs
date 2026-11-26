// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Vector extraction from Arrow columns and scalar distance
//! dispatch for flat rescoring.

use super::*;

impl HybridReader {
    pub(super) fn extract_vector_at(col: &dyn Array, i: usize, expected_dim: usize) -> Option<Vec<f32>> {
        if col.is_null(i) {
            return None;
        }
        let item: arrow::array::ArrayRef = match col.data_type() {
            arrow::datatypes::DataType::FixedSizeList(_, _) => {
                let list = col
                    .as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            arrow::datatypes::DataType::List(_) => {
                let list = col.as_any().downcast_ref::<arrow::array::ListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            arrow::datatypes::DataType::LargeList(_) => {
                let list = col
                    .as_any()
                    .downcast_ref::<arrow::array::LargeListArray>()?;
                if list.is_null(i) {
                    return None;
                }
                list.value(i)
            }
            _ => return None,
        };
        if item.len() != expected_dim {
            return None;
        }
        if let Some(floats) = item.as_any().downcast_ref::<arrow::array::Float32Array>() {
            Some(floats.values().to_vec())
        } else {
            item.as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|doubles| doubles.values().iter().map(|&d| d as f32).collect())
        }
    }

    pub(super) fn compute_vector_distance_f32(v: &[f32], q: &[f32], metric: VectorMetric) -> f32 {
        match metric {
            VectorMetric::L2 => crate::core::index::distance::l2_distance(v, q),
            VectorMetric::Cosine => crate::core::index::distance::cosine_distance(v, q),
            VectorMetric::InnerProduct => -crate::core::index::distance::dot_product(v, q),
            VectorMetric::L1 => crate::core::index::distance::l1_distance(v, q),
            VectorMetric::Hamming => crate::core::index::distance::hamming_distance(v, q),
            VectorMetric::Jaccard => crate::core::index::distance::jaccard_distance(v, q),
        }
    }
}
