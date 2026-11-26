// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Row group pruning from Parquet column statistics.

use super::*;

impl HybridReader {
    /// Prune row groups using Parquet column statistics (min/max/null_count).
    ///
    /// This is the "Parquet Internal Pruning" stage of the Iceberg metadata pipeline:
    /// after manifest-level pruning, we use per-row-group stats to skip entire row groups
    /// before any vector computation. This can eliminate 90-99% of I/O when data is
    /// clustered by the filter columns.
    ///
    /// Returns a list of row group indices that might match the filter. If the filter
    /// references no columns with available stats, all row groups are returned.
    pub(super) fn prune_row_groups(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        filter: &FilterExpr,
    ) -> Vec<usize> {
        let schema = metadata.file_metadata().schema_descr();
        let num_rgs = metadata.num_row_groups();

        let conditions = filter.extract_and_conditions();
        if conditions.is_empty() {
            return (0..num_rgs).collect();
        }

        let mut surviving = Vec::with_capacity(num_rgs);

        for rg_idx in 0..num_rgs {
            let rg = metadata.row_group(rg_idx);
            let mut might_match = true;

            for cond in &conditions {
                if cond.negated {
                    continue;
                }
                let col_name = &cond.column;
                let col_pos = schema
                    .columns()
                    .iter()
                    .position(|c| c.name() == col_name.as_str());

                if let Some(pos) = col_pos {
                    let col_meta = rg.column(pos);
                    let stats = col_meta.statistics();

                    if let Some(s) = stats {
                        // If all rows are null, skip this row group
                        let null_count = s.null_count_opt().unwrap_or(0);
                        if null_count >= rg.num_rows() as u64 {
                            might_match = false;
                            break;
                        }

                        // Only prune if stats are available
                        if Self::has_stats(s) && !Self::row_group_might_match_condition(s, cond) {
                            might_match = false;
                            break;
                        }
                    }
                }
            }

            if might_match {
                surviving.push(rg_idx);
            }
        }

        surviving
    }

    /// Check whether a Statistics enum has usable min/max data.
    pub(super) fn has_stats(stats: &ParquetStats) -> bool {
        match stats {
            ParquetStats::Int32(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Int64(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Float(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Double(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::ByteArray(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            ParquetStats::Boolean(s) => s.min_opt().is_some() || s.max_opt().is_some(),
            _ => false,
        }
    }

    /// Extract min/max as f64 from numeric statistics variants.
    pub(super) fn extract_min_max_as_f64(stats: &ParquetStats) -> Option<(f64, f64)> {
        match stats {
            ParquetStats::Int32(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Int64(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Float(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min as f64, max as f64))
                } else {
                    None
                }
            }
            ParquetStats::Double(s) => {
                if let (Some(&min), Some(&max)) = (s.min_opt(), s.max_opt()) {
                    Some((min, max))
                } else {
                    None
                }
            }
            ParquetStats::ByteArray(s) => {
                // Try to parse byte array as numeric string
                if let (Some(min_bytes), Some(max_bytes)) = (s.min_opt(), s.max_opt()) {
                    if let (Ok(min_f), Ok(max_f)) = (
                        std::str::from_utf8(min_bytes.as_ref()).ok()?.parse::<f64>(),
                        std::str::from_utf8(max_bytes.as_ref()).ok()?.parse::<f64>(),
                    ) {
                        return Some((min_f, max_f));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract min/max as strings from ByteArray statistics.
    pub(super) fn extract_min_max_as_str(stats: &ParquetStats) -> Option<(String, String)> {
        if let ParquetStats::ByteArray(s) = stats {
            if let (Some(min_bytes), Some(max_bytes)) = (s.min_opt(), s.max_opt()) {
                let min_str = std::str::from_utf8(min_bytes.as_ref()).ok()?.to_string();
                let max_str = std::str::from_utf8(max_bytes.as_ref()).ok()?.to_string();
                return Some((min_str, max_str));
            }
        }
        None
    }

    /// Check if a row group's column statistics might overlap with a filter condition.
    pub(super) fn row_group_might_match_condition(
        stats: &ParquetStats,
        filter: &crate::core::planner::QueryFilter,
    ) -> bool {
        // Try numeric comparison first
        if let Some((rg_min, rg_max)) = Self::extract_min_max_as_f64(stats) {
            // Check min bound: if row group's max < filter's min, prune
            if let Some(filter_min) = &filter.min {
                if let Some(fmin) = filter_min.as_f64() {
                    let too_small = if filter.min_inclusive {
                        rg_max < fmin
                    } else {
                        rg_max <= fmin
                    };
                    if too_small {
                        return false;
                    }
                }
            }

            // Check max bound: if row group's min > filter's max, prune
            if let Some(filter_max) = &filter.max {
                if let Some(fmax) = filter_max.as_f64() {
                    let too_large = if filter.max_inclusive {
                        rg_min > fmax
                    } else {
                        rg_min >= fmax
                    };
                    if too_large {
                        return false;
                    }
                }
            }

            // Check IN list
            if let Some(values) = &filter.values {
                let mut possible = false;
                for v in values {
                    if let Some(vf) = v.as_f64() {
                        let in_range = !(vf < rg_min || vf > rg_max);
                        if in_range {
                            possible = true;
                            break;
                        }
                    }
                }
                if !possible {
                    return false;
                }
            }

            return true;
        }

        // Fall back to string comparison
        if let Some((rg_min_str, rg_max_str)) = Self::extract_min_max_as_str(stats) {
            // Check min bound
            if let Some(filter_min) = &filter.min {
                if let Some(fmin) = filter_min.as_str() {
                    let cmp = rg_max_str.as_str().cmp(fmin);
                    let too_small = if filter.min_inclusive {
                        cmp == std::cmp::Ordering::Less
                    } else {
                        cmp != std::cmp::Ordering::Greater
                    };
                    if too_small {
                        return false;
                    }
                }
            }

            // Check max bound
            if let Some(filter_max) = &filter.max {
                if let Some(fmax) = filter_max.as_str() {
                    let cmp = rg_min_str.as_str().cmp(fmax);
                    let too_large = if filter.max_inclusive {
                        cmp == std::cmp::Ordering::Greater
                    } else {
                        cmp != std::cmp::Ordering::Less
                    };
                    if too_large {
                        return false;
                    }
                }
            }

            // Check IN list
            if let Some(values) = &filter.values {
                let mut possible = false;
                for v in values {
                    if let Some(vs) = v.as_str() {
                        let in_range = !(vs < rg_min_str.as_str() || vs > rg_max_str.as_str());
                        if in_range {
                            possible = true;
                            break;
                        }
                    }
                }
                if !possible {
                    return false;
                }
            }

            return true;
        }

        // No usable stats — cannot prune
        true
    }
}
