// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Equality delete application and candidate row id validation
//! against filters and delete masks.

use super::*;

impl HybridReader {
    pub(crate) fn apply_eq_deletes_keep_mask(
        batch: &RecordBatch,
        equality_deletes: &[EqualityDelete],
    ) -> Result<arrow::array::BooleanArray> {
        if equality_deletes.is_empty() || batch.num_rows() == 0 {
            return Ok(arrow::array::BooleanArray::from(vec![
                true;
                batch.num_rows()
            ]));
        }
        let mut keep_mask = arrow::array::BooleanArray::from(vec![true; batch.num_rows()]);
        for delete in equality_deletes {
            if let Some(col) = batch.column_by_name(&delete.column_name) {
                let delete_mask = check_is_in(col, &delete.values)?;
                let not_delete = arrow::compute::not(&delete_mask)?;
                keep_mask = arrow::compute::and(&keep_mask, &not_delete)?;
            }
        }
        Ok(keep_mask)
    }

    pub(super) async fn validate_row_ids_against_filter_and_eq_deletes(
        &self,
        candidate_bitmap: &RoaringBitmap,
        filter: Option<&FilterExpr>,
        equality_deletes: &[EqualityDelete],
    ) -> Result<RoaringBitmap> {
        if candidate_bitmap.is_empty() || (filter.is_none() && equality_deletes.is_empty()) {
            return Ok(candidate_bitmap.clone());
        }
        let pq_path = self.resolve_object_path("parquet");
        let (meta, size) = self.load_cached_parquet_meta(&pq_path).await?;
        let total_rows = meta.metadata().file_metadata().num_rows() as usize;
        if total_rows == 0 {
            return Ok(RoaringBitmap::new());
        }
        let mut target_bitmap = candidate_bitmap.clone();
        target_bitmap &= RoaringBitmap::from_iter(0..(total_rows as u32));
        if target_bitmap.is_empty() {
            return Ok(RoaringBitmap::new());
        }

        let reader = self
            .parquet_reader(self.data_store(), &pq_path, size as u64)
            .await;
        let arrow_meta = (*meta).clone();
        let mut builder = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, arrow_meta);

        let groups = Self::row_groups_covering(builder.metadata(), &target_bitmap);
        let num_groups = builder.metadata().num_row_groups();
        if !groups.is_empty() && groups.len() < num_groups {
            let mut shifted = RoaringBitmap::new();
            let mut base: u32 = 0;
            for (rg, group_start) in &groups {
                let rows = builder.metadata().row_group(*rg).num_rows() as u32;
                for id in target_bitmap.iter() {
                    let id = id as usize;
                    if id >= *group_start && id < *group_start + rows as usize {
                        shifted.insert(base + (id as u32 - *group_start as u32));
                    }
                }
                base += rows;
            }
            builder = builder.with_row_groups(groups.iter().map(|(rg, _)| *rg).collect());
            let selection = self.bitmap_to_row_selection(&shifted, base as usize);
            builder = builder.with_row_selection(selection);
        } else {
            let selection = self.bitmap_to_row_selection(&target_bitmap, total_rows);
            builder = builder.with_row_selection(selection);
        }

        let mut required_cols: Vec<String> = Vec::new();
        for del in equality_deletes {
            if !required_cols.contains(&del.column_name) {
                required_cols.push(del.column_name.clone());
            }
        }
        let filter_cols = filter.map(|f| f.required_columns()).unwrap_or_default();
        for col in &filter_cols {
            if !required_cols.contains(col) {
                required_cols.push(col.clone());
            }
        }

        let parquet_schema = builder.metadata().file_metadata().schema_descr();
        let file_arrow_schema = builder.schema();
        let column_indices: Vec<usize> = required_cols
            .iter()
            .filter_map(|name| file_arrow_schema.index_of(name).ok())
            .collect();
        let projection = ProjectionMask::roots(parquet_schema, column_indices);
        builder = builder.with_projection(projection);

        let mut stream = builder.build()?;
        let row_ids: Vec<u32> = target_bitmap.iter().collect();
        let mut surviving = RoaringBitmap::new();
        let mut current_offset = 0usize;
        let planner = crate::core::planner::QueryPlanner::new();

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let num_rows = batch.num_rows();
            let avail = row_ids.len().saturating_sub(current_offset);
            let rows_to_process = std::cmp::min(num_rows, avail);
            if rows_to_process == 0 {
                break;
            }
            let batch = if rows_to_process < num_rows {
                batch.slice(0, rows_to_process)
            } else {
                batch
            };

            let eq_keep = if equality_deletes.is_empty() {
                None
            } else {
                Some(Self::apply_eq_deletes_keep_mask(&batch, equality_deletes)?)
            };

            let filter_mask = if let Some(expr) = filter {
                let eval_batch = if filter_cols
                    .iter()
                    .any(|c| batch.column_by_name(c).is_none())
                {
                    self.pad_missing_schema_columns(batch, &filter_cols).await?
                } else {
                    batch
                };
                let coerced = expr.coerce_literals(&eval_batch.schema());
                Some(planner.evaluate_expr(&eval_batch, &coerced)?)
            } else {
                None
            };

            for i in 0..rows_to_process {
                if let Some(ref eq_k) = eq_keep {
                    if !eq_k.is_valid(i) || !eq_k.value(i) {
                        continue;
                    }
                }
                if let Some(ref m) = filter_mask {
                    if !m.is_valid(i) || !m.value(i) {
                        continue;
                    }
                }
                surviving.insert(row_ids[current_offset + i]);
            }
            current_offset += rows_to_process;
        }

        Ok(surviving)
    }
}
