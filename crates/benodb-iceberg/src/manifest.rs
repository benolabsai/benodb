// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use anyhow::Result;
use apache_avro::{types::Value as AvroValue, Reader};
use std::io::Read;

use super::types::{IcebergDataFile, IcebergManifestEntry, IcebergManifestListEntry};
use super::value::avro_to_json;

/// Read a manifest list (snap-*.avro) and return its entries
pub fn read_manifest_list<R: Read>(reader: R) -> Result<Vec<IcebergManifestListEntry>> {
    let avro_reader = Reader::new(reader)?;
    let mut entries = Vec::new();

    for record in avro_reader {
        let value = record?;
        if let AvroValue::Record(fields) = value {
            let mut manifest_path = String::new();
            let mut manifest_length = 0;
            let mut partition_spec_id = 0;
            let mut added_snapshot_id = 0;
            let mut content = 0;
            let mut sequence_number = 0;
            let mut min_sequence_number = 0;
            let mut added_files_count = 0;
            let mut existing_files_count = 0;
            let mut deleted_files_count = 0;
            let mut added_rows_count = 0;
            let mut existing_rows_count = 0;
            let mut deleted_rows_count = 0;

            for (name, val) in fields {
                match name.as_str() {
                    "manifest_path" => {
                        if let AvroValue::String(s) = val {
                            manifest_path = s;
                        }
                    }
                    "manifest_length" => {
                        if let AvroValue::Long(s) = val {
                            manifest_length = s;
                        }
                    }
                    "partition_spec_id" => {
                        if let AvroValue::Int(s) = val {
                            partition_spec_id = s;
                        }
                    }
                    "added_snapshot_id" => {
                        if let AvroValue::Long(s) = val {
                            added_snapshot_id = s;
                        }
                    }
                    "content" => {
                        if let AvroValue::Int(s) = val {
                            content = s;
                        }
                    }
                    "sequence_number" => {
                        if let AvroValue::Long(s) = val {
                            sequence_number = s;
                        }
                    }
                    "min_sequence_number" => {
                        if let AvroValue::Long(s) = val {
                            min_sequence_number = s;
                        }
                    }
                    "added_data_files_count" => {
                        if let AvroValue::Int(s) = val {
                            added_files_count = s;
                        }
                    }
                    "existing_data_files_count" => {
                        if let AvroValue::Int(s) = val {
                            existing_files_count = s;
                        }
                    }
                    "deleted_data_files_count" => {
                        if let AvroValue::Int(s) = val {
                            deleted_files_count = s;
                        }
                    }
                    "added_rows_count" => {
                        if let AvroValue::Long(s) = val {
                            added_rows_count = s;
                        }
                    }
                    "existing_rows_count" => {
                        if let AvroValue::Long(s) = val {
                            existing_rows_count = s;
                        }
                    }
                    "deleted_rows_count" => {
                        if let AvroValue::Long(s) = val {
                            deleted_rows_count = s;
                        }
                    }
                    _ => {}
                }
            }

            entries.push(IcebergManifestListEntry {
                manifest_path,
                manifest_length,
                partition_spec_id,
                added_snapshot_id,
                content,
                sequence_number,
                min_sequence_number,
                added_files_count,
                existing_files_count,
                deleted_files_count,
                added_rows_count,
                existing_rows_count,
                deleted_rows_count,
            });
        }
    }
    Ok(entries)
}

/// Read a manifest file (*.avro) and return its entries
pub fn read_manifest<R: Read>(reader: R) -> Result<Vec<IcebergManifestEntry>> {
    let avro_reader = Reader::new(reader)?;
    let mut entries = Vec::new();

    for record in avro_reader {
        let value = record?;
        if let AvroValue::Record(fields) = value {
            let mut status = 0;
            let mut snapshot_id = None;
            let mut data_file = None;

            for (name, val) in fields {
                match name.as_str() {
                    "status" => {
                        if let AvroValue::Int(s) = val {
                            status = s;
                        }
                    }
                    "snapshot_id" => {
                        if let AvroValue::Long(s) = val {
                            snapshot_id = Some(s);
                        }
                    }
                    "data_file" => {
                        if let AvroValue::Record(df_fields) = val {
                            data_file = Some(parse_data_file(df_fields)?);
                        }
                    }
                    _ => {}
                }
            }

            if let Some(df) = data_file {
                entries.push(IcebergManifestEntry {
                    status,
                    snapshot_id,
                    data_file: df,
                    delete_files: Vec::new(),
                });
            }
        }
    }
    Ok(entries)
}

fn parse_data_file(fields: Vec<(String, AvroValue)>) -> Result<IcebergDataFile> {
    let mut content = 0;
    let mut file_path = String::new();
    let mut file_format = String::new();
    let mut partition = Vec::new();
    let mut record_count = 0;
    let mut file_size_in_bytes = 0;

    let mut column_sizes = None;
    let mut value_counts = None;
    let mut null_value_counts = None;
    let mut nan_value_counts = None;
    let mut lower_bounds = None;
    let mut upper_bounds = None;
    let mut equality_ids = None;

    // V3 Deletion Vector fields
    let mut referenced_data_file = None;
    let mut content_offset = None;
    let mut content_size_in_bytes = None;
    let mut index_files = None;
    let mut file_checksum = None;
    let mut first_row_id = None;

    for (name, val) in fields {
        match name.as_str() {
            "content" => {
                if let AvroValue::Int(c) = val {
                    content = c;
                }
            }
            "file_path" => {
                if let AvroValue::String(s) = val {
                    file_path = s;
                }
            }
            "file_format" => {
                if let AvroValue::String(s) = val {
                    file_format = s;
                }
            }
            "partition" => {
                if let AvroValue::Record(p_fields) = val {
                    partition = p_fields.into_iter().map(|(_, v)| avro_to_json(v)).collect();
                }
            }
            "record_count" => {
                if let AvroValue::Long(c) = val {
                    record_count = c;
                }
            }
            "file_size_in_bytes" => {
                if let AvroValue::Long(s) = val {
                    file_size_in_bytes = s;
                }
            }
            "column_sizes" => column_sizes = parse_map_int_long(val),
            "value_counts" => value_counts = parse_map_int_long(val),
            "null_value_counts" => null_value_counts = parse_map_int_long(val),
            "nan_value_counts" => nan_value_counts = parse_map_int_long(val),
            "lower_bounds" => lower_bounds = parse_map_int_bytes(val),
            "upper_bounds" => upper_bounds = parse_map_int_bytes(val),
            "equality_ids" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::Array(items) = inner {
                    let mut ids = Vec::new();
                    for item in items {
                        if let AvroValue::Int(id) = item {
                            ids.push(id);
                        }
                    }
                    equality_ids = Some(ids);
                }
            }
            // V3 Deletion Vector fields
            "referenced_data_file" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::String(s) = inner {
                    referenced_data_file = Some(s);
                }
            }
            "content_offset" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::Long(o) = inner {
                    content_offset = Some(o);
                }
            }
            "content_size_in_bytes" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::Long(s) = inner {
                    content_size_in_bytes = Some(s);
                }
            }
            "index_files" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::String(s) = inner {
                    index_files = Some(s);
                }
            }
            "file_checksum" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::String(s) = inner {
                    file_checksum = Some(s);
                }
            }
            "first_row_id" => {
                let inner = if let AvroValue::Union(_, b) = val {
                    *b
                } else {
                    val
                };
                if let AvroValue::Long(id) = inner {
                    first_row_id = Some(id);
                }
            }
            _ => {}
        }
    }

    Ok(IcebergDataFile {
        content,
        file_path,
        file_format,
        partition,
        record_count,
        file_size_in_bytes,
        column_sizes,
        value_counts,
        null_value_counts,
        nan_value_counts,
        lower_bounds,
        upper_bounds,
        equality_ids,
        referenced_data_file,
        content_offset,
        content_size_in_bytes,
        index_files,
        file_checksum,
        first_row_id,
    })
}

/// Unwrap the nullable-union wrapper Avro puts around every `["null", T]` field.
///
/// These fields are declared nullable in the manifest schema, so apache-avro
/// hands them back as `Union(1, Box(Array(..)))` rather than a bare `Array`.
/// Without this the parsers silently returned `None` for every stats map, which
/// left `column_stats` empty and made the planner's stats pruning inert.
fn unwrap_nullable_union(val: AvroValue) -> AvroValue {
    match val {
        AvroValue::Union(_, inner) => *inner,
        other => other,
    }
}

fn parse_map_int_long(val: AvroValue) -> Option<std::collections::HashMap<i32, i64>> {
    if let AvroValue::Array(items) = unwrap_nullable_union(val) {
        let mut map = std::collections::HashMap::new();
        for item in items {
            if let AvroValue::Record(fields) = item {
                let mut key = 0;
                let mut value = 0;
                for (k, v) in fields {
                    match k.as_str() {
                        "key" => {
                            if let AvroValue::Int(i) = v {
                                key = i;
                            }
                        }
                        "value" => {
                            if let AvroValue::Long(l) = v {
                                value = l;
                            }
                        }
                        _ => {}
                    }
                }
                map.insert(key, value);
            }
        }
        return Some(map);
    }
    None
}

fn parse_map_int_bytes(val: AvroValue) -> Option<std::collections::HashMap<i32, Vec<u8>>> {
    if let AvroValue::Array(items) = unwrap_nullable_union(val) {
        let mut map = std::collections::HashMap::new();
        for item in items {
            if let AvroValue::Record(fields) = item {
                let mut key = 0;
                let mut value = Vec::new();
                for (k, v) in fields {
                    match k.as_str() {
                        "key" => {
                            if let AvroValue::Int(i) = v {
                                key = i;
                            }
                        }
                        "value" => {
                            if let AvroValue::Bytes(b) = v {
                                value = b;
                            }
                        }
                        _ => {}
                    }
                }
                map.insert(key, value);
            }
        }
        return Some(map);
    }
    None
}
