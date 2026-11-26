// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Puffin container support.
//!
//! The container format — framing, footer, blob storage, and the
//! deletion-vector codec — lives in the `benodb-iceberg` crate and is
//! re-exported here. `PuffinIndexWriter` is engine vocabulary: it builds the
//! `IndexFile` inventory entries the manifest carries, so it lives with the
//! engine and delegates container work to the crate's writer.

pub use benodb_iceberg::puffin::*;

use crate::core::manifest::types::IndexFile;
use anyhow::Result;
use object_store::ObjectStore;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Seek, Write};
use uuid::Uuid;

/// High-level Puffin compound index writer that aggregates multiple secondary index
/// artifacts (vector, lexical BM25, roaring bitmaps, CSR graphs) for a segment into
/// a single `.puffin` container file with strict lineage bindings.
pub struct PuffinIndexWriter<W: Write + Seek> {
    writer: PuffinWriter<W>,
    entries: Vec<IndexFile>,
    container_file_path: String,
    source_snapshot_id: i64,
    source_data_checksum: String,
    source_record_count: i64,
}

impl<W: Write + Seek> PuffinIndexWriter<W> {
    pub fn new(
        writer: W,
        container_file_path: String,
        source_snapshot_id: i64,
        source_data_checksum: String,
        source_record_count: i64,
    ) -> Result<Self> {
        let p_writer = PuffinWriter::new(writer)?;
        Ok(Self {
            writer: p_writer,
            entries: Vec::new(),
            container_file_path,
            source_snapshot_id,
            source_data_checksum,
            source_record_count,
        })
    }

    /// Add an index artifact payload as a blob in the Puffin container and generate
    /// its corresponding `IndexFile` metadata entry.
    pub fn add_index_blob(
        &mut self,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        data: &[u8],
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let index_checksum = format!("{:x}", hasher.finalize());

        let (offset, length) = self.writer.add_blob(
            blob_type.to_string(),
            fields,
            self.source_snapshot_id,
            1, // sequence number
            data,
            properties,
        )?;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(offset),
            blob_length: Some(length),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        self.entries.last().ok_or_else(|| {
            anyhow::anyhow!("entry was just pushed")
        })
    }

    /// Add an index artifact from an on-disk file, streaming its payload directly
    /// into the Puffin container in chunks without buffering the whole file in RAM.
    pub fn add_index_blob_from_file(
        &mut self,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        file_path: &str,
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::with_capacity(65536, file);
        let (offset, length, index_checksum) = self.writer.add_blob_from_reader(
            blob_type.to_string(),
            fields,
            self.source_snapshot_id,
            1, // sequence number
            reader,
            properties,
        )?;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(offset),
            blob_length: Some(length),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        self.entries.last().ok_or_else(|| {
            anyhow::anyhow!("entry was just pushed")
        })
    }

    /// Stream an existing uncompressed blob range from `store` into this Puffin
    /// container in bounded chunks without buffering large blobs in RAM.
    #[allow(clippy::too_many_arguments)]
    pub async fn add_index_blob_from_store_range(
        &mut self,
        store: &dyn ObjectStore,
        src_path: &object_store::path::Path,
        src_offset: u64,
        src_length: u64,
        category: &str,
        algorithm: &str,
        column_name: &str,
        blob_type: &str,
        fields: Vec<i32>,
        properties: HashMap<String, String>,
    ) -> Result<&IndexFile> {
        let (dest_offset, length_i64, index_checksum) = self
            .writer
            .add_blob_range_from_store(
                blob_type.to_string(),
                fields,
                self.source_snapshot_id,
                1, // sequence number
                store,
                src_path,
                src_offset,
                src_length,
                properties,
            )
            .await?;

        let entry = IndexFile {
            format_version: 2,
            index_id: Uuid::new_v4().to_string(),
            index_category: category.to_string(),
            algorithm: algorithm.to_string(),
            column_name: column_name.to_string(),
            source_snapshot_id: self.source_snapshot_id,
            source_data_checksum: self.source_data_checksum.clone(),
            source_record_count: self.source_record_count,
            index_checksum: Some(index_checksum),
            file_path: self.container_file_path.clone(),
            blob_type: Some(blob_type.to_string()),
            blob_offset: Some(dest_offset),
            blob_length: Some(length_i64),
            compression: "none".to_string(),
        };

        self.entries.push(entry);
        self.entries.last().ok_or_else(|| {
            anyhow::anyhow!("entry was just pushed")
        })
    }

    pub fn entries(&self) -> &[IndexFile] {
        &self.entries
    }

    /// Finish writing the Puffin container and return the underlying writer
    /// along with the generated `IndexFile` entries.
    pub fn finish(self) -> Result<(W, Vec<IndexFile>)> {
        let writer = self.writer.finish()?;
        Ok((writer, self.entries))
    }
}

impl PuffinIndexWriter<std::io::Cursor<Vec<u8>>> {
    /// Create a new in-memory PuffinIndexWriter.
    pub fn new_in_memory(
        container_file_path: String,
        source_snapshot_id: i64,
        source_data_checksum: String,
        source_record_count: i64,
    ) -> Result<Self> {
        Self::new(
            std::io::Cursor::new(Vec::new()),
            container_file_path,
            source_snapshot_id,
            source_data_checksum,
            source_record_count,
        )
    }

    /// Finish writing the in-memory Puffin container and return the serialized bytes
    /// along with the generated `IndexFile` entries.
    pub fn finish_to_bytes(self) -> Result<(Vec<u8>, Vec<IndexFile>)> {
        let (cursor, entries) = self.finish()?;
        Ok((cursor.into_inner(), entries))
    }
}

/// Convenience reader for index blobs packed inside a Puffin container.
pub struct PuffinIndexReader<R: Read + Seek> {
    reader: PuffinReader<R>,
}

impl<R: Read + Seek> PuffinIndexReader<R> {
    pub fn new(reader: R) -> Result<Self> {
        let p_reader = PuffinReader::new(reader)?;
        Ok(Self { reader: p_reader })
    }

    pub fn blobs(&self) -> &[PuffinBlobMetadata] {
        &self.reader.footer().blobs
    }

    pub fn read_blob_by_index(&mut self, blob_idx: usize) -> Result<Vec<u8>> {
        self.reader.read_blob(blob_idx)
    }

    pub fn read_blob_by_type(&mut self, blob_type: &str) -> Result<Option<Vec<u8>>> {
        let idx = self
            .reader
            .footer()
            .blobs
            .iter()
            .position(|b| b.r#type == blob_type);
        match idx {
            Some(i) => self.reader.read_blob(i).map(Some),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_puffin_writer_and_reader() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let puffin_path = temp_dir.path().join("test.puffin");
        let payload1_path = temp_dir.path().join("payload1.bin");
        let payload2_path = temp_dir.path().join("payload2.bin");

        let data1 = vec![42u8; 100_000];
        let data2 = vec![99u8; 200_000];
        std::fs::write(&payload1_path, &data1)?;
        std::fs::write(&payload2_path, &data2)?;

        let puffin_file = std::fs::File::create(&puffin_path)?;
        let mut writer = PuffinIndexWriter::new(
            std::io::BufWriter::new(puffin_file),
            puffin_path.to_string_lossy().to_string(),
            1,
            "chk123".to_string(),
            300,
        )?;

        let mut props1 = HashMap::new();
        props1.insert("filename".to_string(), "payload1.bin".to_string());
        writer.add_index_blob_from_file(
            "vector",
            "tq8",
            "embedding",
            PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
            payload1_path.to_str().unwrap(),
            vec![],
            props1,
        )?;

        let mut props2 = HashMap::new();
        props2.insert("filename".to_string(), "payload2.bin".to_string());
        writer.add_index_blob_from_file(
            "vector",
            "centroids",
            "embedding",
            PUFFIN_BLOB_HNSW_IVF_CENTROIDS,
            payload2_path.to_str().unwrap(),
            vec![],
            props2,
        )?;

        let (mut file_writer, entries) = writer.finish()?;
        use std::io::Write;
        file_writer.flush()?;
        drop(file_writer);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].blob_length, Some(100_000));
        assert_eq!(entries[1].blob_length, Some(200_000));

        // Read back with PuffinReader
        let read_file = std::fs::File::open(&puffin_path)?;
        let mut reader = PuffinReader::new(read_file)?;
        let blobs = reader.footer().blobs.clone();
        assert_eq!(blobs.len(), 2);

        let read_data1 = reader.read_blob(0)?;
        assert_eq!(read_data1, data1);
        let read_data2 = reader.read_blob(1)?;
        assert_eq!(read_data2, data2);

        Ok(())
    }
}
