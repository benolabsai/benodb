// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Errors owned by the Iceberg format layer.
//!
//! Spec-pure: the variants describe format-level failures only (corrupt
//! payloads, framing violations). Engine concerns — catalogs, manifests of
//! engine index inventory, SQL — belong to the engine's own error type and
//! never appear here.

use std::fmt;

#[derive(Debug)]
pub enum IcebergError {
    /// A `deletion-vector-v1` Puffin blob failed framing or length validation,
    /// so its bitmap cannot be decoded. Surfaced as a typed error rather than
    /// a silently short delete list, which would resurrect deleted rows.
    DeletionVectorDeserialize { reason: String },
}

impl fmt::Display for IcebergError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeletionVectorDeserialize { reason } => {
                write!(f, "deletion vector deserialize error: {reason}")
            }
        }
    }
}

impl std::error::Error for IcebergError {}
