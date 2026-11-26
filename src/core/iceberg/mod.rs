// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Iceberg integration module for BenoDB.
//!
//! The implementation lives in the standalone `benodb-iceberg` crate; this
//! module re-exports it so every `crate::core::iceberg::` path in the engine
//! keeps resolving while the crate extraction proceeds.

pub use benodb_iceberg::*;

// Engine vocabulary now: the enum's Data variant carries the engine's
// ManifestEntry. Re-exported so the historical path keeps resolving.
pub use crate::core::manifest::types::convert_iceberg_to_object;
pub use crate::core::manifest::types::IcebergManifestObject;

// Re-export `iceberg_delete` for backward compatibility with existing callers
// that reference `crate::core::iceberg::iceberg_delete::IcebergDeleteWriter`.
// The actual implementation lives in `delete` module.
pub mod iceberg_delete {
    pub use super::delete::IcebergDeleteWriter;
}
