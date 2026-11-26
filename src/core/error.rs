// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Engine error types, re-exported from the `benodb-iceberg` crate during the
//! crate extraction. The Goal B type surgery re-homes error types so the
//! format crate's public API speaks only Iceberg spec concepts.

pub use benodb_iceberg::error::*;
