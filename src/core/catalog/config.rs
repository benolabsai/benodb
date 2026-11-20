// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use super::CatalogType;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct CatalogConfig {
    pub catalog_type: CatalogType,
    pub config: HashMap<String, String>,
}

impl CatalogConfig {
    pub fn load_from_file(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read catalog config file: {}", path))?;

        // Use toml to deserialize
        let config: CatalogConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse catalog config file: {}", path))?;

        Ok(config)
    }

    pub fn load_default() -> Result<Self> {
        // 1. Check environment variable
        if let Ok(path) = std::env::var("BDB_CONFIG") {
            if fs::metadata(&path).is_ok() {
                return Self::load_from_file(&path);
            }
        }

        // 2. Check current directory
        if fs::metadata("benodb.toml").is_ok() {
            return Self::load_from_file("benodb.toml");
        }

        // 3. Check the home directory
        if let Some(home) = dirs::home_dir() {
            let candidate = home.join(".benodb").join("config.toml");
            if candidate.exists() {
                let candidate_str = candidate.to_str().ok_or_else(|| {
                    anyhow::anyhow!("{} path is not valid UTF-8", candidate.display())
                })?;
                return Self::load_from_file(candidate_str);
            }
        }

        // 4. Fallback/Error
        anyhow::bail!("No configuration file found. Checked ENV 'BDB_CONFIG', ./benodb.toml, and ~/.benodb/config.toml")
    }
}
