// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use super::distance::l2_distance_squared;
use super::ivf::simple_kmeans;
/// Product Quantization (PQ) Implementation
///
/// PQ compresses high-dimensional vectors by splitting them into 'm' sub-vectors
/// and quantizing each sub-vector space into a small codebook (usually 256 centroids).
///
/// This allows:
/// 1. Massive memory reduction (e.g., 1536 floats -> 64 bytes = 96x reduction)
/// 2. Fast search using ADC (Asymmetric Distance Calculation) with lookup tables.
use anyhow::Result;
use tracing;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PqConfig {
    /// Number of sub-vectors (m)
    pub m: usize,
    /// Number of centroids per codebook (usually 256 for 8-bit)
    pub k: usize,
    /// Vector dimensionality
    pub dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PqEncoder {
    pub config: PqConfig,
    /// Codebooks: m x k x (dim/m)
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// SDC Lookup Table: Flat 1D array of size m * 256 * 256 for fast SIMD gathering
    #[serde(skip)]
    pub sdc_lut: Vec<f32>,
}

impl PqEncoder {
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Result<Self> {
        // Validate before dividing: `m` is derived from `dim / compression`, so a
        // compression larger than the dimension yields `m == 0` and a
        // divide-by-zero panic. Degrade to a clear error instead (no-panic policy).
        if config.m == 0 {
            anyhow::bail!(
                "PQ requires m > 0 (got m=0; check dim={} vs compression)",
                config.dim
            );
        }
        if config.dim == 0 || !config.dim.is_multiple_of(config.m) {
            anyhow::bail!(
                "PQ requires dim divisible by m (dim={}, m={})",
                config.dim,
                config.m
            );
        }
        let sub_dim = config.dim / config.m;
        tracing::info!(
            "Training PQ: m={}, k={}, dim={}, sub_dim={}",
            config.m,
            config.k,
            config.dim,
            sub_dim
        );
        use rayon::prelude::*;

        let codebooks: Result<Vec<Vec<Vec<f32>>>> = (0..config.m)
            .into_par_iter()
            .map(|i| {
                let start = i * sub_dim;
                let end = (i + 1) * sub_dim;

                // Extract sub-vectors for this subspace (Still a copy, but parallel)
                let sub_vectors: Vec<Vec<f32>> =
                    vectors.iter().map(|v| v[start..end].to_vec()).collect();

                // Train codebook for this subspace using ultra-fast mini-batch K-Means
                let (centroids, _) = simple_kmeans(&sub_vectors, config.k, 5)?;
                Ok(centroids)
            })
            .collect();

        let mut encoder = Self {
            config,
            codebooks: codebooks?,
            sdc_lut: Vec::new(),
        };
        encoder.init_lut();
        Ok(encoder)
    }

    /// Precompute the Symmetric Distance Computation (SDC) lookup table
    pub fn init_lut(&mut self) {
        if !self.sdc_lut.is_empty() {
            return;
        }
        tracing::debug!(
            "Initializing PQ SDC LUT (m={}, k={})",
            self.config.m,
            self.config.k
        );
        let mut sdc_lut = Vec::with_capacity(self.config.m * self.config.k * self.config.k);
        for i in 0..self.config.m {
            for c1 in 0..self.config.k {
                for c2 in 0..self.config.k {
                    let dist = l2_distance_squared(&self.codebooks[i][c1], &self.codebooks[i][c2]);
                    sdc_lut.push(dist);
                }
            }
        }
        self.sdc_lut = sdc_lut;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn encode_avx2(&self, vector: &[f32]) -> Vec<u8> {
        use std::arch::x86_64::*;
        let mut encoded = Vec::with_capacity(self.config.m);
        let sub_dim = self.config.dim / self.config.m;

        for i in 0..self.config.m {
            let sub_vec = &vector[i * sub_dim..(i + 1) * sub_dim];

            let mut min_dist = f32::MAX;
            let mut best_idx = 0;

            // If sub_dim is a multiple of 8, we can use 256-bit AVX registers
            if sub_dim.is_multiple_of(8) {
                for (j, centroid) in self.codebooks[i].iter().enumerate() {
                    let mut sum = _mm256_setzero_ps();
                    let mut k_idx = 0;
                    while k_idx < sub_dim {
                        let a_val = _mm256_loadu_ps(sub_vec.as_ptr().add(k_idx));
                        let b_val = _mm256_loadu_ps(centroid.as_ptr().add(k_idx));
                        let diff = _mm256_sub_ps(a_val, b_val);
                        let sq = _mm256_mul_ps(diff, diff);
                        sum = _mm256_add_ps(sum, sq);
                        k_idx += 8;
                    }
                    let mut res = [0.0f32; 8];
                    _mm256_storeu_ps(res.as_mut_ptr(), sum);
                    let dist: f32 = res.iter().sum();

                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j as u8;
                    }
                }
            } else {
                for (j, centroid) in self.codebooks[i].iter().enumerate() {
                    let dist = l2_distance_squared(sub_vec, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j as u8;
                    }
                }
            }
            encoded.push(best_idx);
        }
        encoded
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.encode_avx2(vector) };
            }
        }

        let mut encoded = Vec::with_capacity(self.config.m);
        let sub_dim = self.config.dim / self.config.m;

        for i in 0..self.config.m {
            let sub_vec = &vector[i * sub_dim..(i + 1) * sub_dim];

            let mut min_dist = f32::MAX;
            let mut best_idx = 0;

            for (j, centroid) in self.codebooks[i].iter().enumerate() {
                let dist = l2_distance_squared(sub_vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = j as u8;
                }
            }
            encoded.push(best_idx);
        }
        encoded
    }

    /// Compute ADC (Asymmetric Distance Calculation) lookup table
    pub fn compute_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.config.dim / self.config.m;
        let mut lut = Vec::with_capacity(self.config.m);

        for i in 0..self.config.m {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            let sub_query = &query[start..end];

            let mut sub_lut = Vec::with_capacity(self.config.k);
            for centroid in &self.codebooks[i] {
                sub_lut.push(l2_distance_squared(sub_query, centroid));
            }
            lut.push(sub_lut);
        }

        lut
    }

    /// Compute distance using LUT (very fast)
    pub fn distance_from_lut(&self, lut: &[Vec<f32>], encoded: &[u8]) -> f32 {
        let mut dist = 0.0;
        for (i, &code) in encoded.iter().enumerate() {
            dist += lut[i][code as usize];
        }
        dist
    }

    /// Precompute a flat asymmetric lookup table once per query for fast,
    /// allocation-free HNSW graph scoring under `metric`.
    pub fn prepare_query(
        &self,
        query_f32: &[f32],
        metric: crate::core::index::VectorMetric,
    ) -> PreparedPqQuery {
        let m = self.config.m;
        let k = self.config.k;
        if m == 0 || k == 0 || self.codebooks.len() < m {
            return PreparedPqQuery {
                metric,
                m: 0,
                k: 0,
                q_norm_sq: 0.0,
                lut: Vec::new(),
                v_norm_lut: Vec::new(),
            };
        }
        let sub_dim = self.config.dim / m;
        let mut q_buf = vec![0.0f32; self.config.dim];
        let copy_len = query_f32.len().min(self.config.dim);
        q_buf[..copy_len].copy_from_slice(&query_f32[..copy_len]);

        match metric {
            crate::core::index::VectorMetric::Cosine
            | crate::core::index::VectorMetric::InnerProduct => {
                let q_norm_sq = q_buf.iter().map(|x| x * x).sum::<f32>();
                let mut dot_lut = Vec::with_capacity(m * k);
                let mut v_norm_lut = Vec::with_capacity(m * k);
                for i in 0..m {
                    let sub_q = &q_buf[i * sub_dim..(i + 1) * sub_dim];
                    for c in 0..k {
                        if let Some(centroid) = self.codebooks[i].get(c) {
                            let c_len = centroid.len().min(sub_dim);
                            let dot: f32 = sub_q[..c_len]
                                .iter()
                                .zip(&centroid[..c_len])
                                .map(|(a, b)| a * b)
                                .sum();
                            let v_sq: f32 = centroid[..c_len].iter().map(|x| x * x).sum();
                            dot_lut.push(dot);
                            v_norm_lut.push(v_sq);
                        } else {
                            dot_lut.push(0.0);
                            v_norm_lut.push(0.0);
                        }
                    }
                }
                PreparedPqQuery {
                    metric,
                    m,
                    k,
                    q_norm_sq,
                    lut: dot_lut,
                    v_norm_lut,
                }
            }
            crate::core::index::VectorMetric::L1 => {
                let mut l1_lut = Vec::with_capacity(m * k);
                for i in 0..m {
                    let sub_q = &q_buf[i * sub_dim..(i + 1) * sub_dim];
                    for c in 0..k {
                        if let Some(centroid) = self.codebooks[i].get(c) {
                            let c_len = centroid.len().min(sub_dim);
                            let d: f32 = sub_q[..c_len]
                                .iter()
                                .zip(&centroid[..c_len])
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            l1_lut.push(d);
                        } else {
                            l1_lut.push(0.0);
                        }
                    }
                }
                PreparedPqQuery {
                    metric,
                    m,
                    k,
                    q_norm_sq: 0.0,
                    lut: l1_lut,
                    v_norm_lut: Vec::new(),
                }
            }
            _ => {
                let mut sq_lut = Vec::with_capacity(m * k);
                for i in 0..m {
                    let sub_q = &q_buf[i * sub_dim..(i + 1) * sub_dim];
                    for c in 0..k {
                        if let Some(centroid) = self.codebooks[i].get(c) {
                            let c_len = centroid.len().min(sub_dim);
                            sq_lut.push(l2_distance_squared(&sub_q[..c_len], &centroid[..c_len]));
                        } else {
                            sq_lut.push(0.0);
                        }
                    }
                }
                PreparedPqQuery {
                    metric,
                    m,
                    k,
                    q_norm_sq: 0.0,
                    lut: sq_lut,
                    v_norm_lut: Vec::new(),
                }
            }
        }
    }
}

/// Precomputed flat lookup tables for Asymmetric Distance Calculation (ADC)
/// against [`PqEncoder`] codes.
#[derive(Debug, Clone)]
pub struct PreparedPqQuery {
    metric: crate::core::index::VectorMetric,
    m: usize,
    k: usize,
    q_norm_sq: f32,
    lut: Vec<f32>,
    v_norm_lut: Vec<f32>,
}

impl PreparedPqQuery {
    /// Evaluate calibrated asymmetric distance against a stored PQ code.
    /// Uses a 4-way unrolled zero-clamp fast path when `k == 256` so per-segment
    /// HNSW traversal saturates L1 cache bandwidth. Guaranteed finite and non-negative.
    #[inline]
    pub fn distance(&self, encoded: &[u8]) -> f32 {
        if self.m == 0 || self.k == 0 {
            return 0.0;
        }
        let n_sub = self.m.min(encoded.len());
        let fast_256 = self.k == 256 && self.lut.len() >= n_sub.saturating_mul(256);
        let lut = self.lut.as_slice();
        let dist = match self.metric {
            crate::core::index::VectorMetric::Cosine
            | crate::core::index::VectorMetric::InnerProduct => {
                let fast_cos_256 = fast_256 && self.v_norm_lut.len() >= n_sub.saturating_mul(256);
                let v_lut = self.v_norm_lut.as_slice();
                let (dot, v_norm_sq) = if fast_cos_256 {
                    let mut d0 = 0.0f32;
                    let mut d1 = 0.0f32;
                    let mut d2 = 0.0f32;
                    let mut d3 = 0.0f32;
                    let mut v0 = 0.0f32;
                    let mut v1 = 0.0f32;
                    let mut v2 = 0.0f32;
                    let mut v3 = 0.0f32;
                    let mut i = 0usize;
                    while i + 4 <= n_sub {
                        let i0 = (i << 8) + encoded[i] as usize;
                        let i1 = ((i + 1) << 8) + encoded[i + 1] as usize;
                        let i2 = ((i + 2) << 8) + encoded[i + 2] as usize;
                        let i3 = ((i + 3) << 8) + encoded[i + 3] as usize;
                        d0 += lut[i0];
                        d1 += lut[i1];
                        d2 += lut[i2];
                        d3 += lut[i3];
                        v0 += v_lut[i0];
                        v1 += v_lut[i1];
                        v2 += v_lut[i2];
                        v3 += v_lut[i3];
                        i += 4;
                    }
                    let mut dot = (d0 + d1) + (d2 + d3);
                    let mut v_sq = (v0 + v1) + (v2 + v3);
                    while i < n_sub {
                        let idx = (i << 8) + encoded[i] as usize;
                        dot += lut[idx];
                        v_sq += v_lut[idx];
                        i += 1;
                    }
                    (dot, v_sq)
                } else {
                    let mut dot = 0.0f32;
                    let mut v_norm_sq = 0.0f32;
                    for (i, &code) in encoded[..n_sub].iter().enumerate() {
                        let c = (code as usize).min(self.k - 1);
                        let idx = i * self.k + c;
                        if let (Some(&d), Some(&v2)) = (lut.get(idx), v_lut.get(idx)) {
                            dot += d;
                            v_norm_sq += v2;
                        }
                    }
                    (dot, v_norm_sq)
                };
                let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                if denom > 1e-12 {
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                } else {
                    1.0
                }
            }
            crate::core::index::VectorMetric::L1 => {
                if fast_256 {
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let mut i = 0usize;
                    while i + 4 <= n_sub {
                        s0 += lut[(i << 8) + encoded[i] as usize];
                        s1 += lut[((i + 1) << 8) + encoded[i + 1] as usize];
                        s2 += lut[((i + 2) << 8) + encoded[i + 2] as usize];
                        s3 += lut[((i + 3) << 8) + encoded[i + 3] as usize];
                        i += 4;
                    }
                    let mut total = (s0 + s1) + (s2 + s3);
                    while i < n_sub {
                        total += lut[(i << 8) + encoded[i] as usize];
                        i += 1;
                    }
                    total
                } else {
                    let mut total = 0.0f32;
                    for (i, &code) in encoded[..n_sub].iter().enumerate() {
                        let c = (code as usize).min(self.k - 1);
                        if let Some(&val) = lut.get(i * self.k + c) {
                            total += val;
                        }
                    }
                    total
                }
            }
            _ => {
                let dist_sq = if fast_256 {
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    let mut i = 0usize;
                    while i + 4 <= n_sub {
                        s0 += lut[(i << 8) + encoded[i] as usize];
                        s1 += lut[((i + 1) << 8) + encoded[i + 1] as usize];
                        s2 += lut[((i + 2) << 8) + encoded[i + 2] as usize];
                        s3 += lut[((i + 3) << 8) + encoded[i + 3] as usize];
                        i += 4;
                    }
                    let mut total = (s0 + s1) + (s2 + s3);
                    while i < n_sub {
                        total += lut[(i << 8) + encoded[i] as usize];
                        i += 1;
                    }
                    total
                } else {
                    let mut dist_sq = 0.0f32;
                    for (i, &code) in encoded[..n_sub].iter().enumerate() {
                        let c = (code as usize).min(self.k - 1);
                        if let Some(&val) = lut.get(i * self.k + c) {
                            dist_sq += val;
                        }
                    }
                    dist_sq
                };
                dist_sq.max(0.0).sqrt()
            }
        };

        if dist.is_finite() {
            dist.max(0.0)
        } else {
            f32::MAX
        }
    }

    /// High-throughput 4-way unrolled ADC lookup scan over contiguous PQ codes
    /// of stride `code_len`. Uses squared-space early threshold pruning for L2
    /// and a zero-clamp `k == 256` fast path to saturate L1 cache bandwidth.
    pub fn scan_codes<F>(
        &self,
        codes: &[u8],
        code_len: usize,
        initial_threshold: f32,
        mut on_hit: F,
    ) where
        F: FnMut(usize, f32) -> f32,
    {
        if code_len == 0 || self.m == 0 || self.k == 0 {
            return;
        }
        let n_sub = self.m.min(code_len);
        let fast_256 = self.k == 256 && self.lut.len() >= n_sub.saturating_mul(256);
        let mut threshold = initial_threshold;

        match self.metric {
            crate::core::index::VectorMetric::Cosine
            | crate::core::index::VectorMetric::InnerProduct => {
                let fast_cos_256 = fast_256 && self.v_norm_lut.len() >= n_sub.saturating_mul(256);
                let lut = self.lut.as_slice();
                let v_lut = self.v_norm_lut.as_slice();
                for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                    let (dot, v_norm_sq) = if fast_cos_256 {
                        let mut d0 = 0.0f32;
                        let mut d1 = 0.0f32;
                        let mut d2 = 0.0f32;
                        let mut d3 = 0.0f32;
                        let mut v0 = 0.0f32;
                        let mut v1 = 0.0f32;
                        let mut v2 = 0.0f32;
                        let mut v3 = 0.0f32;
                        let mut i = 0usize;
                        while i + 4 <= n_sub {
                            let i0 = (i << 8) + code[i] as usize;
                            let i1 = ((i + 1) << 8) + code[i + 1] as usize;
                            let i2 = ((i + 2) << 8) + code[i + 2] as usize;
                            let i3 = ((i + 3) << 8) + code[i + 3] as usize;
                            d0 += lut[i0];
                            d1 += lut[i1];
                            d2 += lut[i2];
                            d3 += lut[i3];
                            v0 += v_lut[i0];
                            v1 += v_lut[i1];
                            v2 += v_lut[i2];
                            v3 += v_lut[i3];
                            i += 4;
                        }
                        let mut dot = (d0 + d1) + (d2 + d3);
                        let mut v_sq = (v0 + v1) + (v2 + v3);
                        while i < n_sub {
                            let idx = (i << 8) + code[i] as usize;
                            dot += lut[idx];
                            v_sq += v_lut[idx];
                            i += 1;
                        }
                        (dot, v_sq)
                    } else {
                        let mut dot = 0.0f32;
                        let mut v_sq = 0.0f32;
                        for (i, &c) in code[..n_sub].iter().enumerate() {
                            let idx = i * self.k + (c as usize).min(self.k - 1);
                            if let (Some(&d), Some(&v2)) = (lut.get(idx), v_lut.get(idx)) {
                                dot += d;
                                v_sq += v2;
                            }
                        }
                        (dot, v_sq)
                    };
                    let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                    let dist = if denom > 1e-12 {
                        (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                    } else {
                        1.0
                    };
                    if dist.is_finite() && dist < threshold {
                        threshold = on_hit(local_id, dist);
                    }
                }
            }
            crate::core::index::VectorMetric::L1 => {
                let lut = self.lut.as_slice();
                for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                    let dist = if fast_256 {
                        let mut s0 = 0.0f32;
                        let mut s1 = 0.0f32;
                        let mut s2 = 0.0f32;
                        let mut s3 = 0.0f32;
                        let mut i = 0usize;
                        while i + 4 <= n_sub {
                            s0 += lut[(i << 8) + code[i] as usize];
                            s1 += lut[((i + 1) << 8) + code[i + 1] as usize];
                            s2 += lut[((i + 2) << 8) + code[i + 2] as usize];
                            s3 += lut[((i + 3) << 8) + code[i + 3] as usize];
                            i += 4;
                        }
                        let mut total = (s0 + s1) + (s2 + s3);
                        while i < n_sub {
                            total += lut[(i << 8) + code[i] as usize];
                            i += 1;
                        }
                        total
                    } else {
                        let mut total = 0.0f32;
                        for (i, &c) in code[..n_sub].iter().enumerate() {
                            let idx = i * self.k + (c as usize).min(self.k - 1);
                            if let Some(&val) = lut.get(idx) {
                                total += val;
                            }
                        }
                        total
                    };
                    if dist.is_finite() && dist < threshold {
                        threshold = on_hit(local_id, dist.max(0.0));
                    }
                }
            }
            _ => {
                let lut = self.lut.as_slice();
                let to_sq_thresh = |t: f32| -> f32 {
                    if !t.is_finite() {
                        f32::INFINITY
                    } else {
                        t * t * (1.0 + 1e-4) + 1e-6
                    }
                };
                let mut sq_thresh = to_sq_thresh(threshold);
                for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                    let dist_sq = if fast_256 {
                        let mut s0 = 0.0f32;
                        let mut s1 = 0.0f32;
                        let mut s2 = 0.0f32;
                        let mut s3 = 0.0f32;
                        let mut i = 0usize;
                        while i + 4 <= n_sub {
                            s0 += lut[(i << 8) + code[i] as usize];
                            s1 += lut[((i + 1) << 8) + code[i + 1] as usize];
                            s2 += lut[((i + 2) << 8) + code[i + 2] as usize];
                            s3 += lut[((i + 3) << 8) + code[i + 3] as usize];
                            i += 4;
                        }
                        let mut total = (s0 + s1) + (s2 + s3);
                        while i < n_sub {
                            total += lut[(i << 8) + code[i] as usize];
                            i += 1;
                        }
                        total
                    } else {
                        let mut total = 0.0f32;
                        for (i, &c) in code[..n_sub].iter().enumerate() {
                            let idx = i * self.k + (c as usize).min(self.k - 1);
                            if let Some(&val) = lut.get(idx) {
                                total += val;
                            }
                        }
                        total
                    };
                    if dist_sq <= sq_thresh {
                        let dist = dist_sq.max(0.0).sqrt();
                        if dist.is_finite() && dist < threshold {
                            threshold = on_hit(local_id, dist);
                            sq_thresh = to_sq_thresh(threshold);
                        }
                    }
                }
            }
        }
    }
}

impl crate::core::index::Quantizer for PqEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.encode(vector)
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.config.dim);
        for i in 0..self.config.m {
            let code = bytes[i] as usize;
            vector.extend_from_slice(&self.codebooks[i][code]);
        }
        vector
    }

    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32 {
        if self.config.m == 0 || self.codebooks.len() < self.config.m {
            return 0.0;
        }
        let sub_dim = self.config.dim / self.config.m;
        let mut dist_sq = 0.0f32;
        for (i, &code_byte) in encoded.iter().take(self.config.m).enumerate() {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            if end > query.len() {
                break;
            }
            let sub_query = &query[start..end];
            if let Some(centroid) = self.codebooks[i].get(code_byte as usize) {
                dist_sq += l2_distance_squared(sub_query, centroid);
            }
        }
        let dist = dist_sq.max(0.0).sqrt();
        if dist.is_finite() {
            dist
        } else {
            f32::MAX
        }
    }

    fn name(&self) -> String {
        format!("pq_{}", self.config.m)
    }

    fn bits(&self) -> usize {
        8
    }

    fn dim(&self) -> usize {
        self.config.dim
    }
}

use crate::core::index::hnsw_rs::dist::Distance;
use std::sync::Arc;

/// Fast Symmetric Distance Computation (SDC) for PQ using precomputed LUT
#[derive(Clone, Debug)]
pub struct DistPqSdc {
    pub pq: Arc<PqEncoder>,
}

impl Distance<u8> for DistPqSdc {
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { pq_sdc_avx2(a, b, &self.pq.sdc_lut, self.pq.config.m) };
            }
        }

        let mut dist = 0.0;
        let lut = &self.pq.sdc_lut;
        for i in 0..self.pq.config.m {
            let idx = i * 65536 + (a[i] as usize) * 256 + (b[i] as usize);
            dist += lut[idx];
        }
        dist
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pq_sdc_avx2(a: &[u8], b: &[u8], lut: &[f32], m: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();

    // Base offsets for 8 dimensions: [0*65536, 1*65536, 2*65536, ..., 7*65536]
    let base_offsets = _mm256_set_epi32(
        7 * 65536,
        6 * 65536,
        5 * 65536,
        4 * 65536,
        3 * 65536,
        2 * 65536,
        65536,
        0,
    );

    let mut i = 0;
    while i + 8 <= m {
        let a_ptr = a.as_ptr().add(i) as *const i64;
        let b_ptr = b.as_ptr().add(i) as *const i64;

        let a_8 = _mm_loadl_epi64(a_ptr as *const __m128i);
        let b_8 = _mm_loadl_epi64(b_ptr as *const __m128i);

        let a_32 = _mm256_cvtepu8_epi32(a_8);
        let b_32 = _mm256_cvtepu8_epi32(b_8);

        let a_shifted = _mm256_slli_epi32(a_32, 8);
        let combined = _mm256_or_si256(a_shifted, b_32);

        let final_offsets = _mm256_add_epi32(combined, base_offsets);

        // Advance the LUT pointer by 8 dimensions * 65536 entries
        let current_lut = lut.as_ptr().add(i * 65536);
        let vals = _mm256_i32gather_ps::<4>(current_lut, final_offsets);
        sum = _mm256_add_ps(sum, vals);

        i += 8;
    }

    let mut res = [0.0f32; 8];
    _mm256_storeu_ps(res.as_mut_ptr(), sum);

    // Add remaining dimensions if m is not a multiple of 8
    let mut total = res.iter().sum::<f32>();
    while i < m {
        let offset = i * 65536 + (a[i] as usize) * 256 + (b[i] as usize);
        total += lut[offset];
        i += 1;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::index::VectorMetric;

    #[test]
    fn test_pq_prepared_query_l2_and_cosine() {
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..32)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect())
            .collect();
        let pq = PqEncoder::train(&vectors, PqConfig { m: 2, k: 16, dim }).unwrap();

        let query = vec![25.0f32; dim];
        let p_l2 = pq.prepare_query(&query, VectorMetric::L2);
        let code0 = pq.encode(&vectors[0]);
        let code_last = pq.encode(&vectors[31]);
        let d0 = p_l2.distance(&code0);
        let d_last = p_l2.distance(&code_last);
        assert!(
            d_last < d0,
            "expected closer vector ({d_last}) < far ({d0})"
        );
        assert!(
            d_last < 3.0 && d0 > 50.0,
            "calibrated L2 bounds failed: d_last={d_last}, d0={d0}"
        );

        let p_cos = pq.prepare_query(&vectors[5], VectorMetric::Cosine);
        let self_cos = p_cos.distance(&pq.encode(&vectors[5]));
        assert!((0.0..0.25).contains(&self_cos), "got {self_cos}");
    }

    #[test]
    fn test_pq_scan_codes_matches_distance() {
        let dim = 32;
        let vectors: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                (0..dim)
                    .map(|d| ((i * 17 + d * 31) % 97) as f32 * 0.1 - 4.0)
                    .collect()
            })
            .collect();
        for (m, k) in [(8usize, 256usize), (4usize, 16usize)] {
            let pq = PqEncoder::train(&vectors, PqConfig { m, k, dim }).unwrap();
            let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| pq.encode(v)).collect();
            let code_len = encoded[0].len();
            let mut flat = Vec::with_capacity(encoded.len() * code_len);
            for c in &encoded {
                flat.extend_from_slice(c);
            }
            let q = vectors[19].clone();

            for metric in [VectorMetric::L2, VectorMetric::Cosine, VectorMetric::L1] {
                let prep = pq.prepare_query(&q, metric);
                let mut scanned = Vec::new();
                prep.scan_codes(&flat, code_len, f32::INFINITY, |id, dist| {
                    scanned.push((id, dist));
                    f32::INFINITY
                });
                assert_eq!(scanned.len(), vectors.len());
                for (id, dist) in scanned {
                    let ref_dist = prep.distance(&encoded[id]);
                    assert!(
                        (dist - ref_dist).abs() < 1e-3,
                        "m={m} k={k} metric={metric:?} id={id}: scan {dist} != ref {ref_dist}"
                    );
                }
            }
        }
    }
}
