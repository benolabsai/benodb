// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::{Quantizer, VectorMetric};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Standard Uniform Scalar Quantization (SQ8)
///
/// Implements standard coordinate-wise uniform scalar quantization (matching FAISS SQ8
/// and Qdrant int8) without any orthogonal rotations. Used for baseline comparisons
/// and ablation studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqEncoder {
    pub dim: usize,
    pub bits: usize,
    /// Per-dimension minimum offsets: length == dim
    pub offsets: Vec<f32>,
    /// Per-dimension scaling factors: length == dim
    pub scales: Vec<f32>,
    /// Per-dimension inverse scaling factors: length == dim
    pub inv_scales: Vec<f32>,
    #[serde(default)]
    pub metric: VectorMetric,
}

impl SqEncoder {
    pub fn new(dim: usize, bits: usize, offsets: Vec<f32>, scales: Vec<f32>) -> Self {
        let inv_scales = scales
            .iter()
            .map(|&s| if s.abs() < 1e-10 { 1.0 } else { 1.0 / s })
            .collect();
        Self {
            dim,
            bits,
            offsets,
            scales,
            inv_scales,
            metric: VectorMetric::L2,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Train SQ8 by computing per-dimension min and max across sample vectors.
    pub fn train(vectors: &[Vec<f32>], bits: usize, metric: VectorMetric) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train SqEncoder with empty vector set");
        }
        let dim = vectors[0].len();
        let mut min_vals = vec![f32::MAX; dim];
        let mut max_vals = vec![f32::MIN; dim];

        let sample_size = vectors.len().min(10000);
        for i in 0..sample_size {
            // Cosine is norm-invariant. Normalize before measuring the
            // per-dimension range so the quantized codes live on the unit
            // sphere; the HNSW graph ranks SQ codes with byte-space L2, which
            // is then cosine-equivalent. Without this, L2 on the raw vectors is
            // sensitive to the ~5% norm variation and scrambles the top-k on
            // angular datasets (e.g. nytimes-256-angular).
            let mut v = vectors[i].clone();
            if metric == VectorMetric::Cosine {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    let inv = 1.0 / norm;
                    for x in &mut v {
                        *x *= inv;
                    }
                }
            }
            for j in 0..dim {
                let val = v[j];
                if val < min_vals[j] {
                    min_vals[j] = val;
                }
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        let max_quant = ((1 << bits) - 1) as f32;
        let mut scales = Vec::with_capacity(dim);
        let mut inv_scales = Vec::with_capacity(dim);

        for j in 0..dim {
            let span = max_vals[j] - min_vals[j];
            let scale = if span.abs() < 1e-10 {
                1.0
            } else {
                max_quant / span
            };
            scales.push(scale);
            inv_scales.push(if scale.abs() < 1e-10 {
                1.0
            } else {
                1.0 / scale
            });
        }

        Ok(Self {
            dim,
            bits,
            offsets: min_vals,
            scales,
            inv_scales,
            metric,
        })
    }

    /// Precompute the un-clamped query state once for asymmetric distance
    /// evaluation against SQ8 codes.
    pub fn prepare_query(&self, query_f32: &[f32], metric: VectorMetric) -> PreparedSqQuery<'_> {
        let mut query = vec![0.0f32; self.dim];
        let copy_len = query_f32.len().min(self.dim);
        query[..copy_len].copy_from_slice(&query_f32[..copy_len]);

        if metric == VectorMetric::Cosine || self.metric == VectorMetric::Cosine {
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut query {
                    *x *= inv;
                }
            }
        }

        let q_norm_sq = query.iter().map(|x| x * x).sum::<f32>();
        let q_shifted: Vec<f32> = query
            .iter()
            .zip(self.offsets.iter())
            .map(|(&q, &off)| q - off)
            .collect();
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        PreparedSqQuery {
            encoder: self,
            query,
            q_shifted,
            q_norm_sq,
            metric,
            #[cfg(target_arch = "x86_64")]
            use_avx2,
        }
    }
}

/// Pre-processed, un-clamped query state for Asymmetric Distance Calculation
/// (ADC) against [`SqEncoder`] codes.
#[derive(Debug, Clone)]
pub struct PreparedSqQuery<'a> {
    encoder: &'a SqEncoder,
    query: Vec<f32>,
    /// Precomputed `query[j] - offsets[j]` so L2/L1 inner loops compute
    /// `q_shifted[j] - code[j] * inv_scales[j]` in one FMA without loading `offsets`.
    q_shifted: Vec<f32>,
    q_norm_sq: f32,
    metric: VectorMetric,
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    lanes.iter().sum()
}

/// AVX2+FMA squared L2 distance over per-dimension SQ8 codes using precomputed
/// `q_shifted = query - offsets`: `diff = q_shifted - code * inv_scales`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sq8_l2_sq_avx2(q_shifted: &[f32], inv_scales: &[f32], code: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n = q_shifted.len().min(inv_scales.len()).min(code.len());
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let q_ptr = q_shifted.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 16 <= n {
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let s0 = _mm256_loadu_ps(s_ptr.add(i));
        let q0 = _mm256_loadu_ps(q_ptr.add(i));
        let d0 = _mm256_fnmadd_ps(c0, s0, q0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 8) as *const __m128i
        )));
        let s1 = _mm256_loadu_ps(s_ptr.add(i + 8));
        let q1 = _mm256_loadu_ps(q_ptr.add(i + 8));
        let d1 = _mm256_fnmadd_ps(c1, s1, q1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);

        i += 16;
    }

    let mut total = hsum256_ps(_mm256_add_ps(sum0, sum1));
    while i < n {
        let d = *q_ptr.add(i) - (*c_ptr.add(i) as f32) * *s_ptr.add(i);
        total += d * d;
        i += 1;
    }
    total
}

/// AVX2+FMA dot product and decoded squared norm over per-dimension SQ8 codes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sq8_dot_norm_avx2(
    query: &[f32],
    inv_scales: &[f32],
    offsets: &[f32],
    code: &[u8],
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = query
        .len()
        .min(inv_scales.len())
        .min(offsets.len())
        .min(code.len());
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut norm0 = _mm256_setzero_ps();
    let mut norm1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let q_ptr = query.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let o_ptr = offsets.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 16 <= n {
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let dec0 = _mm256_fmadd_ps(
            c0,
            _mm256_loadu_ps(s_ptr.add(i)),
            _mm256_loadu_ps(o_ptr.add(i)),
        );
        let q0 = _mm256_loadu_ps(q_ptr.add(i));
        dot0 = _mm256_fmadd_ps(q0, dec0, dot0);
        norm0 = _mm256_fmadd_ps(dec0, dec0, norm0);

        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 8) as *const __m128i
        )));
        let dec1 = _mm256_fmadd_ps(
            c1,
            _mm256_loadu_ps(s_ptr.add(i + 8)),
            _mm256_loadu_ps(o_ptr.add(i + 8)),
        );
        let q1 = _mm256_loadu_ps(q_ptr.add(i + 8));
        dot1 = _mm256_fmadd_ps(q1, dec1, dot1);
        norm1 = _mm256_fmadd_ps(dec1, dec1, norm1);

        i += 16;
    }

    let mut dot = hsum256_ps(_mm256_add_ps(dot0, dot1));
    let mut v_norm_sq = hsum256_ps(_mm256_add_ps(norm0, norm1));
    while i < n {
        let dec = (*c_ptr.add(i) as f32) * *s_ptr.add(i) + *o_ptr.add(i);
        dot += *q_ptr.add(i) * dec;
        v_norm_sq += dec * dec;
        i += 1;
    }
    (dot, v_norm_sq)
}

/// AVX2+FMA L1 distance over per-dimension SQ8 codes using precomputed `q_shifted`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sq8_l1_avx2(q_shifted: &[f32], inv_scales: &[f32], code: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n = q_shifted.len().min(inv_scales.len()).min(code.len());
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let q_ptr = q_shifted.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 16 <= n {
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let d0 = _mm256_fnmadd_ps(
            c0,
            _mm256_loadu_ps(s_ptr.add(i)),
            _mm256_loadu_ps(q_ptr.add(i)),
        );
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, d0));

        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 8) as *const __m128i
        )));
        let d1 = _mm256_fnmadd_ps(
            c1,
            _mm256_loadu_ps(s_ptr.add(i + 8)),
            _mm256_loadu_ps(q_ptr.add(i + 8)),
        );
        sum1 = _mm256_add_ps(sum1, _mm256_andnot_ps(sign_mask, d1));

        i += 16;
    }

    let mut total = hsum256_ps(_mm256_add_ps(sum0, sum1));
    while i < n {
        total += (*q_ptr.add(i) - (*c_ptr.add(i) as f32) * *s_ptr.add(i)).abs();
        i += 1;
    }
    total
}

/// ARM64 NEON squared L2 distance over per-dimension SQ8 codes using precomputed
/// `q_shifted = query - offsets`: `diff = q_shifted - code * inv_scales`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sq8_l2_sq_neon(q_shifted: &[f32], inv_scales: &[f32], code: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let n = q_shifted.len().min(inv_scales.len()).min(code.len());
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let q_ptr = q_shifted.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 8 <= n {
        let c16 = vmovl_u8(vld1_u8(c_ptr.add(i)));
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        let d0 = vfmsq_f32(vld1q_f32(q_ptr.add(i)), c0, vld1q_f32(s_ptr.add(i)));
        sum0 = vfmaq_f32(sum0, d0, d0);

        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));
        let d1 = vfmsq_f32(vld1q_f32(q_ptr.add(i + 4)), c1, vld1q_f32(s_ptr.add(i + 4)));
        sum1 = vfmaq_f32(sum1, d1, d1);

        i += 8;
    }

    let mut total = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < n {
        let d = *q_ptr.add(i) - (*c_ptr.add(i) as f32) * *s_ptr.add(i);
        total += d * d;
        i += 1;
    }
    total
}

/// ARM64 NEON dot product and decoded squared norm over per-dimension SQ8 codes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sq8_dot_norm_neon(
    query: &[f32],
    inv_scales: &[f32],
    offsets: &[f32],
    code: &[u8],
) -> (f32, f32) {
    use std::arch::aarch64::*;
    let n = query
        .len()
        .min(inv_scales.len())
        .min(offsets.len())
        .min(code.len());
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut norm0 = vdupq_n_f32(0.0);
    let mut norm1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let q_ptr = query.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let o_ptr = offsets.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 8 <= n {
        let c16 = vmovl_u8(vld1_u8(c_ptr.add(i)));
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        let dec0 = vfmaq_f32(vld1q_f32(o_ptr.add(i)), c0, vld1q_f32(s_ptr.add(i)));
        dot0 = vfmaq_f32(dot0, vld1q_f32(q_ptr.add(i)), dec0);
        norm0 = vfmaq_f32(norm0, dec0, dec0);

        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));
        let dec1 = vfmaq_f32(vld1q_f32(o_ptr.add(i + 4)), c1, vld1q_f32(s_ptr.add(i + 4)));
        dot1 = vfmaq_f32(dot1, vld1q_f32(q_ptr.add(i + 4)), dec1);
        norm1 = vfmaq_f32(norm1, dec1, dec1);

        i += 8;
    }

    let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut v_norm_sq = vaddvq_f32(vaddq_f32(norm0, norm1));
    while i < n {
        let dec = (*c_ptr.add(i) as f32) * *s_ptr.add(i) + *o_ptr.add(i);
        dot += *q_ptr.add(i) * dec;
        v_norm_sq += dec * dec;
        i += 1;
    }
    (dot, v_norm_sq)
}

/// ARM64 NEON L1 distance over per-dimension SQ8 codes using precomputed `q_shifted`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sq8_l1_neon(q_shifted: &[f32], inv_scales: &[f32], code: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let n = q_shifted.len().min(inv_scales.len()).min(code.len());
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let q_ptr = q_shifted.as_ptr();
    let s_ptr = inv_scales.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 8 <= n {
        let c16 = vmovl_u8(vld1_u8(c_ptr.add(i)));
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        let d0 = vfmsq_f32(vld1q_f32(q_ptr.add(i)), c0, vld1q_f32(s_ptr.add(i)));
        sum0 = vaddq_f32(sum0, vabsq_f32(d0));

        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));
        let d1 = vfmsq_f32(vld1q_f32(q_ptr.add(i + 4)), c1, vld1q_f32(s_ptr.add(i + 4)));
        sum1 = vaddq_f32(sum1, vabsq_f32(d1));

        i += 8;
    }

    let mut total = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < n {
        total += (*q_ptr.add(i) - (*c_ptr.add(i) as f32) * *s_ptr.add(i)).abs();
        i += 1;
    }
    total
}

impl<'a> PreparedSqQuery<'a> {
    /// Evaluate calibrated asymmetric distance against a stored SQ8 code.
    /// Dispatches to AVX2+FMA on x86_64 and NEON on aarch64 so per-segment HNSW
    /// traversal gets full SIMD parity. Guaranteed finite and non-negative.
    #[inline]
    pub fn distance(&self, encoded: &[u8]) -> f32 {
        let n = self
            .encoder
            .dim
            .min(self.query.len())
            .min(encoded.len())
            .min(self.encoder.offsets.len())
            .min(self.encoder.inv_scales.len())
            .min(self.q_shifted.len());

        let dist = match self.metric {
            VectorMetric::Cosine | VectorMetric::InnerProduct => {
                let (dot, v_norm_sq) = {
                    #[cfg(target_arch = "aarch64")]
                    {
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        unsafe {
                            sq8_dot_norm_neon(
                                &self.query[..n],
                                &self.encoder.inv_scales[..n],
                                &self.encoder.offsets[..n],
                                &encoded[..n],
                            )
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        #[cfg(target_arch = "x86_64")]
                        if self.use_avx2 {
                            // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                            unsafe {
                                sq8_dot_norm_avx2(
                                    &self.query[..n],
                                    &self.encoder.inv_scales[..n],
                                    &self.encoder.offsets[..n],
                                    &encoded[..n],
                                )
                            }
                        } else {
                            let mut dot = 0.0f32;
                            let mut v_norm_sq = 0.0f32;
                            for j in 0..n {
                                let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                    + self.encoder.offsets[j];
                                dot += self.query[j] * decoded;
                                v_norm_sq += decoded * decoded;
                            }
                            (dot, v_norm_sq)
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let mut dot = 0.0f32;
                            let mut v_norm_sq = 0.0f32;
                            for j in 0..n {
                                let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                    + self.encoder.offsets[j];
                                dot += self.query[j] * decoded;
                                v_norm_sq += decoded * decoded;
                            }
                            (dot, v_norm_sq)
                        }
                    }
                };
                let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                if denom > 1e-12 {
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                } else {
                    1.0
                }
            }
            VectorMetric::L1 => {
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is mandatory baseline on aarch64.
                    unsafe {
                        sq8_l1_neon(
                            &self.q_shifted[..n],
                            &self.encoder.inv_scales[..n],
                            &encoded[..n],
                        )
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    #[cfg(target_arch = "x86_64")]
                    if self.use_avx2 {
                        // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                        unsafe {
                            sq8_l1_avx2(
                                &self.q_shifted[..n],
                                &self.encoder.inv_scales[..n],
                                &encoded[..n],
                            )
                        }
                    } else {
                        let mut dist = 0.0f32;
                        for j in 0..n {
                            let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                + self.encoder.offsets[j];
                            dist += (self.query[j] - decoded).abs();
                        }
                        dist
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let mut dist = 0.0f32;
                        for j in 0..n {
                            let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                + self.encoder.offsets[j];
                            dist += (self.query[j] - decoded).abs();
                        }
                        dist
                    }
                }
            }
            _ => {
                let dist_sq = {
                    #[cfg(target_arch = "aarch64")]
                    {
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        unsafe {
                            sq8_l2_sq_neon(
                                &self.q_shifted[..n],
                                &self.encoder.inv_scales[..n],
                                &encoded[..n],
                            )
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        #[cfg(target_arch = "x86_64")]
                        if self.use_avx2 {
                            // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                            unsafe {
                                sq8_l2_sq_avx2(
                                    &self.q_shifted[..n],
                                    &self.encoder.inv_scales[..n],
                                    &encoded[..n],
                                )
                            }
                        } else {
                            let mut dist_sq = 0.0f32;
                            for j in 0..n {
                                let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                    + self.encoder.offsets[j];
                                let diff = self.query[j] - decoded;
                                dist_sq += diff * diff;
                            }
                            dist_sq
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let mut dist_sq = 0.0f32;
                            for j in 0..n {
                                let decoded = (encoded[j] as f32 * self.encoder.inv_scales[j])
                                    + self.encoder.offsets[j];
                                let diff = self.query[j] - decoded;
                                dist_sq += diff * diff;
                            }
                            dist_sq
                        }
                    }
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

    /// SIMD-accelerated batch scan over contiguous SQ8 codes of stride `code_len`.
    /// Hoists AVX2+FMA detection once per cluster and prunes L2 candidates in
    /// squared space before `sqrt`.
    pub fn scan_codes<F>(
        &self,
        codes: &[u8],
        code_len: usize,
        initial_threshold: f32,
        mut on_hit: F,
    ) where
        F: FnMut(usize, f32) -> f32,
    {
        if code_len == 0 {
            return;
        }
        let n = self
            .encoder
            .dim
            .min(self.query.len())
            .min(code_len)
            .min(self.encoder.offsets.len())
            .min(self.encoder.inv_scales.len())
            .min(self.q_shifted.len());
        if n == 0 {
            return;
        }

        let mut threshold = initial_threshold;

        #[cfg(target_arch = "aarch64")]
        {
            let inv_scales = &self.encoder.inv_scales[..n];
            match self.metric {
                VectorMetric::Cosine | VectorMetric::InnerProduct => {
                    let query = &self.query[..n];
                    let offsets = &self.encoder.offsets[..n];
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        let (dot, v_norm_sq) =
                            unsafe { sq8_dot_norm_neon(query, inv_scales, offsets, &code[..n]) };
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
                VectorMetric::L1 => {
                    let q_shifted = &self.q_shifted[..n];
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        let dist = unsafe { sq8_l1_neon(q_shifted, inv_scales, &code[..n]) };
                        if dist.is_finite() && dist < threshold {
                            threshold = on_hit(local_id, dist.max(0.0));
                        }
                    }
                }
                _ => {
                    let q_shifted = &self.q_shifted[..n];
                    let to_sq_thresh = |t: f32| -> f32 {
                        if !t.is_finite() {
                            f32::INFINITY
                        } else {
                            t * t * (1.0 + 1e-4) + 1e-6
                        }
                    };
                    let mut sq_thresh = to_sq_thresh(threshold);
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        let dist_sq = unsafe { sq8_l2_sq_neon(q_shifted, inv_scales, &code[..n]) };
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

        #[cfg(not(target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            #[cfg(not(target_arch = "x86_64"))]
            let use_avx2 = false;

            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    let inv_scales = &self.encoder.inv_scales[..n];
                    match self.metric {
                        VectorMetric::Cosine | VectorMetric::InnerProduct => {
                            let query = &self.query[..n];
                            let offsets = &self.encoder.offsets[..n];
                            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                                // SAFETY: `avx2` and `fma` were verified above.
                                let (dot, v_norm_sq) = unsafe {
                                    sq8_dot_norm_avx2(query, inv_scales, offsets, &code[..n])
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
                            return;
                        }
                        VectorMetric::L1 => {
                            let q_shifted = &self.q_shifted[..n];
                            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                                // SAFETY: `avx2` and `fma` were verified above.
                                let dist =
                                    unsafe { sq8_l1_avx2(q_shifted, inv_scales, &code[..n]) };
                                if dist.is_finite() && dist < threshold {
                                    threshold = on_hit(local_id, dist.max(0.0));
                                }
                            }
                            return;
                        }
                        _ => {
                            let q_shifted = &self.q_shifted[..n];
                            let to_sq_thresh = |t: f32| -> f32 {
                                if !t.is_finite() {
                                    f32::INFINITY
                                } else {
                                    t * t * (1.0 + 1e-4) + 1e-6
                                }
                            };
                            let mut sq_thresh = to_sq_thresh(threshold);
                            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                                // SAFETY: `avx2` and `fma` were verified above.
                                let dist_sq =
                                    unsafe { sq8_l2_sq_avx2(q_shifted, inv_scales, &code[..n]) };
                                if dist_sq <= sq_thresh {
                                    let dist = dist_sq.max(0.0).sqrt();
                                    if dist.is_finite() && dist < threshold {
                                        threshold = on_hit(local_id, dist);
                                        sq_thresh = to_sq_thresh(threshold);
                                    }
                                }
                            }
                            return;
                        }
                    }
                }
            }

            // Portable scalar fallback.
            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                let dist = self.distance(code);
                if dist < threshold {
                    threshold = on_hit(local_id, dist);
                }
            }
        }
    }
}

impl Quantizer for SqEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let max_quant = ((1 << self.bits) - 1) as f32;
        let mut encoded = Vec::with_capacity(self.dim);
        // Cosine is norm-invariant: normalize before quantizing so the stored
        // codes and the query codes both live on the unit sphere. The graph
        // ranks with byte-space L2, which is then cosine-equivalent.
        let normalized: Vec<f32>;
        let v: &[f32] = if self.metric == VectorMetric::Cosine {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                normalized = vector.iter().map(|x| x * inv).collect();
                &normalized
            } else {
                vector
            }
        } else {
            vector
        };
        for j in 0..self.dim.min(v.len()) {
            let q = ((v[j] - self.offsets[j]) * self.scales[j]).round();
            encoded.push(q.clamp(0.0, max_quant) as u8);
        }
        encoded
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.dim);
        for j in 0..self.dim.min(bytes.len()) {
            decoded.push((bytes[j] as f32 * self.inv_scales[j]) + self.offsets[j]);
        }
        decoded
    }

    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32 {
        self.prepare_query(query, self.metric).distance(encoded)
    }

    fn name(&self) -> String {
        format!("sq_{}bit", self.bits)
    }

    fn bits(&self) -> usize {
        self.bits
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Distance over SQ8 codes that decodes each dimension with its own scale and
/// then applies the index metric.
///
/// The HNSW graph stores `u8` codes, so a plain byte-space L2 (`DistL2u8`) is
/// only correct for isotropic quantization. SQ8 uses per-dimension scales, and
/// for cosine the byte-space L2 is dominated by the vector norm: an all-zero
/// vector sits at distance ~1.0 from every query, closer than the ~1.4 between
/// real unit vectors, so it ranks first instead of last. Decoding restores the
/// metric (cosine for angular datasets, L2 otherwise).
#[derive(Clone)]
pub struct DistSq8 {
    pub encoder: Arc<SqEncoder>,
}

impl Distance<u8> for DistSq8 {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        let n = va.len().min(vb.len()).min(self.encoder.dim);
        match self.encoder.metric {
            VectorMetric::Cosine => {
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for j in 0..n {
                    let a = va[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let b = vb[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let denom = (na * nb).sqrt();
                if denom > 1e-12 {
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                } else {
                    1.0
                }
            }
            _ => {
                let mut dist_sq = 0.0f32;
                for j in 0..n {
                    let a = va[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let b = vb[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let diff = a - b;
                    dist_sq += diff * diff;
                }
                dist_sq
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_encode_decode_roundtrip() {
        let dim = 16;
        let vectors = vec![
            (0..dim).map(|i| i as f32 * 1.5).collect::<Vec<f32>>(),
            (0..dim).map(|i| -(i as f32) * 2.0).collect::<Vec<f32>>(),
        ];
        let encoder = SqEncoder::train(&vectors, 8, VectorMetric::L2).unwrap();
        assert_eq!(encoder.dim, dim);
        assert_eq!(encoder.bits, 8);

        let test_vec = (0..dim).map(|i| (i as f32) * 0.5).collect::<Vec<f32>>();
        let encoded = encoder.encode(&test_vec);
        assert_eq!(encoded.len(), dim);
        let decoded = encoder.decode(&encoded);
        for j in 0..dim {
            assert!((test_vec[j] - decoded[j]).abs() < 0.5);
        }
    }

    #[test]
    fn test_sq8_adc_calibrated_l2_and_out_of_range_query() {
        let dim = 8;
        let seg1: Vec<Vec<f32>> = vec![vec![194.0; dim], vec![195.0; dim], vec![196.0; dim]];
        let seg2: Vec<Vec<f32>> = vec![vec![197.0; dim]];
        let query = vec![197.0f32; dim];

        let enc1 = SqEncoder::train(&seg1, 8, VectorMetric::L2).unwrap();
        let enc2 = SqEncoder::train(&seg2, 8, VectorMetric::L2).unwrap();

        let d_196 = enc1
            .prepare_query(&query, VectorMetric::L2)
            .distance(&enc1.encode(&seg1[2]));
        let d_197 = enc2
            .prepare_query(&query, VectorMetric::L2)
            .distance(&enc2.encode(&seg2[0]));

        let expected_196 = (dim as f32).sqrt();
        assert!(
            (d_196 - expected_196).abs() < 0.2,
            "SQ8 out-of-range query must not clamp to 0; got {d_196}, expected ~{expected_196}"
        );
        assert!(
            d_197 < 0.05,
            "SQ8 exact single-row match must score ~0, got {d_197}"
        );
    }

    #[test]
    fn test_sq8_scan_codes_matches_distance() {
        for dim in [12usize, 16, 128] {
            let mut rng_state: u64 = 777 + dim as u64;
            let mut rand_f32 = || {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u32 << 31) as f32) * 10.0 - 5.0
            };
            let vectors: Vec<Vec<f32>> = (0..48)
                .map(|_| (0..dim).map(|_| rand_f32()).collect())
                .collect();
            let q = vectors[5].clone();

            for metric in [VectorMetric::L2, VectorMetric::Cosine, VectorMetric::L1] {
                let enc = SqEncoder::train(&vectors, 8, metric).unwrap();
                let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| enc.encode(v)).collect();
                let code_len = encoded[0].len();
                let mut flat = Vec::with_capacity(encoded.len() * code_len);
                for c in &encoded {
                    flat.extend_from_slice(c);
                }

                let prep = enc.prepare_query(&q, metric);
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
                        "metric={metric:?} dim={dim} id={id}: scan {dist} != ref {ref_dist}"
                    );
                }
            }
        }
    }
}
