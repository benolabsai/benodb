// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::{Quantizer, VectorMetric};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Deterministic, stateless Rademacher sign function.
/// Maps dimension index `i` deterministically to +1.0 or -1.0 using SplitMix64.
/// Eliminates the DC energy spike on non-negative feature spaces (e.g., GIST, MNIST).
#[inline(always)]
pub fn rademacher_sign(i: usize) -> f32 {
    let mut x = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    if (x & 1) == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Converts an f32 scalar into a 2-byte IEEE 754-2008 binary16 (f16) in little-endian order.
#[inline(always)]
pub fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    let h_bits = if exp == 255 {
        // Inf / NaN
        ((sign as u16) << 15) | 0x7C00 | ((mant >> 13) as u16)
    } else if exp > 142 {
        // Overflow -> Clamp to finite max (~65504.0)
        ((sign as u16) << 15) | 0x7BFF
    } else if exp < 103 {
        // Underflow -> 0
        (sign as u16) << 15
    } else if exp < 113 {
        // Subnormal
        let m = (mant | 0x800000) >> (113 - exp);
        ((sign as u16) << 15) | ((m >> 13) as u16)
    } else {
        // Normal
        let new_exp = (exp - 127 + 15) as u16;
        let new_mant = (mant >> 13) as u16;
        ((sign as u16) << 15) | (new_exp << 10) | new_mant
    };
    h_bits.to_le_bytes()
}

/// Converts a 2-byte IEEE 754-2008 binary16 (f16) into an f32.
#[inline(always)]
pub fn f16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let h_bits = u16::from_le_bytes(bytes);
    let sign = ((h_bits >> 15) & 0x1) as u32;
    let exp = ((h_bits >> 10) & 0x1F) as u32;
    let mant = (h_bits & 0x3FF) as u32;

    let f_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 113;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf / NaN
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        // Normal
        let new_exp = exp + 127 - 15;
        (sign << 31) | (new_exp << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

fn default_true() -> bool {
    true
}

/// BenoQuant™ (Randomized Walsh-Hadamard Transform + Scalar Quantization)
///
/// High-performance vector quantization engine that leverages deterministic
/// Rademacher sign modulation combined with the Fast Walsh-Hadamard Transform (FWHT)
/// to achieve outlier-robust, isotropic scalar quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenoQuantEncoder {
    pub dim: usize,
    pub bits: usize,
    /// Global or segment-level min value for scaling
    pub offset: f32,
    /// Global or segment-level scale factor
    pub scale: f32,
    #[serde(default)]
    pub metric: VectorMetric,
    #[serde(default = "default_true")]
    pub use_rademacher: bool,
}

/// Backward compatibility alias for the legacy TurboQuant name.
pub type TurboQuantEncoder = BenoQuantEncoder;

impl BenoQuantEncoder {
    pub fn new(dim: usize, bits: usize, offset: f32, scale: f32) -> Self {
        Self {
            dim,
            bits,
            offset,
            scale,
            metric: VectorMetric::L2,
            use_rademacher: true,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Train the quantizer by finding the min/max values across a sample of rotated vectors.
    /// BenoQuant ALWAYS uses deterministic Rademacher sign modulation ($H \cdot D$)
    /// to eliminate DC energy spikes on non-negative domains.
    pub fn train(vectors: &[Vec<f32>], bits: usize, metric: VectorMetric) -> Result<Self> {
        let dim = vectors[0].len();
        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;

        // Sample up to 10k vectors for training
        let sample_size = vectors.len().min(10000);
        for i in 0..sample_size {
            let mut rotated = vectors[i].clone();
            if metric == VectorMetric::Cosine {
                let norm = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    let inv = 1.0 / norm;
                    for x in &mut rotated {
                        *x *= inv;
                    }
                } else if !rotated.is_empty() {
                    let inv_sqrt_d = 1.0 / (rotated.len() as f32).sqrt();
                    rotated.fill(inv_sqrt_d);
                }
            }
            // Pad to power of 2 for FWHT if necessary
            if !rotated.len().is_power_of_two() {
                let next_pow2 = rotated.len().next_power_of_two();
                rotated.resize(next_pow2, 0.0);
            }
            // Deterministic Rademacher sign modulation: eliminates DC spike on non-negative domains
            for (j, val) in rotated.iter_mut().enumerate() {
                *val *= rademacher_sign(j);
            }
            fwht(&mut rotated);

            for &val in &rotated {
                if val < global_min {
                    global_min = val;
                }
                if val > global_max {
                    global_max = val;
                }
            }
        }

        let scale = if (global_max - global_min).abs() < 1e-10 {
            1.0
        } else {
            ((1 << bits) - 1) as f32 / (global_max - global_min)
        };

        Ok(Self {
            dim,
            bits,
            offset: global_min,
            scale,
            metric,
            use_rademacher: true,
        })
    }

    /// Precompute the rotated, un-clamped query state once for asymmetric
    /// distance evaluation across encoded vectors.
    pub fn prepare_query(
        &self,
        query_f32: &[f32],
        metric: VectorMetric,
    ) -> PreparedBenoQuantQuery<'_> {
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if self.dim == 0 {
            return PreparedBenoQuantQuery {
                encoder: self,
                rotated_query: Vec::new(),
                q_scaled: Vec::new(),
                q_norm_sq: 0.0,
                inv_scale: 1.0,
                inv_n: 1.0,
                metric,
                #[cfg(target_arch = "x86_64")]
                use_avx2,
            };
        }

        let mut rotated_query = vec![0.0f32; self.dim];
        let copy_len = query_f32.len().min(self.dim);
        rotated_query[..copy_len].copy_from_slice(&query_f32[..copy_len]);

        if metric == VectorMetric::Cosine || self.metric == VectorMetric::Cosine {
            let norm = rotated_query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut rotated_query {
                    *x *= inv;
                }
            } else if !rotated_query.is_empty() {
                let inv_sqrt_d = 1.0 / (rotated_query.len() as f32).sqrt();
                rotated_query.fill(inv_sqrt_d);
            }
        }

        let n = self.dim.next_power_of_two();
        rotated_query.resize(n, 0.0);
        for (j, val) in rotated_query.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }
        fwht(&mut rotated_query);

        let q_norm_sq = rotated_query.iter().map(|x| x * x).sum::<f32>();
        let inv_scale = if self.scale.abs() < 1e-10 {
            1.0
        } else {
            1.0 / self.scale
        };
        let inv_n = 1.0 / (n.max(1) as f32);
        let q_scaled = if self.bits != 4
            && !matches!(metric, VectorMetric::Cosine | VectorMetric::InnerProduct)
        {
            let eff_scale = 1.0 / inv_scale;
            rotated_query
                .iter()
                .map(|&q| (q - self.offset) * eff_scale)
                .collect()
        } else {
            Vec::new()
        };

        PreparedBenoQuantQuery {
            encoder: self,
            rotated_query,
            q_scaled,
            q_norm_sq,
            inv_scale,
            inv_n,
            metric,
            #[cfg(target_arch = "x86_64")]
            use_avx2,
        }
    }
}

/// Pre-rotated, un-clamped query state for Asymmetric Distance Calculation (ADC)
/// against [`BenoQuantEncoder`] codes.
///
/// Computing `distance` directly in the Walsh-Hadamard rotated space avoids
/// per-candidate inverse FWHT transforms while returning calibrated Euclidean L2
/// (`sqrt(dist_sq / N)`) or Cosine (`1 - cos`) distances comparable across
/// segments with independent quantization ranges.
#[derive(Debug, Clone)]
pub struct PreparedBenoQuantQuery<'a> {
    encoder: &'a BenoQuantEncoder,
    rotated_query: Vec<f32>,
    /// Pre-shifted and scaled query `(rotated_query - offset) / inv_scale` for
    /// fast SIMD L2 u8 batch scanning (`(q_scaled[i] - code[i] as f32)^2`).
    q_scaled: Vec<f32>,
    q_norm_sq: f32,
    inv_scale: f32,
    inv_n: f32,
    metric: VectorMetric,
    #[cfg(target_arch = "x86_64")]
    use_avx2: bool,
}

/// Upper bound on the raw (pre-scale, pre-root) squared distance that can still
/// beat `cal_thresh` once `raw_sq * factor` is scaled back and rooted.
///
/// The margin covers floating-point rounding in the SIMD kernel so that a
/// candidate the exact `sqrt` path would have admitted is never pruned here.
#[inline]
fn raw_sq_threshold_for(cal_thresh: f32, factor: f32) -> f32 {
    if !cal_thresh.is_finite() || factor <= 0.0 {
        f32::INFINITY
    } else {
        ((cal_thresh * cal_thresh) / factor) * (1.0 + 1e-4) + 1e-6
    }
}

/// Squared L2 between the pre-scaled query and one `u8` code, four independent
/// FMA accumulators deep.
///
/// `q_scaled.len()` must be at least `code.len()`; callers pass slices of equal
/// length so no bounds check appears in the loop. Four chains cover the 4-cycle
/// `vfmadd` latency, which retires 32 dimensions per iteration instead of 16 on
/// two chains. A single 8-lane step and a scalar tail finish any remainder, so
/// every width from 0 upward is handled by the same function.
///
/// Measured against the two-chain form by
/// `bench_scan_l2_u8_codes_throughput`: 1.01x at 128 dimensions, 1.19x at 384,
/// 1.21x at 768, 1.24x at 1024. The 128-dimension case is almost entirely flat
/// because the loop body is too short to expose the dependency chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_sq_f32_u8_avx2(q_scaled: &[f32], code: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let len = code.len();
    debug_assert!(q_scaled.len() >= len);
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let mut i = 0usize;
    let q_ptr = q_scaled.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 32 <= len {
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i)), c0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 8) as *const __m128i
        )));
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i + 8)), c1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);

        let c2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 16) as *const __m128i
        )));
        let d2 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i + 16)), c2);
        sum2 = _mm256_fmadd_ps(d2, d2, sum2);

        let c3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 24) as *const __m128i
        )));
        let d3 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i + 24)), c3);
        sum3 = _mm256_fmadd_ps(d3, d3, sum3);

        i += 32;
    }

    while i + 8 <= len {
        let c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let d = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i)), c);
        sum0 = _mm256_fmadd_ps(d, d, sum0);
        i += 8;
    }

    let paired = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let mut total = hsum256_ps(paired);

    while i < len {
        let d = *q_ptr.add(i) - (*c_ptr.add(i) as f32);
        total += d * d;
        i += 1;
    }
    total
}

/// Feature-tagged driver for [`PreparedBenoQuantQuery::scan_l2_u8_codes`].
///
/// The candidate loop lives inside a `#[target_feature(enable = "avx2",
/// enable = "fma")]` function so that [`l2_sq_f32_u8_avx2`] is compiled
/// together with it: the code slice is taken once with `get_unchecked` against
/// the `chunks_exact` invariant, and the kernel's registers stay live across
/// the branch back to the top of the loop.
///
/// Moving the loop across this boundary measured 0.98x to 1.00x across
/// 128-to-1024 dimensions, so it is kept for the removed per-candidate bounds
/// check and as the natural place to hang candidate batching later - not
/// because the loop boundary itself pays.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scan_l2_u8_codes_avx2<F>(
    q_scaled: &[f32],
    codes: &[u8],
    code_len: usize,
    count: usize,
    factor: f32,
    initial_threshold: f32,
    on_hit: &mut F,
) where
    F: FnMut(usize, f32) -> f32,
{
    let mut threshold = initial_threshold;
    let mut raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
        // SAFETY: `chunks_exact(code_len)` yields exactly `code_len >= count`
        // bytes, and the caller sliced `q_scaled` to `count`.
        let code = code.get_unchecked(..count);
        let raw_sq = l2_sq_f32_u8_avx2(q_scaled, code);
        if raw_sq <= raw_sq_threshold {
            let dist = (raw_sq * factor).max(0.0).sqrt();
            if dist.is_finite() && dist < threshold {
                threshold = on_hit(local_id, dist);
                raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn l2_sq_f32_u8_scalar(q_scaled: &[f32], code: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (&q, &c) in q_scaled.iter().zip(code.iter()) {
        let d = q - (c as f32);
        sum += d * d;
    }
    sum
}

/// ARM64 NEON squared L2 distance over pre-scaled query and 8-bit codes.
/// Processes 8 dimensions per iteration across two 4-lane `float32x4_t`
/// accumulators (full Apple M-series / AWS Graviton parity with AVX2+FMA).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_sq_f32_u8_neon(q_scaled: &[f32], code: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let len = q_scaled.len().min(code.len());
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let q_ptr = q_scaled.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 8 <= len {
        let c8 = vld1_u8(c_ptr.add(i));
        let c16 = vmovl_u8(c8);
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));

        let d0 = vsubq_f32(vld1q_f32(q_ptr.add(i)), c0);
        sum0 = vfmaq_f32(sum0, d0, d0);

        let d1 = vsubq_f32(vld1q_f32(q_ptr.add(i + 4)), c1);
        sum1 = vfmaq_f32(sum1, d1, d1);

        i += 8;
    }

    let mut total = vaddvq_f32(vaddq_f32(sum0, sum1));
    while i < len {
        let d = *q_ptr.add(i) - (*c_ptr.add(i) as f32);
        total += d * d;
        i += 1;
    }
    total
}

/// ARM64 NEON dot product and decoded squared norm for 8-bit BenoQuant codes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_norm_f32_u8_neon(
    rotated_query: &[f32],
    code: &[u8],
    inv_scale: f32,
    offset: f32,
) -> (f32, f32) {
    use std::arch::aarch64::*;
    let len = rotated_query.len().min(code.len());
    let inv_scale_v = vdupq_n_f32(inv_scale);
    let offset_v = vdupq_n_f32(offset);
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut norm0 = vdupq_n_f32(0.0);
    let mut norm1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    let q_ptr = rotated_query.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 8 <= len {
        let c8 = vld1_u8(c_ptr.add(i));
        let c16 = vmovl_u8(c8);
        let c0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(c16)));
        let dec0 = vfmaq_f32(offset_v, c0, inv_scale_v);
        let q0 = vld1q_f32(q_ptr.add(i));
        dot0 = vfmaq_f32(dot0, q0, dec0);
        norm0 = vfmaq_f32(norm0, dec0, dec0);

        let c1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(c16)));
        let dec1 = vfmaq_f32(offset_v, c1, inv_scale_v);
        let q1 = vld1q_f32(q_ptr.add(i + 4));
        dot1 = vfmaq_f32(dot1, q1, dec1);
        norm1 = vfmaq_f32(norm1, dec1, dec1);

        i += 8;
    }

    let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut v_norm_sq = vaddvq_f32(vaddq_f32(norm0, norm1));
    while i < len {
        let dec = (*c_ptr.add(i) as f32) * inv_scale + offset;
        dot += *q_ptr.add(i) * dec;
        v_norm_sq += dec * dec;
        i += 1;
    }
    (dot, v_norm_sq)
}

/// ARM64 NEON squared L2 distance in rotated space for 4-bit nibble-packed
/// BenoQuant codes. Unpacks 8 packed bytes (16 dimensions) per iteration using
/// `vand_u8` / `vshr_n_u8::<4>` / `vzip1_u8` / `vzip2_u8` across 4 FMA pipes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l2_sq_f32_u4_neon(
    rotated_query: &[f32],
    nibble_bytes: &[u8],
    step: f32,
    base: f32,
) -> f32 {
    use std::arch::aarch64::*;
    let n = rotated_query
        .len()
        .min(nibble_bytes.len().saturating_mul(2));
    let step_v = vdupq_n_f32(step);
    let base_v = vdupq_n_f32(base);
    let mask_0f = vdup_n_u8(0x0F);
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let q_ptr = rotated_query.as_ptr();
    let c_ptr = nibble_bytes.as_ptr();
    let mut byte_i = 0usize;
    let mut dim_i = 0usize;

    while dim_i + 16 <= n && byte_i + 8 <= nibble_bytes.len() {
        let packed = vld1_u8(c_ptr.add(byte_i));
        let lo = vand_u8(packed, mask_0f);
        let hi = vshr_n_u8::<4>(packed);
        let first8 = vzip1_u8(lo, hi);
        let next8 = vzip2_u8(lo, hi);

        let f16 = vmovl_u8(first8);
        let v0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(f16)));
        let dec0 = vfmaq_f32(base_v, v0, step_v);
        let d0 = vsubq_f32(vld1q_f32(q_ptr.add(dim_i)), dec0);
        sum0 = vfmaq_f32(sum0, d0, d0);

        let v1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(f16)));
        let dec1 = vfmaq_f32(base_v, v1, step_v);
        let d1 = vsubq_f32(vld1q_f32(q_ptr.add(dim_i + 4)), dec1);
        sum1 = vfmaq_f32(sum1, d1, d1);

        let n16 = vmovl_u8(next8);
        let v2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(n16)));
        let dec2 = vfmaq_f32(base_v, v2, step_v);
        let d2 = vsubq_f32(vld1q_f32(q_ptr.add(dim_i + 8)), dec2);
        sum2 = vfmaq_f32(sum2, d2, d2);

        let v3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(n16)));
        let dec3 = vfmaq_f32(base_v, v3, step_v);
        let d3 = vsubq_f32(vld1q_f32(q_ptr.add(dim_i + 12)), dec3);
        sum3 = vfmaq_f32(sum3, d3, d3);

        byte_i += 8;
        dim_i += 16;
    }

    let mut total = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    while dim_i < n && byte_i < nibble_bytes.len() {
        let b = *c_ptr.add(byte_i);
        let dec_low = ((b & 0x0F) as f32) * step + base;
        let diff_low = *q_ptr.add(dim_i) - dec_low;
        total += diff_low * diff_low;
        dim_i += 1;
        if dim_i < n {
            let dec_high = (((b >> 4) & 0x0F) as f32) * step + base;
            let diff_high = *q_ptr.add(dim_i) - dec_high;
            total += diff_high * diff_high;
            dim_i += 1;
        }
        byte_i += 1;
    }
    total
}

/// ARM64 NEON dot product and decoded squared norm for 4-bit nibble-packed
/// BenoQuant codes.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_norm_f32_u4_neon(
    rotated_query: &[f32],
    nibble_bytes: &[u8],
    step: f32,
    base: f32,
) -> (f32, f32) {
    use std::arch::aarch64::*;
    let n = rotated_query
        .len()
        .min(nibble_bytes.len().saturating_mul(2));
    let step_v = vdupq_n_f32(step);
    let base_v = vdupq_n_f32(base);
    let mask_0f = vdup_n_u8(0x0F);
    let mut dot0 = vdupq_n_f32(0.0);
    let mut dot1 = vdupq_n_f32(0.0);
    let mut norm0 = vdupq_n_f32(0.0);
    let mut norm1 = vdupq_n_f32(0.0);

    let q_ptr = rotated_query.as_ptr();
    let c_ptr = nibble_bytes.as_ptr();
    let mut byte_i = 0usize;
    let mut dim_i = 0usize;

    while dim_i + 16 <= n && byte_i + 8 <= nibble_bytes.len() {
        let packed = vld1_u8(c_ptr.add(byte_i));
        let lo = vand_u8(packed, mask_0f);
        let hi = vshr_n_u8::<4>(packed);
        let first8 = vzip1_u8(lo, hi);
        let next8 = vzip2_u8(lo, hi);

        let f16 = vmovl_u8(first8);
        let dec0 = vfmaq_f32(base_v, vcvtq_f32_u32(vmovl_u16(vget_low_u16(f16))), step_v);
        dot0 = vfmaq_f32(dot0, vld1q_f32(q_ptr.add(dim_i)), dec0);
        norm0 = vfmaq_f32(norm0, dec0, dec0);

        let dec1 = vfmaq_f32(base_v, vcvtq_f32_u32(vmovl_u16(vget_high_u16(f16))), step_v);
        dot1 = vfmaq_f32(dot1, vld1q_f32(q_ptr.add(dim_i + 4)), dec1);
        norm1 = vfmaq_f32(norm1, dec1, dec1);

        let n16 = vmovl_u8(next8);
        let dec2 = vfmaq_f32(base_v, vcvtq_f32_u32(vmovl_u16(vget_low_u16(n16))), step_v);
        dot0 = vfmaq_f32(dot0, vld1q_f32(q_ptr.add(dim_i + 8)), dec2);
        norm0 = vfmaq_f32(norm0, dec2, dec2);

        let dec3 = vfmaq_f32(base_v, vcvtq_f32_u32(vmovl_u16(vget_high_u16(n16))), step_v);
        dot1 = vfmaq_f32(dot1, vld1q_f32(q_ptr.add(dim_i + 12)), dec3);
        norm1 = vfmaq_f32(norm1, dec3, dec3);

        byte_i += 8;
        dim_i += 16;
    }

    let mut dot = vaddvq_f32(vaddq_f32(dot0, dot1));
    let mut v_norm_sq = vaddvq_f32(vaddq_f32(norm0, norm1));
    while dim_i < n && byte_i < nibble_bytes.len() {
        let b = *c_ptr.add(byte_i);
        let dec_low = ((b & 0x0F) as f32) * step + base;
        dot += *q_ptr.add(dim_i) * dec_low;
        v_norm_sq += dec_low * dec_low;
        dim_i += 1;
        if dim_i < n {
            let dec_high = (((b >> 4) & 0x0F) as f32) * step + base;
            dot += *q_ptr.add(dim_i) * dec_high;
            v_norm_sq += dec_high * dec_high;
            dim_i += 1;
        }
        byte_i += 1;
    }
    (dot, v_norm_sq)
}

/// AVX2+FMA horizontal sum of an 8-lane `f32` register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    lanes.iter().sum()
}

/// AVX2+FMA dot product and decoded squared norm for 8-bit BenoQuant codes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_norm_f32_u8_avx2(
    rotated_query: &[f32],
    code: &[u8],
    inv_scale: f32,
    offset: f32,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let len = rotated_query.len().min(code.len());
    let inv_scale_v = _mm256_set1_ps(inv_scale);
    let offset_v = _mm256_set1_ps(offset);
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut norm0 = _mm256_setzero_ps();
    let mut norm1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let q_ptr = rotated_query.as_ptr();
    let c_ptr = code.as_ptr();

    while i + 16 <= len {
        let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i) as *const __m128i
        )));
        let dec0 = _mm256_fmadd_ps(c0, inv_scale_v, offset_v);
        let q0 = _mm256_loadu_ps(q_ptr.add(i));
        dot0 = _mm256_fmadd_ps(q0, dec0, dot0);
        norm0 = _mm256_fmadd_ps(dec0, dec0, norm0);

        let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
            c_ptr.add(i + 8) as *const __m128i
        )));
        let dec1 = _mm256_fmadd_ps(c1, inv_scale_v, offset_v);
        let q1 = _mm256_loadu_ps(q_ptr.add(i + 8));
        dot1 = _mm256_fmadd_ps(q1, dec1, dot1);
        norm1 = _mm256_fmadd_ps(dec1, dec1, norm1);

        i += 16;
    }

    let mut dot = hsum256_ps(_mm256_add_ps(dot0, dot1));
    let mut v_norm_sq = hsum256_ps(_mm256_add_ps(norm0, norm1));

    while i < len {
        let dec = (*c_ptr.add(i) as f32) * inv_scale + offset;
        dot += *q_ptr.add(i) * dec;
        v_norm_sq += dec * dec;
        i += 1;
    }
    (dot, v_norm_sq)
}

/// AVX2+FMA squared L2 distance in rotated space for 4-bit nibble-packed
/// BenoQuant codes (`nibble_bytes` packs two 4-bit codes per byte: low = even
/// dim, high = odd dim). Processes 16 dimensions (8 packed bytes) per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_sq_f32_u4_avx2(
    rotated_query: &[f32],
    nibble_bytes: &[u8],
    step: f32,
    base: f32,
) -> f32 {
    use std::arch::x86_64::*;
    let n = rotated_query
        .len()
        .min(nibble_bytes.len().saturating_mul(2));
    let step_v = _mm256_set1_ps(step);
    let base_v = _mm256_set1_ps(base);
    let mask_0f = _mm_set1_epi8(0x0F);
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();

    let q_ptr = rotated_query.as_ptr();
    let c_ptr = nibble_bytes.as_ptr();
    let mut byte_i = 0usize;
    let mut dim_i = 0usize;

    while dim_i + 16 <= n && byte_i + 8 <= nibble_bytes.len() {
        let packed = _mm_loadl_epi64(c_ptr.add(byte_i) as *const __m128i);
        let lo = _mm_and_si128(packed, mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
        let interleaved = _mm_unpacklo_epi8(lo, hi);

        let v0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
        let dec0 = _mm256_fmadd_ps(v0, step_v, base_v);
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(dim_i)), dec0);
        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

        let hi64 = _mm_srli_si128::<8>(interleaved);
        let v1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi64));
        let dec1 = _mm256_fmadd_ps(v1, step_v, base_v);
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(dim_i + 8)), dec1);
        sum1 = _mm256_fmadd_ps(d1, d1, sum1);

        byte_i += 8;
        dim_i += 16;
    }

    let mut total = hsum256_ps(_mm256_add_ps(sum0, sum1));
    while dim_i < n && byte_i < nibble_bytes.len() {
        let b = *c_ptr.add(byte_i);
        let dec_low = ((b & 0x0F) as f32) * step + base;
        let diff_low = *q_ptr.add(dim_i) - dec_low;
        total += diff_low * diff_low;
        dim_i += 1;
        if dim_i < n {
            let dec_high = (((b >> 4) & 0x0F) as f32) * step + base;
            let diff_high = *q_ptr.add(dim_i) - dec_high;
            total += diff_high * diff_high;
            dim_i += 1;
        }
        byte_i += 1;
    }
    total
}

/// AVX2+FMA dot product and decoded squared norm for 4-bit nibble-packed
/// BenoQuant codes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_norm_f32_u4_avx2(
    rotated_query: &[f32],
    nibble_bytes: &[u8],
    step: f32,
    base: f32,
) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = rotated_query
        .len()
        .min(nibble_bytes.len().saturating_mul(2));
    let step_v = _mm256_set1_ps(step);
    let base_v = _mm256_set1_ps(base);
    let mask_0f = _mm_set1_epi8(0x0F);
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut norm0 = _mm256_setzero_ps();
    let mut norm1 = _mm256_setzero_ps();

    let q_ptr = rotated_query.as_ptr();
    let c_ptr = nibble_bytes.as_ptr();
    let mut byte_i = 0usize;
    let mut dim_i = 0usize;

    while dim_i + 16 <= n && byte_i + 8 <= nibble_bytes.len() {
        let packed = _mm_loadl_epi64(c_ptr.add(byte_i) as *const __m128i);
        let lo = _mm_and_si128(packed, mask_0f);
        let hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask_0f);
        let interleaved = _mm_unpacklo_epi8(lo, hi);

        let v0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(interleaved));
        let dec0 = _mm256_fmadd_ps(v0, step_v, base_v);
        let q0 = _mm256_loadu_ps(q_ptr.add(dim_i));
        dot0 = _mm256_fmadd_ps(q0, dec0, dot0);
        norm0 = _mm256_fmadd_ps(dec0, dec0, norm0);

        let hi64 = _mm_srli_si128::<8>(interleaved);
        let v1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi64));
        let dec1 = _mm256_fmadd_ps(v1, step_v, base_v);
        let q1 = _mm256_loadu_ps(q_ptr.add(dim_i + 8));
        dot1 = _mm256_fmadd_ps(q1, dec1, dot1);
        norm1 = _mm256_fmadd_ps(dec1, dec1, norm1);

        byte_i += 8;
        dim_i += 16;
    }

    let mut dot = hsum256_ps(_mm256_add_ps(dot0, dot1));
    let mut v_norm_sq = hsum256_ps(_mm256_add_ps(norm0, norm1));
    while dim_i < n && byte_i < nibble_bytes.len() {
        let b = *c_ptr.add(byte_i);
        let dec_low = ((b & 0x0F) as f32) * step + base;
        dot += *q_ptr.add(dim_i) * dec_low;
        v_norm_sq += dec_low * dec_low;
        dim_i += 1;
        if dim_i < n {
            let dec_high = (((b >> 4) & 0x0F) as f32) * step + base;
            dot += *q_ptr.add(dim_i) * dec_high;
            v_norm_sq += dec_high * dec_high;
            dim_i += 1;
        }
        byte_i += 1;
    }
    (dot, v_norm_sq)
}

#[inline]
fn split_u4_code(encoded: &[u8], packed_len: usize, encoder_dim: usize) -> (&[u8], f32) {
    let (nibble_bytes, alpha) = if encoded.len() >= packed_len + 2 {
        let tail = encoded.len();
        let alpha_bytes = [encoded[tail - 2], encoded[tail - 1]];
        (&encoded[..packed_len], f16_bytes_to_f32(alpha_bytes))
    } else if encoded.len() >= 2 && (encoded.len() - 2) * 2 >= encoder_dim {
        let tail = encoded.len();
        let alpha_bytes = [encoded[tail - 2], encoded[tail - 1]];
        (&encoded[..tail - 2], f16_bytes_to_f32(alpha_bytes))
    } else {
        (encoded, 1.0f32)
    };
    let alpha = if alpha.is_finite() { alpha } else { 1.0 };
    (nibble_bytes, alpha)
}

impl<'a> PreparedBenoQuantQuery<'a> {
    /// Evaluate calibrated asymmetric distance against a stored BenoQuant code.
    /// Dispatches to AVX2+FMA on x86_64 and NEON on aarch64 for both 8-bit and
    /// 4-bit codes so per-segment HNSW graph traversal gets full SIMD parity.
    /// Guaranteed finite and non-negative.
    #[inline]
    pub fn distance(&self, encoded: &[u8]) -> f32 {
        let n = self.rotated_query.len();
        if n == 0 {
            return 0.0;
        }

        let dist = if self.encoder.bits == 4 {
            let packed_len = n.div_ceil(2);
            let (nibble_bytes, alpha) = split_u4_code(encoded, packed_len, self.encoder.dim);
            let step = alpha * self.inv_scale;
            let base = alpha * self.encoder.offset;

            match self.metric {
                VectorMetric::Cosine | VectorMetric::InnerProduct => {
                    let (dot, v_norm_sq) = {
                        #[cfg(target_arch = "aarch64")]
                        {
                            // SAFETY: NEON is mandatory baseline on aarch64.
                            unsafe {
                                dot_norm_f32_u4_neon(&self.rotated_query, nibble_bytes, step, base)
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            #[cfg(target_arch = "x86_64")]
                            if self.use_avx2 {
                                // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                                unsafe {
                                    dot_norm_f32_u4_avx2(
                                        &self.rotated_query,
                                        nibble_bytes,
                                        step,
                                        base,
                                    )
                                }
                            } else {
                                let mut dot = 0.0f32;
                                let mut v_norm_sq = 0.0f32;
                                let mut idx = 0usize;
                                for &b in nibble_bytes {
                                    if idx >= n {
                                        break;
                                    }
                                    let decoded_low = ((b & 0x0F) as f32) * step + base;
                                    dot += self.rotated_query[idx] * decoded_low;
                                    v_norm_sq += decoded_low * decoded_low;
                                    idx += 1;
                                    if idx < n {
                                        let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                                        dot += self.rotated_query[idx] * decoded_high;
                                        v_norm_sq += decoded_high * decoded_high;
                                        idx += 1;
                                    }
                                }
                                (dot, v_norm_sq)
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                let mut dot = 0.0f32;
                                let mut v_norm_sq = 0.0f32;
                                let mut idx = 0usize;
                                for &b in nibble_bytes {
                                    if idx >= n {
                                        break;
                                    }
                                    let decoded_low = ((b & 0x0F) as f32) * step + base;
                                    dot += self.rotated_query[idx] * decoded_low;
                                    v_norm_sq += decoded_low * decoded_low;
                                    idx += 1;
                                    if idx < n {
                                        let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                                        dot += self.rotated_query[idx] * decoded_high;
                                        v_norm_sq += decoded_high * decoded_high;
                                        idx += 1;
                                    }
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
                _ => {
                    let dist_sq = {
                        #[cfg(target_arch = "aarch64")]
                        {
                            // SAFETY: NEON is mandatory baseline on aarch64.
                            unsafe {
                                l2_sq_f32_u4_neon(&self.rotated_query, nibble_bytes, step, base)
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            #[cfg(target_arch = "x86_64")]
                            if self.use_avx2 {
                                // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                                unsafe {
                                    l2_sq_f32_u4_avx2(&self.rotated_query, nibble_bytes, step, base)
                                }
                            } else {
                                let mut dist_sq = 0.0f32;
                                let mut idx = 0usize;
                                for &b in nibble_bytes {
                                    if idx >= n {
                                        break;
                                    }
                                    let decoded_low = ((b & 0x0F) as f32) * step + base;
                                    let diff_low = self.rotated_query[idx] - decoded_low;
                                    dist_sq += diff_low * diff_low;
                                    idx += 1;
                                    if idx < n {
                                        let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                                        let diff_high = self.rotated_query[idx] - decoded_high;
                                        dist_sq += diff_high * diff_high;
                                        idx += 1;
                                    }
                                }
                                dist_sq
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                let mut dist_sq = 0.0f32;
                                let mut idx = 0usize;
                                for &b in nibble_bytes {
                                    if idx >= n {
                                        break;
                                    }
                                    let decoded_low = ((b & 0x0F) as f32) * step + base;
                                    let diff_low = self.rotated_query[idx] - decoded_low;
                                    dist_sq += diff_low * diff_low;
                                    idx += 1;
                                    if idx < n {
                                        let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                                        let diff_high = self.rotated_query[idx] - decoded_high;
                                        dist_sq += diff_high * diff_high;
                                        idx += 1;
                                    }
                                }
                                dist_sq
                            }
                        }
                    };
                    (dist_sq * self.inv_n).max(0.0).sqrt()
                }
            }
        } else {
            let count = n.min(encoded.len());
            match self.metric {
                VectorMetric::Cosine | VectorMetric::InnerProduct => {
                    let (dot, v_norm_sq) = {
                        #[cfg(target_arch = "aarch64")]
                        {
                            // SAFETY: NEON is mandatory baseline on aarch64.
                            unsafe {
                                dot_norm_f32_u8_neon(
                                    &self.rotated_query[..count],
                                    &encoded[..count],
                                    self.inv_scale,
                                    self.encoder.offset,
                                )
                            }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            #[cfg(target_arch = "x86_64")]
                            if self.use_avx2 {
                                // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                                unsafe {
                                    dot_norm_f32_u8_avx2(
                                        &self.rotated_query[..count],
                                        &encoded[..count],
                                        self.inv_scale,
                                        self.encoder.offset,
                                    )
                                }
                            } else {
                                let mut dot = 0.0f32;
                                let mut v_norm_sq = 0.0f32;
                                for (&q, &e) in
                                    self.rotated_query[..count].iter().zip(&encoded[..count])
                                {
                                    let decoded = (e as f32) * self.inv_scale + self.encoder.offset;
                                    dot += q * decoded;
                                    v_norm_sq += decoded * decoded;
                                }
                                (dot, v_norm_sq)
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                let mut dot = 0.0f32;
                                let mut v_norm_sq = 0.0f32;
                                for (&q, &e) in
                                    self.rotated_query[..count].iter().zip(&encoded[..count])
                                {
                                    let decoded = (e as f32) * self.inv_scale + self.encoder.offset;
                                    dot += q * decoded;
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
                _ => {
                    let count = self.q_scaled.len().min(encoded.len());
                    let factor = self.inv_scale * self.inv_scale * self.inv_n;
                    let raw_sq = {
                        #[cfg(target_arch = "aarch64")]
                        {
                            // SAFETY: NEON is mandatory baseline on aarch64.
                            unsafe { l2_sq_f32_u8_neon(&self.q_scaled[..count], &encoded[..count]) }
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            #[cfg(target_arch = "x86_64")]
                            if self.use_avx2 {
                                // SAFETY: `avx2` and `fma` were verified in `prepare_query`.
                                unsafe {
                                    l2_sq_f32_u8_avx2(&self.q_scaled[..count], &encoded[..count])
                                }
                            } else {
                                l2_sq_f32_u8_scalar(&self.q_scaled[..count], &encoded[..count])
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                l2_sq_f32_u8_scalar(&self.q_scaled[..count], &encoded[..count])
                            }
                        }
                    };
                    (raw_sq * factor).max(0.0).sqrt()
                }
            }
        };

        if dist.is_finite() {
            dist.max(0.0)
        } else {
            f32::MAX
        }
    }

    /// True when this prepared query can use the fast AVX2/unrolled L2 u8 batch
    /// scan path (8-bit non-cosine/IP).
    #[inline]
    pub fn is_fast_l2_u8(&self) -> bool {
        !self.q_scaled.is_empty()
    }

    /// Linearly scan a contiguous block of `u8` codes of uniform stride `code_len`,
    /// invoking `on_hit(local_id, calibrated_dist)` whenever a code beats the
    /// running calibrated distance threshold. `on_hit` returns the updated
    /// threshold (e.g. the new worst distance in the caller's top-k heap).
    ///
    /// Runtime SIMD detection (`avx2` + `fma`) is hoisted once per call, the
    /// squared-space threshold avoids per-vector `sqrt` and scaling for
    /// candidates that cannot enter the top-k heap, and the L2 kernel runs four
    /// FMA accumulators deep (32 dimensions per iteration).
    pub fn scan_l2_u8_codes<F>(
        &self,
        codes: &[u8],
        code_len: usize,
        initial_threshold: f32,
        mut on_hit: F,
    ) where
        F: FnMut(usize, f32) -> f32,
    {
        if code_len == 0 || self.q_scaled.is_empty() {
            return;
        }
        let count = self.q_scaled.len().min(code_len);
        let q_scaled = &self.q_scaled[..count];
        let factor = self.inv_scale * self.inv_scale * self.inv_n;

        #[cfg(target_arch = "aarch64")]
        {
            let mut threshold = initial_threshold;
            let mut raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                // SAFETY: NEON is mandatory baseline on aarch64.
                let raw_sq = unsafe { l2_sq_f32_u8_neon(q_scaled, &code[..count]) };
                if raw_sq <= raw_sq_threshold {
                    let dist = (raw_sq * factor).max(0.0).sqrt();
                    if dist.is_finite() && dist < threshold {
                        threshold = on_hit(local_id, dist);
                        raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
                    }
                }
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: `avx2` and `fma` were detected immediately above, and
                    // `chunks_exact(code_len)` bounds every code slice to `code_len`
                    // bytes, which is at least `count`.
                    unsafe {
                        scan_l2_u8_codes_avx2(
                            q_scaled,
                            codes,
                            code_len,
                            count,
                            factor,
                            initial_threshold,
                            &mut on_hit,
                        );
                    }
                    return;
                }
            }

            let mut threshold = initial_threshold;
            let mut raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                let raw_sq = l2_sq_f32_u8_scalar(q_scaled, &code[..count]);
                if raw_sq <= raw_sq_threshold {
                    let dist = (raw_sq * factor).max(0.0).sqrt();
                    if dist.is_finite() && dist < threshold {
                        threshold = on_hit(local_id, dist);
                        raw_sq_threshold = raw_sq_threshold_for(threshold, factor);
                    }
                }
            }
        }
    }

    /// Unified SIMD-accelerated batch scan over contiguous BenoQuant codes
    /// (8-bit and 4-bit, L2 and Cosine/IP). Uses hoisted AVX2+FMA on x86_64
    /// and native NEON on aarch64 (Apple Silicon / AWS Graviton), pruning L2
    /// candidates in squared space before `sqrt`.
    pub fn scan_codes<F>(
        &self,
        codes: &[u8],
        code_len: usize,
        initial_threshold: f32,
        mut on_hit: F,
    ) where
        F: FnMut(usize, f32) -> f32,
    {
        if code_len == 0 || self.rotated_query.is_empty() {
            return;
        }
        if self.is_fast_l2_u8() {
            self.scan_l2_u8_codes(codes, code_len, initial_threshold, on_hit);
            return;
        }

        let n = self.rotated_query.len();
        let is_cosine = matches!(
            self.metric,
            VectorMetric::Cosine | VectorMetric::InnerProduct
        );

        let mut threshold = initial_threshold;

        #[cfg(target_arch = "aarch64")]
        {
            if self.encoder.bits == 4 {
                let packed_len = n.div_ceil(2);
                if is_cosine {
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        let (nibble_bytes, alpha) =
                            split_u4_code(code, packed_len, self.encoder.dim);
                        let step = alpha * self.inv_scale;
                        let base = alpha * self.encoder.offset;
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        let (dot, v_norm_sq) = unsafe {
                            dot_norm_f32_u4_neon(&self.rotated_query, nibble_bytes, step, base)
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
                } else {
                    let mut raw_sq_threshold = raw_sq_threshold_for(threshold, self.inv_n);
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        let (nibble_bytes, alpha) =
                            split_u4_code(code, packed_len, self.encoder.dim);
                        let step = alpha * self.inv_scale;
                        let base = alpha * self.encoder.offset;
                        // SAFETY: NEON is mandatory baseline on aarch64.
                        let raw_sq = unsafe {
                            l2_sq_f32_u4_neon(&self.rotated_query, nibble_bytes, step, base)
                        };
                        if raw_sq <= raw_sq_threshold {
                            let dist = (raw_sq * self.inv_n).max(0.0).sqrt();
                            if dist.is_finite() && dist < threshold {
                                threshold = on_hit(local_id, dist);
                                raw_sq_threshold = raw_sq_threshold_for(threshold, self.inv_n);
                            }
                        }
                    }
                }
            } else if is_cosine {
                let count = n.min(code_len);
                let rq = &self.rotated_query[..count];
                for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                    // SAFETY: NEON is mandatory baseline on aarch64.
                    let (dot, v_norm_sq) = unsafe {
                        dot_norm_f32_u8_neon(
                            rq,
                            &code[..count],
                            self.inv_scale,
                            self.encoder.offset,
                        )
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
            } else {
                for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                    let dist = self.distance(code);
                    if dist < threshold {
                        threshold = on_hit(local_id, dist);
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

            if self.encoder.bits == 4 {
                let packed_len = n.div_ceil(2);
                if is_cosine {
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        {
                            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                                let (nibble_bytes, alpha) =
                                    split_u4_code(code, packed_len, self.encoder.dim);
                                let step = alpha * self.inv_scale;
                                let base = alpha * self.encoder.offset;
                                // SAFETY: `avx2` and `fma` were verified above.
                                let (dot, v_norm_sq) = unsafe {
                                    dot_norm_f32_u4_avx2(
                                        &self.rotated_query,
                                        nibble_bytes,
                                        step,
                                        base,
                                    )
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
                    }
                } else {
                    let mut raw_sq_threshold = raw_sq_threshold_for(threshold, self.inv_n);
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        {
                            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                                let (nibble_bytes, alpha) =
                                    split_u4_code(code, packed_len, self.encoder.dim);
                                let step = alpha * self.inv_scale;
                                let base = alpha * self.encoder.offset;
                                // SAFETY: `avx2` and `fma` were verified above.
                                let raw_sq = unsafe {
                                    l2_sq_f32_u4_avx2(&self.rotated_query, nibble_bytes, step, base)
                                };
                                if raw_sq <= raw_sq_threshold {
                                    let dist = (raw_sq * self.inv_n).max(0.0).sqrt();
                                    if dist.is_finite() && dist < threshold {
                                        threshold = on_hit(local_id, dist);
                                        raw_sq_threshold =
                                            raw_sq_threshold_for(threshold, self.inv_n);
                                    }
                                }
                            }
                            return;
                        }
                    }
                }
            } else if is_cosine && use_avx2 {
                #[cfg(target_arch = "x86_64")]
                {
                    let count = n.min(code_len);
                    let rq = &self.rotated_query[..count];
                    for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                        // SAFETY: `avx2` and `fma` were verified above.
                        let (dot, v_norm_sq) = unsafe {
                            dot_norm_f32_u8_avx2(
                                rq,
                                &code[..count],
                                self.inv_scale,
                                self.encoder.offset,
                            )
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
            }

            // Portable scalar fallback for non-AVX2/non-aarch64 hosts.
            for (local_id, code) in codes.chunks_exact(code_len).enumerate() {
                let dist = self.distance(code);
                if dist < threshold {
                    threshold = on_hit(local_id, dist);
                }
            }
        }
    }
}

impl Quantizer for BenoQuantEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut rotated = vector.to_vec();
        if self.metric == VectorMetric::Cosine {
            let norm = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut rotated {
                    *x *= inv;
                }
            } else if !rotated.is_empty() {
                let inv_sqrt_d = 1.0 / (rotated.len() as f32).sqrt();
                rotated.fill(inv_sqrt_d);
            }
        }
        // Pad to power of 2
        if !rotated.len().is_power_of_two() {
            let next_pow2 = rotated.len().next_power_of_two();
            rotated.resize(next_pow2, 0.0);
        }
        // Deterministic Rademacher sign modulation: eliminates DC spike on non-negative domains
        for (j, val) in rotated.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }
        fwht(&mut rotated);

        let max_val = ((1 << self.bits) - 1) as f32;
        let quantized: Vec<u8> = rotated
            .iter()
            .map(|&val| {
                let q = ((val - self.offset) * self.scale).round();
                q.clamp(0.0, max_val) as u8
            })
            .collect();

        if self.bits == 4 {
            // Compute optimal 2-byte least-squares length renormalization scalar:
            // alpha = <y, y_hat> / ||y_hat||^2
            let mut dot = 0.0f32;
            let mut norm_sq = 0.0f32;
            let inv_scale = 1.0 / self.scale;
            for (&y, &q) in rotated.iter().zip(quantized.iter()) {
                let y_hat = (q as f32 * inv_scale) + self.offset;
                dot += y * y_hat;
                norm_sq += y_hat * y_hat;
            }
            let alpha = if norm_sq > 1e-12 {
                (dot / norm_sq).clamp(0.01, 100.0)
            } else {
                1.0
            };
            let alpha_bytes = f32_to_f16_bytes(alpha);

            // Pack two 4-bit nibbles into one byte
            let mut packed = Vec::with_capacity(quantized.len().div_ceil(2) + 2);
            for chunk in quantized.chunks(2) {
                let low = chunk[0] & 0x0F;
                let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
                packed.push(low | (high << 4));
            }
            // Append 2-byte float16 scalar
            packed.extend_from_slice(&alpha_bytes);
            packed
        } else {
            quantized
        }
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let (decoded_bytes, alpha) = if self.bits == 4 {
            // Check for trailing 2-byte f16 scalar
            let (nibble_bytes, alpha) = if bytes.len() >= 2 && (bytes.len() - 2) * 2 >= self.dim {
                let alpha_bytes = [bytes[bytes.len() - 2], bytes[bytes.len() - 1]];
                (&bytes[..bytes.len() - 2], f16_bytes_to_f32(alpha_bytes))
            } else {
                (bytes, 1.0f32)
            };
            let mut unpacked = Vec::with_capacity(nibble_bytes.len() * 2);
            for &b in nibble_bytes {
                unpacked.push(b & 0x0F);
                unpacked.push((b >> 4) & 0x0F);
            }
            (unpacked, alpha)
        } else {
            (bytes.to_vec(), 1.0f32)
        };

        let mut rotated: Vec<f32> = decoded_bytes
            .iter()
            .map(|&b| ((b as f32 / self.scale) + self.offset) * alpha)
            .collect();

        // Invert FWHT (FWHT is self-inverse up to scaling by 1/N)
        fwht(&mut rotated);
        let n = rotated.len() as f32;
        rotated.iter_mut().for_each(|x| *x /= n);

        // Invert Rademacher sign modulation: s_j * s_j = 1
        for (j, val) in rotated.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }

        // Truncate to original dimension
        rotated.truncate(self.dim);
        rotated
    }

    fn distance_adc(&self, query_f32: &[f32], encoded: &[u8]) -> f32 {
        self.prepare_query(query_f32, self.metric).distance(encoded)
    }

    fn name(&self) -> String {
        format!("benoquant_{}bit", self.bits)
    }

    fn bits(&self) -> usize {
        self.bits
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Fast Walsh-Hadamard Transform (FWHT) - Portable Implementation
pub fn fwht(data: &mut [f32]) {
    let n = data.len();
    if !n.is_power_of_two() {
        return;
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rademacher_sign_balance() {
        let count = 1024;
        let pos = (0..count).filter(|&i| rademacher_sign(i) > 0.0).count();
        let neg = count - pos;
        let diff = (pos as isize - neg as isize).abs();
        // Over 1024 dimensions, difference between +1 and -1 should be less than 80 (~5%)
        assert!(
            diff < 80,
            "Rademacher signs should be approximately balanced: pos={}, neg={}",
            pos,
            neg
        );
    }

    #[test]
    fn test_dc_spike_elimination() {
        // Create a non-negative domain vector (e.g. all 1.0s, dimension 1024)
        let dim = 1024;
        let non_negative = vec![1.0f32; dim];

        // 1. Unmodulated FWHT produces massive coordinate-0 DC spike
        let mut unmodulated = non_negative.clone();
        fwht(&mut unmodulated);
        assert_eq!(unmodulated[0], 1024.0);
        for &v in &unmodulated[1..] {
            assert_eq!(v, 0.0);
        }

        // 2. Rademacher-modulated FWHT distributes energy isotropically
        let mut modulated = non_negative.clone();
        for (j, v) in modulated.iter_mut().enumerate() {
            *v *= rademacher_sign(j);
        }
        fwht(&mut modulated);

        // Modulated coordinate 0 should no longer be 1024.0! It should be scaled O(sqrt(d))
        assert!(
            modulated[0].abs() < 100.0,
            "Modulated coordinate 0 should be bounded by O(sqrt(d)), got {}",
            modulated[0]
        );
    }

    #[test]
    fn test_benoquant_cosine_monotonicity() {
        let dim = 64;
        let mut rng_state: u64 = 42;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..200 {
            let scale = 1.0 + rand_f32().abs() * 10.0;
            let v: Vec<f32> = (0..dim).map(|_| rand_f32() * scale).collect();
            vectors.push(v);
        }

        let encoder = BenoQuantEncoder::train(&vectors, 8, VectorMetric::Cosine).unwrap();
        assert_eq!(encoder.metric, VectorMetric::Cosine);

        let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| encoder.encode(v)).collect();

        let q: Vec<f32> = (0..dim).map(|_| rand_f32() * 5.0).collect();

        let mut exact_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::core::index::distance::cosine_distance(&q, v)))
            .collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();

        let q_encoded = encoder.encode(&q);
        let dist_fn = crate::core::index::distance::DistL2u8;
        let mut bq_dists: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dist_fn.distance(&q_encoded, v)))
            .collect();
        bq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bq_top10: std::collections::HashSet<usize> =
            bq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let overlap = exact_top10.intersection(&bq_top10).count();
        assert!(
            overlap >= 8,
            "BenoQuant Cosine top-10 overlap should be >= 8, got {}",
            overlap
        );
    }

    #[test]
    fn test_benoquant_4bit_recall_monotonicity() {
        let dim = 64;
        let mut rng_state: u64 = 12345;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..200 {
            let scale = 1.0 + rand_f32().abs() * 5.0;
            let v: Vec<f32> = (0..dim).map(|_| rand_f32() * scale).collect();
            vectors.push(v);
        }

        let encoder = BenoQuantEncoder::train(&vectors, 4, VectorMetric::L2).unwrap();
        assert_eq!(encoder.bits, 4);

        let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| encoder.encode(v)).collect();
        // 64 dimensions / 2 nibbles per byte + 2 bytes f16 scalar = 34 bytes
        assert_eq!(encoded[0].len(), dim / 2 + 2);

        let q: Vec<f32> = (0..dim).map(|_| rand_f32() * 3.0).collect();

        let mut exact_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::core::index::distance::l2_distance(&q, v)))
            .collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();

        let q_encoded = encoder.encode(&q);
        assert_eq!(q_encoded.len(), dim / 2 + 2);

        let dist_fn = crate::core::index::distance::DistL2u4;
        let self_dist = dist_fn.distance(&q_encoded, &q_encoded);
        assert!(
            self_dist.abs() < 1e-4,
            "Self distance should be ~0, got {self_dist}"
        );

        let mut bq_dists: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dist_fn.distance(&q_encoded, v)))
            .collect();
        bq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bq_top10: std::collections::HashSet<usize> =
            bq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let overlap = exact_top10.intersection(&bq_top10).count();
        assert!(
            overlap >= 6,
            "BenoQuant 4-bit L2 top-10 overlap should be >= 6 (got {})",
            overlap
        );
    }

    #[test]
    fn test_benoquant_adc_calibrated_l2_and_out_of_range_query() {
        let dim = 8;
        let seg1: Vec<Vec<f32>> = vec![vec![0.0; dim], vec![25.0; dim], vec![50.0; dim]];
        let seg2: Vec<Vec<f32>> = vec![vec![194.0; dim], vec![197.0; dim], vec![200.0; dim]];
        let query = vec![197.0f32; dim];

        for bits in [8usize, 4usize] {
            let enc1 = BenoQuantEncoder::train(&seg1, bits, VectorMetric::L2).unwrap();
            let enc2 = BenoQuantEncoder::train(&seg2, bits, VectorMetric::L2).unwrap();

            let code_0 = enc1.encode(&seg1[0]);
            let code_50 = enc1.encode(&seg1[2]);
            let code_197 = enc2.encode(&seg2[1]);

            let prep1 = enc1.prepare_query(&query, VectorMetric::L2);
            let d_0 = prep1.distance(&code_0);
            let d_50 = prep1.distance(&code_50);
            let d_197 = enc2
                .prepare_query(&query, VectorMetric::L2)
                .distance(&code_197);

            let expected_50 = (197.0 - 50.0) * (dim as f32).sqrt();
            let tol_far = if bits == 8 { 5.0 } else { 35.0 };
            let max_exact = if bits == 8 { 3.0 } else { 45.0 };
            assert!(
                (d_50 - expected_50).abs() < tol_far,
                "bits={bits}: out-of-range query to [50; 8] must stay calibrated near {expected_50}, got {d_50}"
            );
            assert!(
                d_50 < d_0,
                "bits={bits}: out-of-range ordering [50; 8] ({d_50}) must beat [0; 8] ({d_0})"
            );
            assert!(
                d_197 < max_exact,
                "bits={bits}: exact match [197; 8] must score < {max_exact}, got {d_197}"
            );
            assert!(
                d_197 < d_50,
                "bits={bits}: exact match {d_197} must beat out-of-range neighbor {d_50}"
            );
        }
    }

    #[test]
    fn test_benoquant_adc_cosine() {
        let dim = 16;
        let mut rng_state: u64 = 99;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rand_f32() * 5.0).collect())
            .collect();
        let enc = BenoQuantEncoder::train(&vectors, 8, VectorMetric::Cosine).unwrap();
        let q = vectors[7].clone();
        let prep = enc.prepare_query(&q, VectorMetric::Cosine);
        let self_dist = prep.distance(&enc.encode(&q));
        assert!(self_dist < 0.02, "self cosine distance got {self_dist}");

        let opp: Vec<f32> = q.iter().map(|x| -x).collect();
        let opp_dist = prep.distance(&enc.encode(&opp));
        assert!(
            opp_dist > 1.8,
            "opposite cosine distance should be ~2.0, got {opp_dist}"
        );
    }

    #[test]
    fn test_scan_l2_u8_codes_matches_distance() {
        for dim in [12usize, 16, 128] {
            let mut rng_state: u64 = 12345 + dim as u64;
            let mut rand_f32 = || {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u32 << 31) as f32) * 10.0 - 5.0
            };
            let vectors: Vec<Vec<f32>> = (0..64)
                .map(|_| (0..dim).map(|_| rand_f32()).collect())
                .collect();
            let enc = BenoQuantEncoder::train(&vectors, 8, VectorMetric::L2).unwrap();
            let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| enc.encode(v)).collect();
            let code_len = encoded[0].len();
            let mut flat = Vec::with_capacity(encoded.len() * code_len);
            for c in &encoded {
                flat.extend_from_slice(c);
            }

            let q = vectors[13].clone();
            let prep = enc.prepare_query(&q, VectorMetric::L2);
            assert!(prep.is_fast_l2_u8());

            let mut scanned = Vec::new();
            prep.scan_l2_u8_codes(&flat, code_len, f32::INFINITY, |id, dist| {
                scanned.push((id, dist));
                f32::INFINITY
            });
            assert_eq!(scanned.len(), vectors.len());
            for (id, dist) in scanned {
                let ref_dist = prep.distance(&encoded[id]);
                assert!(
                    (dist - ref_dist).abs() < 1e-3,
                    "dim={dim} id={id}: batch scan {dist} != ref {ref_dist}"
                );
            }
        }
    }

    #[test]
    fn test_scan_codes_tq4_and_cosine_match_distance() {
        for dim in [12usize, 16, 128] {
            let mut rng_state: u64 = 54321 + dim as u64;
            let mut rand_f32 = || {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u32 << 31) as f32) * 10.0 - 5.0
            };
            let vectors: Vec<Vec<f32>> = (0..48)
                .map(|_| (0..dim).map(|_| rand_f32()).collect())
                .collect();
            let q = vectors[9].clone();

            for (bits, metric) in [
                (4usize, VectorMetric::L2),
                (4usize, VectorMetric::Cosine),
                (8usize, VectorMetric::Cosine),
            ] {
                let enc = BenoQuantEncoder::train(&vectors, bits, metric).unwrap();
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
                let n = prep.rotated_query.len();
                let packed_len = n.div_ceil(2);
                for (id, dist) in scanned {
                    let code = &encoded[id];
                    let ref_dist = prep.distance(code);
                    // Independent scalar reference (never shares SIMD codegen with
                    // `distance()` or `scan_codes()`).
                    let scalar_ref = if bits == 4 {
                        let (nb, alpha) = split_u4_code(code, packed_len, enc.dim);
                        let step = alpha * prep.inv_scale;
                        let base = alpha * enc.offset;
                        if metric == VectorMetric::Cosine {
                            let (mut dot, mut v2, mut idx) = (0.0f32, 0.0f32, 0usize);
                            for &b in nb {
                                if idx >= n {
                                    break;
                                }
                                let lo = ((b & 0x0F) as f32) * step + base;
                                dot += prep.rotated_query[idx] * lo;
                                v2 += lo * lo;
                                idx += 1;
                                if idx < n {
                                    let hi = (((b >> 4) & 0x0F) as f32) * step + base;
                                    dot += prep.rotated_query[idx] * hi;
                                    v2 += hi * hi;
                                    idx += 1;
                                }
                            }
                            let denom = (prep.q_norm_sq * v2).sqrt();
                            if denom > 1e-12 {
                                (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                            } else {
                                1.0
                            }
                        } else {
                            let (mut sq, mut idx) = (0.0f32, 0usize);
                            for &b in nb {
                                if idx >= n {
                                    break;
                                }
                                let lo = ((b & 0x0F) as f32) * step + base;
                                let d0 = prep.rotated_query[idx] - lo;
                                sq += d0 * d0;
                                idx += 1;
                                if idx < n {
                                    let hi = (((b >> 4) & 0x0F) as f32) * step + base;
                                    let d1 = prep.rotated_query[idx] - hi;
                                    sq += d1 * d1;
                                    idx += 1;
                                }
                            }
                            (sq * prep.inv_n).max(0.0).sqrt()
                        }
                    } else {
                        let count = n.min(code.len());
                        let (mut dot, mut v2) = (0.0f32, 0.0f32);
                        for (&rq, &e) in prep.rotated_query[..count].iter().zip(&code[..count]) {
                            let dec = (e as f32) * prep.inv_scale + enc.offset;
                            dot += rq * dec;
                            v2 += dec * dec;
                        }
                        let denom = (prep.q_norm_sq * v2).sqrt();
                        if denom > 1e-12 {
                            (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                        } else {
                            1.0
                        }
                    };
                    assert!(
                        (dist - ref_dist).abs() < 1e-3 && (dist - scalar_ref).abs() < 1e-3,
                        "bits={bits} metric={metric:?} dim={dim} id={id}: scan={dist}, dist={ref_dist}, scalar={scalar_ref}"
                    );
                }
            }
        }
    }

    /// Every width must agree with the scalar reference: the 4-accumulator main
    /// loop (32 dims), the single 8-lane step, and the scalar tail are three
    /// separate code paths, and an off-by-one in any of them silently changes
    /// distances without changing the candidate count.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_l2_sq_u8_kernel_covers_every_width() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let q: Vec<f32> = (0..72).map(|_| (next() % 256) as f32 - 128.0).collect();
        let c: Vec<u8> = (0..72).map(|_| (next() % 256) as u8).collect();

        for len in 1..=72usize {
            let expected = l2_sq_f32_u8_scalar(&q[..len], &c[..len]);
            // SAFETY: `avx2` and `fma` were detected above.
            let actual = unsafe { l2_sq_f32_u8_avx2(&q[..len], &c[..len]) };
            let tol = expected.abs() * 1e-5 + 1e-2;
            assert!(
                (actual - expected).abs() <= tol,
                "len={len}: avx2 {actual} != scalar {expected}"
            );
        }
    }

    /// The pre-change u8 L2 kernel, kept as the baseline arm of the timing
    /// diagnostic: two FMA accumulators, 16 dimensions per iteration, and a
    /// store-then-scalar-sum horizontal reduce through the stack.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn l2_sq_f32_u8_avx2_two_chain(q_scaled: &[f32], code: &[u8]) -> f32 {
        use std::arch::x86_64::*;
        let len = q_scaled.len().min(code.len());
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0usize;
        let q_ptr = q_scaled.as_ptr();
        let c_ptr = code.as_ptr();

        while i + 16 <= len {
            let c0 = _mm_loadl_epi64(c_ptr.add(i) as *const __m128i);
            let c0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c0));
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i)), c0_f32);
            sum0 = _mm256_fmadd_ps(d0, d0, sum0);

            let c1 = _mm_loadl_epi64(c_ptr.add(i + 8) as *const __m128i);
            let c1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c1));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(q_ptr.add(i + 8)), c1_f32);
            sum1 = _mm256_fmadd_ps(d1, d1, sum1);

            i += 16;
        }

        let sum = _mm256_add_ps(sum0, sum1);
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
        let mut total: f32 = lanes.iter().sum();

        while i < len {
            let d = *q_ptr.add(i) - (*c_ptr.add(i) as f32);
            total += d * d;
            i += 1;
        }
        total
    }

    /// Timing diagnostic for the u8 L2 scan kernel. All arms evaluate the same
    /// kernel over the same codes with the same squared-space prune, so the
    /// differences isolate two separate questions:
    ///
    /// * two-chain vs four-chain = what the accumulator unroll bought.
    /// * loop-inside vs four-chain = what moving the candidate loop across the
    ///   `#[target_feature]` boundary bought (inlining the kernel into it).
    ///
    /// ```text
    /// cargo test --release --lib bench_scan_l2_u8_codes -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing diagnostic"]
    #[cfg(target_arch = "x86_64")]
    fn bench_scan_l2_u8_codes_throughput() {
        use std::time::Instant;
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            println!("no avx2+fma; skipping");
            return;
        }
        // Below any achievable distance, so the squared-space compare prunes
        // every candidate and the measurement isolates the kernel itself.
        const PRUNE_BELOW_EVERYTHING: f32 = 1e-6;
        let candidates = 20_000usize;
        let reps = 20usize;

        for dim in [128usize, 384, 768, 1024] {
            let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let q: Vec<f32> = (0..dim).map(|_| (next() % 256) as f32 - 128.0).collect();
            let codes: Vec<u8> = (0..dim * candidates)
                .map(|_| (next() % 256) as u8)
                .collect();

            // Arm A: loop inside the feature-tagged callee, kernel inlined.
            let mut sink_a = 0.0f32;
            let start = Instant::now();
            for _ in 0..reps {
                // SAFETY: `avx2` and `fma` were detected above.
                unsafe {
                    scan_l2_u8_codes_avx2(
                        &q,
                        &codes,
                        dim,
                        dim,
                        1.0,
                        PRUNE_BELOW_EVERYTHING,
                        &mut |_, d| {
                            sink_a += d;
                            d
                        },
                    );
                }
            }
            let inlined = start.elapsed().as_secs_f64();

            // Arm B: the pre-change shape - loop outside, kernel called once per
            // candidate. The feature mismatch prevents inlining here, exactly as
            // it did before.
            let mut sink_b = 0.0f32;
            let raw_sq_threshold = raw_sq_threshold_for(PRUNE_BELOW_EVERYTHING, 1.0);
            let start = Instant::now();
            for _ in 0..reps {
                for code in codes.chunks_exact(dim) {
                    // SAFETY: `avx2` and `fma` were detected above.
                    let raw_sq = unsafe { l2_sq_f32_u8_avx2(&q, code) };
                    if raw_sq <= raw_sq_threshold {
                        sink_b += raw_sq;
                    }
                }
            }
            let called = start.elapsed().as_secs_f64();

            // Arm C: the pre-change kernel - two FMA chains, 16 dims per
            // iteration, horizontal sum through the stack.
            let mut sink_c = 0.0f32;
            let start = Instant::now();
            for _ in 0..reps {
                for code in codes.chunks_exact(dim) {
                    // SAFETY: `avx2` and `fma` were detected above.
                    let raw_sq = unsafe { l2_sq_f32_u8_avx2_two_chain(&q, code) };
                    if raw_sq <= raw_sq_threshold {
                        sink_c += raw_sq;
                    }
                }
            }
            let two_chain = start.elapsed().as_secs_f64();

            let per_vector = (candidates * reps) as f64;
            let dims = (dim as f64) * per_vector;
            println!(
                "dim={dim:4}  two-chain {:7.2} ns/vec  four-chain {:7.2} ns/vec  x{:.2}  |  loop-inside {:7.2} ns/vec  x{:.2}  ({:.2} dims/ns)",
                two_chain / per_vector * 1e9,
                called / per_vector * 1e9,
                two_chain / called,
                inlined / per_vector * 1e9,
                called / inlined,
                dims / (called * 1e9),
            );
            assert_eq!(sink_a, 0.0);
            assert_eq!(sink_b, 0.0);
            assert_eq!(sink_c, 0.0);
        }
    }
}
