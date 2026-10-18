#![allow(unsafe_op_in_unsafe_fn)]

//! AVX2 IQ kernels.

use crate::kernel::Q38Q8KBlock;
use crate::kernel::Q38_Q8_K_TILE;
use crate::kernel::iq::{
    IQ1_DELTA, IQ1S_REPACK_BLOCK_BYTES, f16_at, iq4_nl_group, repack_base_d, repack_grid_index,
    repack_signed_scale, u16_at, u32_at,
};
use crate::kernel::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS64, KVALUES_IQ4NL,
};
use std::arch::x86_64::*;

#[inline]
pub(crate) fn dot_iq1_s_repacked_q8_k(weights: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq1_s_repacked_avx2(weights, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq4_xs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq4_xs_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq4_nl_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq4_nl_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq3_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq3_s_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq1_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq1_s_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq1_m_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq1_m_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq2_xxs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq2_xxs_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq2_xs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq2_xs_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq2_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq2_s_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_iq3_xxs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_iq3_xxs_q8_k_avx2(data, q, blocks) }
}

// ---------------------------------------------------------------------------
// AVX2 integer kernels
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32x8(values: __m256) -> f32 {
    let mut sum = _mm_add_ps(
        _mm256_extractf128_ps(values, 1),
        _mm256_castps256_ps128(values),
    );
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

/// `sum(|left| * (sign(left) * right)) * scale` over 32 signed bytes.
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_scaled_32(left: __m256i, right: __m256i, scale: i32) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(scale as i16))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq1_s_repacked_avx2(weights: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    let mut accumulated_correction = 0.0f32;
    for block in 0..blocks {
        let weight = &weights[block * IQ1S_REPACK_BLOCK_BYTES..][..IQ1S_REPACK_BLOCK_BYTES];
        let activation = &q[block];
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut correction = 0i32;
        for group in (0..8).step_by(2) {
            let mut grid = [0i64; 8];
            for lane in 0..8 {
                let at = group * 4 + lane;
                grid[lane] = IQ1S_GRID[repack_grid_index(weight, at)] as i64;
            }
            let quant0 = _mm256_set_epi64x(grid[3], grid[2], grid[1], grid[0]);
            let quant1 = _mm256_set_epi64x(grid[7], grid[6], grid[5], grid[4]);
            let activation0 =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let activation1 = _mm256_loadu_si256(
                activation.quants[(group + 1) * 32..].as_ptr() as *const __m256i
            );
            let signed0 = repack_signed_scale(weight, group);
            let signed1 = repack_signed_scale(weight, group + 1);
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant0, activation0, signed0.abs()),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant1, activation1, signed1.abs()),
            );
            correction += signed0
                * (activation.sums[group * 2] as i32 + activation.sums[group * 2 + 1] as i32);
            correction += signed1
                * (activation.sums[(group + 1) * 2] as i32
                    + activation.sums[(group + 1) * 2 + 1] as i32);
        }
        let d = repack_base_d(weight) * activation.scale;
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
        accumulated_correction += d * correction as f32;
    }
    sum_f32x8(accumulated) + IQ1_DELTA * accumulated_correction
}

/// `sum(left * right)` over 32 signed bytes, widened to 8 × `i32`.
///
/// `vpmaddubsw` needs one unsigned operand, so the classic sign trick is used:
/// negate `right` where `left` is negative and feed `|left|` instead.
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_32(left: __m256i, right: __m256i) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

/// `0xff` in every lane whose `signs` bit is set, `0x00` otherwise.
///
/// `signs` holds four bytes; byte `n` covers lanes `[8n, 8n+8)` and its bits
/// cover those lanes in order.
#[target_feature(enable = "avx2")]
unsafe fn raw_iq_sign_mask(signs: &[u8]) -> __m256i {
    const SIGN_SOURCE: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    const SIGN_BIT: [u8; 32] = [
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1,
        2, 4, 8, 16, 32, 64, 128,
    ];
    let packed = _mm256_set1_epi32(u32_at(signs) as i32);
    let source = _mm256_loadu_si256(SIGN_SOURCE.as_ptr() as *const __m256i);
    let bits = _mm256_loadu_si256(SIGN_BIT.as_ptr() as *const __m256i);
    let selected = _mm256_and_si256(_mm256_shuffle_epi8(packed, source), bits);
    _mm256_cmpeq_epi8(selected, bits)
}

/// Negate `activation` in the lanes selected by `signs`.
#[target_feature(enable = "avx2")]
unsafe fn apply_raw_iq_signs(activation: __m256i, signs: &[u8]) -> __m256i {
    let negative = raw_iq_sign_mask(signs);
    _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative)
}

/// Expand two consecutive IQ3_S groups (16 codes) into their 3-bit grid vectors.
///
/// `codes` holds the low 8 bits of the 16 grid indices and `high` supplies bit
/// 8 for each of the two groups, matching the scalar IQ3_S index build.
#[target_feature(enable = "avx2")]
unsafe fn iq3_s_grid_vectors(codes: &[u8], high: &[u8], output: &mut [__m256i; 2]) {
    let packed = _mm_loadu_si128(codes.as_ptr() as *const __m128i);
    let shifts = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    let high_mask = _mm256_set1_epi32(0x100);
    let low0 = _mm256_cvtepu8_epi32(packed);
    let low1 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(packed));
    let high0 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[0] as i32), shifts),
        high_mask,
    );
    let high1 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[1] as i32), shifts),
        high_mask,
    );
    let mut indices = [0u32; 16];
    _mm256_storeu_si256(
        indices.as_mut_ptr() as *mut __m256i,
        _mm256_or_si256(low0, high0),
    );
    _mm256_storeu_si256(
        indices.as_mut_ptr().add(8) as *mut __m256i,
        _mm256_or_si256(low1, high1),
    );
    let mut grids = [0i32; 16];
    for (slot, index) in grids.iter_mut().zip(indices.iter()) {
        *slot = IQ3S_GRID[*index as usize] as i32;
    }
    output[0] = _mm256_loadu_si256(grids.as_ptr() as *const __m256i);
    output[1] = _mm256_loadu_si256(grids.as_ptr().add(8) as *const __m256i);
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq4_xs_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nibble_mask = _mm_set1_epi8(15);
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 136..(block + 1) * 136];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let scale_high = u16_at(&weights[2..]);
        let scale_low = &weights[4..8];
        let quants = &weights[8..136];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in 0..8 {
            let scale = ((scale_low[group / 2] >> (4 * (group % 2))) & 15) as i32
                | (((scale_high >> (2 * group)) & 3) as i32) << 4;
            let packed = _mm_loadu_si128(quants[group * 16..].as_ptr() as *const __m128i);
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble_mask));
            let high = _mm_shuffle_epi8(
                table,
                _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask),
            );
            let quant = _mm256_set_m128i(high, low);
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let dot = products_s8_s8_32(quant, activation_group);
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale - 32)),
            );
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

/// Fused IQ4_XS: one weight decode, `LANES` activations (`LANES` is a const
/// generic so the lane accumulators stay in registers; see the Q4_K kernel).
///
/// C routes IQ4_XS through `Q38_DEFINE_IQ_TILE`, which just calls the
/// single-token kernel once per token. IQ4_XS is 29% of this checkpoint's
/// bytes, so it gets a real fused kernel here instead.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq4_xs_q8_k_tile_avx2<const LANES: usize>(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
    out: &mut [f32; Q38_Q8_K_TILE],
) {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nibble_mask = _mm_set1_epi8(15);
    let mut accumulated = [_mm256_setzero_ps(); LANES];
    for block in 0..blocks {
        let weights = &data[block * 136..(block + 1) * 136];
        let base_d = f16_at(weights);
        let scale_high = u16_at(&weights[2..]);
        let scale_low = &weights[4..8];
        let quants = &weights[8..136];
        let mut weighted = [_mm256_setzero_si256(); LANES];
        for group in 0..8 {
            let scale = ((scale_low[group / 2] >> (4 * (group % 2))) & 15) as i32
                | (((scale_high >> (2 * group)) & 3) as i32) << 4;
            let packed = _mm_loadu_si128(quants[group * 16..].as_ptr() as *const __m128i);
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble_mask));
            let high = _mm_shuffle_epi8(
                table,
                _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask),
            );
            let quant = _mm256_set_m128i(high, low);
            let scale = _mm256_set1_epi32(scale - 32);
            for lane in 0..LANES {
                let activation = &q[lane * blocks + block];
                let values =
                    _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
                let dot = products_s8_s8_32(quant, values);
                weighted[lane] =
                    _mm256_add_epi32(weighted[lane], _mm256_mullo_epi32(dot, scale));
            }
        }
        for lane in 0..LANES {
            let activation = &q[lane * blocks + block];
            let d = base_d * activation.scale;
            accumulated[lane] = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(weighted[lane]),
                accumulated[lane],
            );
        }
    }
    for lane in 0..LANES {
        out[lane] = sum_f32x8(accumulated[lane]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq4_nl_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nibble_mask = _mm_set1_epi8(15);
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 144..(block + 1) * 144];
        let activation = &q[block];
        // IQ4_NL carries a per-group scale, so each group folds into the
        // accumulator on its own; matches `q38_dot_iq4_nl_q8_k`.
        for group in 0..8 {
            let (base_d, quants) = iq4_nl_group(weights, group);
            let d = base_d * activation.scale;
            let packed = _mm_loadu_si128(quants.as_ptr() as *const __m128i);
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble_mask));
            let high = _mm_shuffle_epi8(
                table,
                _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask),
            );
            let quant = _mm256_set_m128i(high, low);
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let dot = products_s8_s8_32(quant, activation_group);
            accumulated = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(dot), accumulated);
        }
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq3_s_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 110..(block + 1) * 110];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let indices = &weights[2..66];
        let high = &weights[66..74];
        let signs = &weights[74..106];
        let scales = &weights[106..110];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in (0..8).step_by(2) {
            let mut quant = [_mm256_setzero_si256(); 2];
            iq3_s_grid_vectors(
                &indices[group * 8..group * 8 + 16],
                &high[group..group + 2],
                &mut quant,
            );
            for pair in 0..2 {
                let current = group + pair;
                let activation_group = _mm256_loadu_si256(
                    activation.quants[current * 32..].as_ptr() as *const __m256i,
                );
                let signed = apply_raw_iq_signs(activation_group, &signs[current * 4..]);
                let dot = products_u8_s8_32(quant[pair], signed);
                let scale = 1 + 2 * ((scales[group / 2] >> (4 * pair)) & 15) as i32;
                weighted_lanes = _mm256_add_epi32(
                    weighted_lanes,
                    _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)),
                );
            }
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

/// `sum(a * b)` over 32 unsigned bytes `a` and 32 signed bytes `b`.
#[target_feature(enable = "avx2")]
unsafe fn products_u8_s8_32(unsigned_values: __m256i, signed_values: __m256i) -> __m256i {
    let pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

/// `0xff` in every lane that the four 7-bit sign indices mark negative.
///
/// `ksigns64` packs one byte per sign bit, so the four indices together cover
/// the 32 lanes as four `i64` words.
#[target_feature(enable = "avx2")]
unsafe fn iq_sign_mask(s0: u32, s1: u32, s2: u32, s3: u32) -> __m256i {
    _mm256_set_epi64x(
        KSIGNS64[s3 as usize] as i64,
        KSIGNS64[s2 as usize] as i64,
        KSIGNS64[s1 as usize] as i64,
        KSIGNS64[s0 as usize] as i64,
    )
}

/// Negate `activation` in the lanes selected by four `ksigns64` indices.
#[target_feature(enable = "avx2")]
unsafe fn apply_iq_signs(activation: __m256i, s0: u32, s1: u32, s2: u32, s3: u32) -> __m256i {
    let negative = iq_sign_mask(s0, s1, s2, s3);
    _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative)
}

/// Expand one IQ1_S group quadruple into its 8-byte grid vector.
///
/// Mirrors the portable branch of `q38_iq1_s_grid32`: four low index bytes in
/// `indices`, three high bits each taken from `high`.
#[target_feature(enable = "avx2")]
unsafe fn iq1_s_grid32(indices: &[u8], high: u16) -> __m256i {
    let high = high as u32;
    let at = |low: u8, bits: u32| IQ1S_GRID[(low as u32 | bits) as usize] as i64;
    _mm256_set_epi64x(
        at(indices[3], (high >> 1) & 0x700),
        at(indices[2], (high << 2) & 0x700),
        at(indices[1], (high << 5) & 0x700),
        at(indices[0], (high << 8) & 0x700),
    )
}

/// Horizontal sum of the eight `i32` lanes.
#[target_feature(enable = "avx2")]
unsafe fn sum_i32_8(values: __m256i) -> i32 {
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, values);
    lanes.iter().sum()
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq1_s_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    let mut accumulated_correction = 0.0f32;
    for block in 0..blocks {
        let w = &data[block * 50..(block + 1) * 50];
        let activation = &q[block];
        let d = f16_at(w) * activation.scale;
        let indices = &w[2..34];
        let high = &w[34..50];
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut correction = 0i32;
        for group in (0..8).step_by(2) {
            let high0 = u16_at(&high[group * 2..]);
            let high1 = u16_at(&high[(group + 1) * 2..]);
            let quant0 = iq1_s_grid32(&indices[group * 4..], high0);
            let quant1 = iq1_s_grid32(&indices[(group + 1) * 4..], high1);
            let activation0 =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let activation1 = _mm256_loadu_si256(
                activation.quants[(group + 1) * 32..].as_ptr() as *const __m256i,
            );
            let scale0 = 1 + 2 * ((high0 >> 12) & 7) as i32;
            let scale1 = 1 + 2 * ((high1 >> 12) & 7) as i32;
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant0, activation0, scale0),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant1, activation1, scale1),
            );
            correction += scale0
                * if high0 & 0x8000 != 0 { -1 } else { 1 }
                * (activation.sums[group * 2] as i32 + activation.sums[group * 2 + 1] as i32);
            correction += scale1
                * if high1 & 0x8000 != 0 { -1 } else { 1 }
                * (activation.sums[(group + 1) * 2] as i32
                    + activation.sums[(group + 1) * 2 + 1] as i32);
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
        accumulated_correction += d * correction as f32;
    }
    sum_f32x8(accumulated) + IQ1_DELTA * accumulated_correction
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq1_m_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    /// `0x01` in every byte, i.e. the "+1" delta that leaves activations as-is.
    const PLUS_ONES: i64 = 0x0101_0101_0101_0101;
    let one = _mm256_set1_epi8(1);
    let mut total = 0.0f32;
    for block in 0..blocks {
        let w = &data[block * 56..(block + 1) * 56];
        let activation = &q[block];
        let indices = &w[0..32];
        let high = &w[32..48];
        let packed_scales = &w[48..56];
        let scale_bits = (u16_at(packed_scales) >> 12)
            | ((u16_at(&packed_scales[2..]) >> 8) & 0x00f0)
            | ((u16_at(&packed_scales[4..]) >> 4) & 0x0f00)
            | (u16_at(&packed_scales[6..]) & 0xf000);
        let d = crate::kernel::quant::f16_to_f32(scale_bits) * activation.scale;
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut correction_lanes = _mm256_setzero_si256();
        for group in (0..8).step_by(2) {
            let h = &high[group * 2..];
            let idx = &indices[group * 4..];
            let at = |low: u8, bits: u32| IQ1S_GRID[(low as u32 | bits) as usize] as i64;
            let quant0 = _mm256_set_epi64x(
                at(idx[3], ((h[1] as u32) << 4) & 0x700),
                at(idx[2], ((h[1] as u32) << 8) & 0x700),
                at(idx[1], ((h[0] as u32) << 4) & 0x700),
                at(idx[0], ((h[0] as u32) << 8) & 0x700),
            );
            let quant1 = _mm256_set_epi64x(
                at(idx[7], ((h[3] as u32) << 4) & 0x700),
                at(idx[6], ((h[3] as u32) << 8) & 0x700),
                at(idx[5], ((h[2] as u32) << 4) & 0x700),
                at(idx[4], ((h[2] as u32) << 8) & 0x700),
            );
            // A set delta bit flips the sign of the whole 8-lane word.
            let delta = |byte: u8, bit: u8| if byte & bit != 0 { -1i64 } else { PLUS_ONES };
            let delta0 = _mm256_set_epi64x(
                delta(h[1], 0x80),
                delta(h[1], 0x08),
                delta(h[0], 0x80),
                delta(h[0], 0x08),
            );
            let delta1 = _mm256_set_epi64x(
                delta(h[3], 0x80),
                delta(h[3], 0x08),
                delta(h[2], 0x80),
                delta(h[2], 0x08),
            );
            let activation0 =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let activation1 = _mm256_loadu_si256(
                activation.quants[(group + 1) * 32..].as_ptr() as *const __m256i,
            );
            let scales = u16_at(&packed_scales[(group / 2) * 2..]);
            let scale0 = 1 + 2 * ((scales >> 0) & 7) as i32;
            let scale1 = 1 + 2 * ((scales >> 3) & 7) as i32;
            let scale2 = 1 + 2 * ((scales >> 6) & 7) as i32;
            let scale3 = 1 + 2 * ((scales >> 9) & 7) as i32;
            let scales0 =
                _mm256_set_m128i(_mm_set1_epi16(scale1 as i16), _mm_set1_epi16(scale0 as i16));
            let scales1 =
                _mm256_set_m128i(_mm_set1_epi16(scale3 as i16), _mm_set1_epi16(scale2 as i16));
            let dot0 = _mm256_maddubs_epi16(
                _mm256_abs_epi8(quant0),
                _mm256_sign_epi8(activation0, quant0),
            );
            let dot1 = _mm256_maddubs_epi16(
                _mm256_abs_epi8(quant1),
                _mm256_sign_epi8(activation1, quant1),
            );
            let correction0 = _mm256_maddubs_epi16(one, _mm256_sign_epi8(activation0, delta0));
            let correction1 = _mm256_maddubs_epi16(one, _mm256_sign_epi8(activation1, delta1));
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_add_epi32(
                    _mm256_madd_epi16(dot0, scales0),
                    _mm256_madd_epi16(dot1, scales1),
                ),
            );
            correction_lanes = _mm256_add_epi32(
                correction_lanes,
                _mm256_add_epi32(
                    _mm256_madd_epi16(correction0, scales0),
                    _mm256_madd_epi16(correction1, scales1),
                ),
            );
        }
        let weighted = sum_i32_8(weighted_lanes);
        let correction = sum_i32_8(correction_lanes);
        total += d * (weighted as f32 + IQ1_DELTA * correction as f32);
    }
    total
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq2_xxs_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let w = &data[block * 66..(block + 1) * 66];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(w) * activation.scale;
        let codes = &w[2..66];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in 0..8 {
            let code = &codes[group * 8..group * 8 + 8];
            let metadata = u32_at(&code[4..]);
            let scale = 1 + 2 * (metadata >> 28) as i32;
            let quant = _mm256_set_epi64x(
                IQ2XXS_GRID[code[3] as usize] as i64,
                IQ2XXS_GRID[code[2] as usize] as i64,
                IQ2XXS_GRID[code[1] as usize] as i64,
                IQ2XXS_GRID[code[0] as usize] as i64,
            );
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let signed_activation = apply_iq_signs(
                activation_group,
                metadata & 127,
                (metadata >> 7) & 127,
                (metadata >> 14) & 127,
                (metadata >> 21) & 127,
            );
            let dot = products_u8_s8_32(quant, signed_activation);
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)),
            );
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq2_xs_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let w = &data[block * 74..(block + 1) * 74];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(w) * activation.scale;
        let codes = &w[2..66];
        let scales = &w[66..74];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in 0..8 {
            let code_bytes = &codes[group * 8..group * 8 + 8];
            let mut code = [0u16; 4];
            for (section, slot) in code.iter_mut().enumerate() {
                *slot = u16_at(&code_bytes[section * 2..]);
            }
            let scale0 = 1 + 2 * (scales[group] & 15) as i32;
            let scale1 = 1 + 2 * (scales[group] >> 4) as i32;
            let quant = _mm256_set_epi64x(
                IQ2XS_GRID[(code[3] & 511) as usize] as i64,
                IQ2XS_GRID[(code[2] & 511) as usize] as i64,
                IQ2XS_GRID[(code[1] & 511) as usize] as i64,
                IQ2XS_GRID[(code[0] & 511) as usize] as i64,
            );
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let signed_activation = apply_iq_signs(
                activation_group,
                (code[0] >> 9) as u32,
                (code[1] >> 9) as u32,
                (code[2] >> 9) as u32,
                (code[3] >> 9) as u32,
            );
            let dot = products_u8_s8_32(quant, signed_activation);
            let scale = _mm256_set_m128i(_mm_set1_epi32(scale1), _mm_set1_epi32(scale0));
            weighted_lanes = _mm256_add_epi32(weighted_lanes, _mm256_mullo_epi32(dot, scale));
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq2_s_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let scale_mask = _mm_set1_epi8(15);
    let scale_one = _mm_set1_epi8(1);
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let w = &data[block * 82..(block + 1) * 82];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(w) * activation.scale;
        let indices = &w[2..34];
        let signs = &w[34..66];
        let high = &w[66..74];
        let scales = &w[74..82];
        // One byte per group holding `1 + 2 * (low nibble)`, then the same for
        // the high nibbles. Widening to u16 keeps `scale0` in the low 128-bit
        // lane and `scale1` in the high one, which is what the shuffle below
        // broadcasts per section pair.
        let packed_scales = u64::from_le_bytes(scales.try_into().unwrap());
        let mut scale_bytes =
            _mm_set_epi64x((packed_scales >> 4) as i64, packed_scales as i64);
        scale_bytes = _mm_add_epi8(
            _mm_slli_epi16::<1>(_mm_and_si128(scale_bytes, scale_mask)),
            scale_one,
        );
        let scale_words = _mm256_cvtepu8_epi16(scale_bytes);
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in 0..8 {
            let code = &indices[group * 4..group * 4 + 4];
            let high_bits = high[group] as u32;
            let grid_at = |low: u8, shift: u32| -> i64 {
                IQ2S_GRID[(low as u32 | ((high_bits << shift) & 0x300)) as usize] as i64
            };
            let quant = _mm256_set_epi64x(
                grid_at(code[3], 2),
                grid_at(code[2], 4),
                grid_at(code[1], 6),
                grid_at(code[0], 8),
            );
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let signed_activation = apply_raw_iq_signs(activation_group, &signs[group * 4..]);
            let dot16 = _mm256_maddubs_epi16(quant, signed_activation);
            let control = ((2 * group) | ((2 * group + 1) << 8)) as i16;
            let scale = _mm256_shuffle_epi8(scale_words, _mm256_set1_epi16(control));
            weighted_lanes = _mm256_add_epi32(weighted_lanes, _mm256_madd_epi16(dot16, scale));
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq3_xxs_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let w = &data[block * 98..(block + 1) * 98];
        let activation = &q[block];
        let d = 0.25 * f16_at(w) * activation.scale;
        let indices = &w[2..66];
        let metadata = &w[66..98];
        let mut weighted_lanes = [_mm256_setzero_si256(); 2];
        for group in 0..8 {
            let packed = u32_at(&metadata[group * 4..]);
            let scale = 1 + 2 * (packed >> 28) as i32;
            let code = &indices[group * 8..group * 8 + 8];
            let quant = _mm256_set_epi32(
                IQ3XXS_GRID[code[7] as usize] as i32,
                IQ3XXS_GRID[code[6] as usize] as i32,
                IQ3XXS_GRID[code[5] as usize] as i32,
                IQ3XXS_GRID[code[4] as usize] as i32,
                IQ3XXS_GRID[code[3] as usize] as i32,
                IQ3XXS_GRID[code[2] as usize] as i32,
                IQ3XXS_GRID[code[1] as usize] as i32,
                IQ3XXS_GRID[code[0] as usize] as i32,
            );
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let signed_activation = apply_iq_signs(
                activation_group,
                packed & 127,
                (packed >> 7) & 127,
                (packed >> 14) & 127,
                (packed >> 21) & 127,
            );
            let dot = products_u8_s8_32(quant, signed_activation);
            weighted_lanes[group & 1] = _mm256_add_epi32(
                weighted_lanes[group & 1],
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)),
            );
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(weighted_lanes[0], weighted_lanes[1])),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Q38_Q8_K_BLOCK_SIZE;
    use crate::gguf::{GgmlType, Gguf};
    use crate::kernel::iq::{
        dot_iq1_m_q8_k, dot_iq1_s_q8_k, dot_iq2_s_q8_k, dot_iq2_xxs_q8_k, dot_iq2_xs_q8_k,
        dot_iq3_s_q8_k, dot_iq3_xxs_q8_k, dot_iq4_nl_q8_k, dot_iq4_xs_q8_k,
    };
    use crate::kernel::tensor::{dot_q2_k_q8_k_scalar, row_bytes};
    use crate::kernel::x86::tensor::dot_q2_k_q8_k_avx2;
    use crate::kernel::quant::quantize_q8_k;

    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn byte(&mut self) -> u8 {
            (self.next() >> 13) as u8
        }
    }

    fn avx2_available() -> bool {
        crate::kernel::avx2_fma()
    }

    /// Random bytes with a finite f16 written at each of `scale_offsets`.
    fn synthetic_blocks_at(
        blocks: usize,
        block_bytes: usize,
        seed: u32,
        scale_offsets: &[usize],
    ) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let at = block * block_bytes;
            for i in 0..block_bytes {
                data[at + i] = rng.byte();
            }
            for &offset in scale_offsets {
                let scale = 0x3800u16 | (rng.next() as u16 & 0x03ff);
                data[at + offset..at + offset + 2].copy_from_slice(&scale.to_le_bytes());
            }
        }
        data
    }

    fn synthetic_blocks(blocks: usize, block_bytes: usize, seed: u32) -> Vec<u8> {
        synthetic_blocks_at(blocks, block_bytes, seed, &[0])
    }

    /// IQ1_M assembles its scale from four nibbles of `packed_scales`, so those
    /// are pinned to a finite f16 while the rest of the block stays random.
    fn synthetic_iq1_m(blocks: usize, seed: u32) -> Vec<u8> {
        const BLOCK: usize = 56;
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * BLOCK];
        for block in 0..blocks {
            let at = block * BLOCK;
            for i in 0..BLOCK {
                data[at + i] = rng.byte();
            }
            // `packed_scales` sits at offset 48; each of its byte pairs feeds one
            // nibble of `scale_bits`, taken from the second byte of the pair.
            let low = rng.byte();
            data[at + 49] = ((low & 0x0f) << 4) | (data[at + 49] & 0x0f);
            data[at + 51] = (((low >> 4) & 0x0f) << 4) | (data[at + 51] & 0x0f);
            data[at + 53] = 0xc0 | (data[at + 53] & 0x0f);
            data[at + 55] = 0x30 | (data[at + 55] & 0x0f);
        }
        data
    }

    fn synthetic_iq4_nl(blocks: usize, seed: u32) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * 144];
        for block in 0..blocks {
            for group in 0..8 {
                let at = block * 144 + group * 18;
                for i in 0..18 {
                    data[at + i] = rng.byte();
                }
                let scale = 0x3800u16 | (rng.next() as u16 & 0x03ff);
                data[at..at + 2].copy_from_slice(&scale.to_le_bytes());
            }
        }
        data
    }

    fn activation_blocks(blocks: usize, seed: u32) -> Vec<Q38Q8KBlock> {
        let width = blocks * Q38_Q8_K_BLOCK_SIZE;
        let mut rng = Rng(seed);
        let values = (0..width)
            .map(|_| (rng.next() as f32 / u32::MAX as f32) * 4.0 - 2.0)
            .collect::<Vec<_>>();
        let mut activation = (0..blocks)
            .map(|_| Q38Q8KBlock::default())
            .collect::<Vec<_>>();
        quantize_q8_k(&mut activation, &values).unwrap();
        activation
    }

    /// Every IQ `q8_k` kernel, AVX2 against its scalar reference.
    ///
    /// The weights are synthetic random bytes (only the scale fields pinned
    /// finite), which hits more grid indices, sign masks and nibble
    /// combinations than any single real tensor would.
    #[test]
    fn avx2_matches_scalar_iq_kernels() {
        if !avx2_available() {
            println!("AVX2 + FMA unavailable; skipping");
            return;
        }
        let blocks = 16usize;
        for seed in [0x2545_f491u32, 0x9e37_79b9, 0x0123_4567] {
            let activation = activation_blocks(blocks, seed ^ 0xdead_beef);
            let reldiff = |a: f32, b: f32| (a - b).abs() / b.abs().max(1.0);
            let check = |name: &str, scalar: f32, vector: f32| {
                let error = reldiff(vector, scalar);
                println!("seed={seed:#x}: {name} rel={error:.3e}");
                assert!(
                    error < 1e-5,
                    "{name} seed {seed:#x}: avx2 {vector} vs scalar {scalar}"
                );
            };

            let xs = synthetic_blocks(blocks, 136, seed);
            check(
                "IQ4_XS",
                dot_iq4_xs_q8_k(&xs, &activation, blocks),
                unsafe { dot_iq4_xs_q8_k_avx2(&xs, &activation, blocks) },
            );

            let nl = synthetic_iq4_nl(blocks, seed);
            check(
                "IQ4_NL",
                dot_iq4_nl_q8_k(&nl, &activation, blocks),
                unsafe { dot_iq4_nl_q8_k_avx2(&nl, &activation, blocks) },
            );

            let iq3 = synthetic_blocks(blocks, 110, seed);
            check(
                "IQ3_S",
                dot_iq3_s_q8_k(&iq3, &activation, blocks),
                unsafe { dot_iq3_s_q8_k_avx2(&iq3, &activation, blocks) },
            );

            let iq1s = synthetic_blocks(blocks, 50, seed);
            check(
                "IQ1_S",
                dot_iq1_s_q8_k(&iq1s, &activation, blocks),
                unsafe { dot_iq1_s_q8_k_avx2(&iq1s, &activation, blocks) },
            );

            let iq1m = synthetic_iq1_m(blocks, seed);
            check(
                "IQ1_M",
                dot_iq1_m_q8_k(&iq1m, &activation, blocks),
                unsafe { dot_iq1_m_q8_k_avx2(&iq1m, &activation, blocks) },
            );

            let iq2xxs = synthetic_blocks(blocks, 66, seed);
            check(
                "IQ2_XXS",
                dot_iq2_xxs_q8_k(&iq2xxs, &activation, blocks),
                unsafe { dot_iq2_xxs_q8_k_avx2(&iq2xxs, &activation, blocks) },
            );

            let iq2xs = synthetic_blocks(blocks, 74, seed);
            check(
                "IQ2_XS",
                dot_iq2_xs_q8_k(&iq2xs, &activation, blocks),
                unsafe { dot_iq2_xs_q8_k_avx2(&iq2xs, &activation, blocks) },
            );

            let iq2s = synthetic_blocks(blocks, 82, seed);
            check(
                "IQ2_S",
                dot_iq2_s_q8_k(&iq2s, &activation, blocks),
                unsafe { dot_iq2_s_q8_k_avx2(&iq2s, &activation, blocks) },
            );

            let iq3xxs = synthetic_blocks(blocks, 98, seed);
            check(
                "IQ3_XXS",
                dot_iq3_xxs_q8_k(&iq3xxs, &activation, blocks),
                unsafe { dot_iq3_xxs_q8_k_avx2(&iq3xxs, &activation, blocks) },
            );
        }
    }

    /// The synthetic test above covers every legal byte pattern; this one runs
    /// the same comparison on real weights.
    ///
    /// `Qwen3.8-27B-UD-IQ1_M.gguf` is the only file that exercises these
    /// kernels (IQ1_S 163 tensors, IQ2_XXS 110, IQ1_M 35, IQ2_S/IQ2_XS 25,
    /// IQ3_XXS 21, Q2_K 8). Skipped when the model is not on disk.
    #[test]
    fn avx2_matches_scalar_real_iq_tensors() {
        if !avx2_available() {
            println!("AVX2 + FMA unavailable; skipping");
            return;
        }
        type Scalar = fn(&[u8], &[Q38Q8KBlock], usize) -> f32;
        type Vector = unsafe fn(&[u8], &[Q38Q8KBlock], usize) -> f32;
        fn kernels(ty: GgmlType) -> Option<(Scalar, Vector)> {
            Some(match ty {
                GgmlType::Q2K => (dot_q2_k_q8_k_scalar, dot_q2_k_q8_k_avx2),
                GgmlType::Iq1S => (dot_iq1_s_q8_k, dot_iq1_s_q8_k_avx2),
                GgmlType::Iq1M => (dot_iq1_m_q8_k, dot_iq1_m_q8_k_avx2),
                GgmlType::Iq2Xxs => (dot_iq2_xxs_q8_k, dot_iq2_xxs_q8_k_avx2),
                GgmlType::Iq2Xs => (dot_iq2_xs_q8_k, dot_iq2_xs_q8_k_avx2),
                GgmlType::Iq2S => (dot_iq2_s_q8_k, dot_iq2_s_q8_k_avx2),
                GgmlType::Iq3Xxs => (dot_iq3_xxs_q8_k, dot_iq3_xxs_q8_k_avx2),
                _ => return None,
            })
        }

        let Ok(gguf) = Gguf::open("Qwen3.8-27B-UD-IQ1_M.gguf") else {
            println!("IQ1_M model not present; skipping");
            return;
        };
        let mut seen: Vec<String> = Vec::new();
        let mut checked = 0usize;
        for tensor in gguf.tensors() {
            if tensor.n_dims != 2 {
                continue;
            }
            let Some((scalar_fn, vector_fn)) = kernels(tensor.ty) else {
                continue;
            };
            let key = format!("{:?}", tensor.ty);
            if seen.contains(&key) {
                continue;
            }
            let width = tensor.shape[0] as usize;
            if width == 0 || width % Q38_Q8_K_BLOCK_SIZE != 0 {
                continue;
            }
            let Some(rbytes) = row_bytes(tensor.ty, width) else {
                continue;
            };
            seen.push(key.clone());

            let rows = tensor.shape[1];
            let blocks = width / Q38_Q8_K_BLOCK_SIZE;
            let activation = activation_blocks(blocks, 0x2545_f491 ^ blocks as u32);
            for row in [0u64, rows - 1] {
                let at = row as usize * rbytes;
                let data = &tensor.data[at..at + rbytes];
                let scalar = scalar_fn(data, &activation, blocks);
                let vector = unsafe { vector_fn(data, &activation, blocks) };
                let rel = (vector - scalar).abs() / scalar.abs().max(1.0);
                println!("{key} row={row}: scalar={scalar} avx2={vector} rel={rel:.3e}");
                assert!(
                    rel < 1e-5,
                    "{key} row {row}: avx2 {vector} vs scalar {scalar}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 12, "only {checked} rows exercised");
    }
}
