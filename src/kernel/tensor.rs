#![allow(unsafe_op_in_unsafe_fn)]

use crate::gguf::{GgmlType, TensorEntry};
use crate::kernel::quant::{bf16_to_f32, f16_to_f32};
use crate::kernel::{Q38_Q8_K_BLOCK_SIZE, iq};
use crate::kernel::{
    Q38_Q8_K_TILE, Q38CoreError, Q38CoreResult, Q38Q8KBlock, Q38TensorOps, q38_parallel_over,
};
use rayon::prelude::*;
const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

#[inline]
pub(crate) fn read_u16(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

#[inline]
pub(crate) fn read_f32(data: &[u8]) -> f32 {
    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

pub(crate) fn scale_min_k4(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

pub(crate) fn unpack_q3_scales(packed: &[u8], scales: &mut [i8; 16]) {
    for index in 0..16 {
        let low = if index < 8 {
            packed[index] & 15
        } else {
            packed[index - 8] >> 4
        };
        let high = (packed[8 + index % 4] >> (2 * (index / 4))) & 3;
        scales[index] = ((low | (high << 4)) as i32 - 32) as i8;
    }
}

pub(crate) fn row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
    let block_elements = ty.block_elements() as usize;
    let block_bytes = ty.block_bytes() as usize;
    if block_elements == 0 || block_bytes == 0 || width % block_elements != 0 {
        return None;
    }
    Some(width / block_elements * block_bytes)
}

fn supported_f32_gemv(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::Q8_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    ) || iq::supports_f32(ty)
}

fn supported_q8_k(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    ) || iq::supports_q8_k(ty)
}

// ---- runtime dispatch ----

#[inline]
fn dot_f32(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_f32(data, input, n);
    }
    return crate::kernel::scalar::tensor::dot_f32_scalar(data, input, n);
}

#[inline]
fn dot_f16(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma_f16c() {
        return crate::kernel::x86::tensor::dot_f16(data, input, n);
    }
    return crate::kernel::scalar::tensor::dot_f16_scalar(data, input, n);
}

#[inline]
fn dot_q4_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_q4_k_q8_k(data, q, blocks);
    }
    return crate::kernel::scalar::tensor::dot_q4_k_q8_k_scalar(data, q, blocks);
}

#[inline]
fn dot_q6_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_q6_k_q8_k(data, q, blocks);
    }
    return crate::kernel::scalar::tensor::dot_q6_k_q8_k_scalar(data, q, blocks);
}

#[inline]
fn dot_q3_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_q3_k_q8_k(data, q, blocks);
    }
    return crate::kernel::scalar::tensor::dot_q3_k_q8_k_scalar(data, q, blocks);
}

#[inline]
fn dot_q5_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_q5_k_q8_k(data, q, blocks);
    }
    return crate::kernel::scalar::tensor::dot_q5_k_q8_k_scalar(data, q, blocks);
}

#[inline]
fn dot_q2_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::avx2_fma() {
        return crate::kernel::x86::tensor::dot_q2_k_q8_k(data, q, blocks);
    }
    return crate::kernel::scalar::tensor::dot_q2_k_q8_k_scalar(data, q, blocks);
}

#[inline]
fn dot_q8_0(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    // Q8_0 blocks are 32 values wide; anything else stays on the scalar path.
    if n % 32 == 0 && super::avx2_fma_f16c() {
        return crate::kernel::x86::tensor::dot_q8_0(data, input, n);
    }
    return crate::kernel::scalar::tensor::dot_q8_0_scalar(data, input, n);
}

fn dot_row(data: &[u8], input: &[f32], n: usize, ty: GgmlType) -> f32 {
    match ty {
        GgmlType::F32 => dot_f32(data, input, n),
        GgmlType::F16 => dot_f16(data, input, n),
        GgmlType::Bf16 => crate::kernel::scalar::tensor::dot_bf16_scalar(data, input, n),
        GgmlType::Q8_0 => dot_q8_0(data, input, n),
        GgmlType::Q2K => crate::kernel::scalar::tensor::dot_q2_k_scalar(data, input, n),
        GgmlType::Q3K => crate::kernel::scalar::tensor::dot_q3_k_scalar(data, input, n),
        GgmlType::Q4K => crate::kernel::scalar::tensor::dot_q4_k_scalar(data, input, n),
        GgmlType::Q5K => crate::kernel::scalar::tensor::dot_q5_k_scalar(data, input, n),
        GgmlType::Q6K => crate::kernel::scalar::tensor::dot_q6_k_scalar(data, input, n),
        ty if iq::supports_f32(ty) => iq::dot_row_f32(data, input, n, ty),
        _ => f32::NAN,
    }
}

fn dot_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize, ty: GgmlType) -> f32 {
    match ty {
        GgmlType::Q2K => dot_q2_k_q8_k(data, q, blocks),
        GgmlType::Q3K => dot_q3_k_q8_k(data, q, blocks),
        GgmlType::Q4K => dot_q4_k_q8_k(data, q, blocks),
        GgmlType::Q5K => dot_q5_k_q8_k(data, q, blocks),
        GgmlType::Q6K => dot_q6_k_q8_k(data, q, blocks),
        ty if iq::supports_q8_k(ty) => iq::dot_q8_k(data, q, blocks, ty),
        _ => f32::NAN,
    }
}

/// Fused counterpart of [`dot_q8_k`]: the same weight row against `LANES`
/// consecutive token slabs, laid out token-major
/// (`activations[lane * blocks + block]`).
fn dot_q8_k_tile<const LANES: usize>(
    data: &[u8],
    activations: &[Q38Q8KBlock],
    blocks: usize,
    ty: GgmlType,
    out: &mut [f32; Q38_Q8_K_TILE],
) {
    debug_assert!(LANES > 0 && LANES <= Q38_Q8_K_TILE);
    debug_assert!(activations.len() >= LANES * blocks && blocks > 0);
    #[cfg(target_arch = "x86_64")]
    if tile_kernels_available() {
        match ty {
            GgmlType::Q2K => {
                return unsafe {
                    crate::kernel::x86::tensor::dot_q2_k_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            GgmlType::Q3K => {
                return unsafe {
                    crate::kernel::x86::tensor::dot_q3_k_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            GgmlType::Q4K => {
                return unsafe {
                    crate::kernel::x86::tensor::dot_q4_k_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            GgmlType::Q5K => {
                return unsafe {
                    crate::kernel::x86::tensor::dot_q5_k_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            GgmlType::Q6K => {
                return unsafe {
                    crate::kernel::x86::tensor::dot_q6_k_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            GgmlType::Iq4Xs => {
                return unsafe {
                    crate::kernel::x86::iq::dot_iq4_xs_q8_k_tile_avx2::<LANES>(
                        data,
                        activations,
                        blocks,
                        out,
                    )
                };
            }
            _ => {}
        }
    }
    for lane in 0..LANES {
        out[lane] = dot_q8_k(
            data,
            &activations[lane * blocks..(lane + 1) * blocks],
            blocks,
            ty,
        );
    }
}

/// Tail entry point: the batch remainder (`batch % Q38_Q8_K_TILE`) is only
/// known at run time, so the lane count is dispatched here onto the const
/// monomorphisations instead of decoding the weights token by token.
fn dot_q8_k_tile_tail(
    data: &[u8],
    activations: &[Q38Q8KBlock],
    blocks: usize,
    lanes: usize,
    ty: GgmlType,
    out: &mut [f32; Q38_Q8_K_TILE],
) {
    match lanes {
        2 => dot_q8_k_tile::<2>(data, activations, blocks, ty, out),
        3 => dot_q8_k_tile::<3>(data, activations, blocks, ty, out),
        _ => {
            for lane in 0..lanes {
                out[lane] = dot_q8_k(
                    data,
                    &activations[lane * blocks..(lane + 1) * blocks],
                    blocks,
                    ty,
                );
            }
        }
    }
}

/// Fused kernels need AVX2, and `Q38_TILE=0` forces the per-token path so the
/// two can be compared on the same binary.
#[cfg(target_arch = "x86_64")]
fn tile_kernels_available() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("Q38_TILE") {
        Ok(value) => !matches!(value.as_str(), "0" | "off" | "false"),
        Err(_) => true,
    })
}

fn dequantize_quant_row(data: &[u8], output: &mut [f32], ty: GgmlType) {
    if iq::supports_f32(ty) {
        iq::dequantize_row(data, output, ty);
        return;
    }
    let stride = ty.block_elements() as usize;
    let block_bytes = ty.block_bytes() as usize;
    let mut basis = [0.0f32; BLOCK];
    for (block_index, block) in output.chunks_mut(stride).enumerate() {
        for i in 0..stride {
            basis[i] = 1.0;
            block[i] = dot_row(
                &data[block_index * block_bytes..(block_index + 1) * block_bytes],
                &basis[..stride],
                stride,
                ty,
            );
            basis[i] = 0.0;
        }
    }
}

impl TensorEntry<'_> {
    /// Repacked IQ1_S view of this tensor, when [`crate::kernel::Q38Iq1sRepack`]
    /// built one and it covers every row of every block.
    #[inline]
    fn repacked_iq1_s(&self, blocks: usize) -> Option<&[u8]> {
        if self.ty != GgmlType::Iq1S {
            return None;
        }
        let repacked = self.iq1_s_repack.as_deref()?;
        let rows = self.shape[1] as usize;
        if repacked.len() < rows * blocks * iq::IQ1S_REPACK_BLOCK_BYTES {
            return None;
        }
        Some(repacked)
    }
}

/// Runs `dot(row)` for every row, in order, and stores the result.
///
/// The row is the outer loop because the weight row is what is being streamed
/// out of the 16 GB mapping: visiting row `r` once and reusing it for the whole
/// batch is what keeps the matrix from being re-read per token. Splitting by
/// row rather than by token preserves that order, and the result is
/// bit-identical to the serial walk because each output element is still the
/// same `dot` call.
fn gemv_rows<F>(output: &mut [f32], rows: usize, per_row: usize, dot: F)
where
    F: Fn(usize) -> f32 + Sync,
{
    let output = &mut output[..rows];
    if q38_parallel_over(rows, per_row) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| *slot = dot(row));
    } else {
        for (row, slot) in output.iter_mut().enumerate() {
            *slot = dot(row);
        }
    }
}

/// Same as [`gemv_rows`], for the batched kernels whose output is token-major:
/// `output[token * rows + row]`.
///
/// Rust has no safe way to spell "column block of a token-major slice", so the
/// row loop writes through [`RowTargets`] rather than reordering the loop to
/// put the token outermost — that would re-read the whole matrix once per
/// token and undo the single-pass advantage.
fn gemm_rows<F>(output: &mut [f32], rows: usize, batch: usize, per_row: usize, dot: F)
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    if q38_parallel_over(rows, per_row) {
        let targets = RowTargets {
            base: output.as_mut_ptr(),
            rows,
        };
        (0..rows).into_par_iter().for_each(|row| {
            for token in 0..batch {
                let value = dot(row, token);
                // SAFETY: `gemm_rows` only calls `at` with its own `row`, and
                // `RowTargets`'s contract makes those destinations disjoint.
                unsafe { targets.at(row, token).write(value) };
            }
        });
    } else {
        for row in 0..rows {
            for token in 0..batch {
                output[token * rows + row] = dot(row, token);
            }
        }
    }
}

/// Row-parallel driver for the fused batched kernel.
///
/// Same contract as [`gemm_rows`] — the weight row stays the outer loop, so it
/// is streamed once and reused for the whole batch — but a row now produces
/// [`Q38_Q8_K_TILE`] values per fused call instead of one value per token.
/// The tail (batch % TILE, and 1) falls back to narrower tiles and finally to
/// the single-token kernel, so the result is independent of the batch size.
#[allow(clippy::too_many_arguments)]
fn gemm_rows_tile(
    output: &mut [f32],
    rows: usize,
    batch: usize,
    rbytes: usize,
    weights: &[u8],
    activations: &[Q38Q8KBlock],
    blocks: usize,
    per_row: usize,
    ty: GgmlType,
) {
    let targets = RowTargets {
        base: output.as_mut_ptr(),
        rows,
    };
    let run = |row: usize| {
        let data = &weights[row * rbytes..(row + 1) * rbytes];
        let mut values = [0.0f32; Q38_Q8_K_TILE];
        let mut token = 0usize;
        while token + Q38_Q8_K_TILE <= batch {
            let slab = &activations[token * blocks..(token + Q38_Q8_K_TILE) * blocks];
            dot_q8_k_tile::<Q38_Q8_K_TILE>(data, slab, blocks, ty, &mut values);
            for (lane, value) in values.iter().enumerate() {
                // SAFETY: `row` is owned by this task and `token + lane` is a
                // distinct column, so every destination is written once.
                unsafe { targets.at(row, token + lane).write(*value) };
            }
            token += Q38_Q8_K_TILE;
        }
        let tail = batch - token;
        if tail >= 2 {
            let slab = &activations[token * blocks..];
            dot_q8_k_tile_tail(data, slab, blocks, tail, ty, &mut values);
            for (lane, value) in values[..tail].iter().enumerate() {
                unsafe { targets.at(row, token + lane).write(*value) };
            }
            token += tail;
        }
        while token < batch {
            let value = dot_q8_k(
                data,
                &activations[token * blocks..(token + 1) * blocks],
                blocks,
                ty,
            );
            unsafe { targets.at(row, token).write(value) };
            token += 1;
        }
    };
    if q38_parallel_over(rows, per_row) {
        (0..rows).into_par_iter().for_each(|row| run(row));
    } else {
        for row in 0..rows {
            run(row);
        }
    }
}

/// The strided write view the batched row loop needs.
///
/// Row `r` owns exactly `{token * rows + r | token < batch}`, and `rows` is the
/// stride, so two different rows can never name the same slot. That
/// disjointness is what makes handing one row per worker sound, and it is the
/// only invariant this type carries.
#[derive(Clone, Copy)]
struct RowTargets {
    base: *mut f32,
    rows: usize,
}

// SAFETY: `RowTargets` is only ever copied into workers that each own a
// distinct `row`, so no two threads write the same address.
unsafe impl Send for RowTargets {}
// SAFETY: see `Send`.
unsafe impl Sync for RowTargets {}

impl RowTargets {
    /// Pointer to `output[token * rows + row]`.
    ///
    /// # Safety
    /// The caller must pass a `row` no other live worker is using, and the
    /// buffer must hold at least `(token + 1) * rows` elements.
    #[inline]
    unsafe fn at(&self, row: usize, token: usize) -> *mut f32 {
        unsafe { self.base.add(token * self.rows + row) }
    }
}

impl Q38TensorOps for TensorEntry<'_> {
    fn tensor_row_f32(&self, output: &mut [f32], row: u64) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1];
        if self.n_dims != 2 || row >= rows {
            return Err(Q38CoreError::RowOutOfBounds { row, rows });
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if output.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: output.len() as u64,
            });
        }
        let data = &self.data[row as usize * rbytes..(row as usize + 1) * rbytes];
        match self.ty {
            GgmlType::F32 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = read_f32(&data[i * 4..]);
                }
            }
            GgmlType::F16 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = f16_to_f32(read_u16(&data[i * 2..]));
                }
            }
            GgmlType::Bf16 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = bf16_to_f32(&data[i * 2..]);
                }
            }
            ty if supported_f32_gemv(ty) => {
                dequantize_quant_row(data, &mut output[..width], self.ty);
            }
            _ => return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32)),
        }
        Ok(())
    }

    fn tensor_dot_row_f32(&self, input: &[f32], row: u64) -> Q38CoreResult<f32> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1];
        if self.n_dims != 2 || row >= rows {
            return Err(Q38CoreError::RowOutOfBounds { row, rows });
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: input.len() as u64,
            });
        }
        let data = &self.data[row as usize * rbytes..(row as usize + 1) * rbytes];
        Ok(dot_row(data, &input[..width], width, self.ty))
    }

    fn tensor_gemv_f32(&self, output: &mut [f32], input: &[f32]) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        if self.n_dims != 2 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: rows as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let input = &input[..width];
        gemv_rows(output, rows, width, |row| {
            dot_row(
                &weights[row * rbytes..(row + 1) * rbytes],
                input,
                width,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemv_q8_k(&self, output: &mut [f32], input: &[Q38Q8KBlock]) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        if self.n_dims != 2 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_q8_k(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        let blocks = width / BLOCK;
        if input.len() < blocks {
            return Err(Q38CoreError::ShapeMismatch {
                expected: blocks as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: rows as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let activations = &input[..blocks];
        if let Some(repacked) = self.repacked_iq1_s(blocks) {
            let stride = blocks * iq::IQ1S_REPACK_BLOCK_BYTES;
            gemv_rows(output, rows, width, |row| {
                iq::dot_iq1_s_repacked_q8_k(&repacked[row * stride..], activations, blocks)
            });
            return Ok(());
        }
        gemv_rows(output, rows, width, |row| {
            dot_q8_k(
                &weights[row * rbytes..(row + 1) * rbytes],
                activations,
                blocks,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemm_f32(
        &self,
        output: &mut [f32],
        input: &[f32],
        batch_size: u32,
    ) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        let batch = batch_size as usize;
        if self.n_dims != 2 || batch == 0 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < batch * width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * width) as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < batch * rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * rows) as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        gemm_rows(output, rows, batch, width * batch, |row, token| {
            let start = token * width;
            dot_row(
                &weights[row * rbytes..(row + 1) * rbytes],
                &input[start..start + width],
                width,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemm_q8_k(
        &self,
        output: &mut [f32],
        input: &[Q38Q8KBlock],
        batch_size: u32,
    ) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        let batch = batch_size as usize;
        if self.n_dims != 2 || batch == 0 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_q8_k(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes =
            row_bytes(self.ty, width).ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        let blocks = width / BLOCK;
        if input.len() < batch * blocks {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * blocks) as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < batch * rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * rows) as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let activations = &input[..batch * blocks];
        if let Some(repacked) = self.repacked_iq1_s(blocks) {
            let stride = blocks * iq::IQ1S_REPACK_BLOCK_BYTES;
            gemm_rows(output, rows, batch, width * batch, |row, token| {
                let start = token * blocks;
                iq::dot_iq1_s_repacked_q8_k(
                    &repacked[row * stride..],
                    &activations[start..start + blocks],
                    blocks,
                )
            });
            return Ok(());
        }
        gemm_rows_tile(
            output,
            rows,
            batch,
            rbytes,
            weights,
            activations,
            blocks,
            width * batch,
            self.ty,
        );
        Ok(())
    }
}
