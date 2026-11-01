use crate::kernel::{
    Q38_Q8_K_BLOCK_SIZE, Q38Q8KBlock,
    quant::{bf16_to_f32, f16_to_f32},
    tensor::{read_f32, read_u16, scale_min_k4, unpack_q3_scales},
};

const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

// ---- scalar dot products (f32 input) ----

pub(crate) fn dot_f32_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = read_f32(&data[i * 4..]).mul_add(input[i], total);
    }
    total
}

pub(crate) fn dot_f16_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = f16_to_f32(read_u16(&data[i * 2..])).mul_add(input[i], total);
    }
    total
}

pub(crate) fn dot_bf16_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = bf16_to_f32(&data[i * 2..]).mul_add(input[i], total);
    }
    total
}

pub(crate) fn dot_q8_0_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let d = f16_to_f32(read_u16(&data[block * 34..]));
        for i in 0..32 {
            let q = data[block * 34 + 2 + i] as i8;
            total = (d * q as f32).mul_add(input[base + i], total);
        }
        base += 32;
        block += 1;
    }
    total
}

pub(crate) fn dot_q2_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 84;
        let scales = &data[b..b + 16];
        let quants = &data[b + 16..b + 80];
        let d = f16_to_f32(read_u16(&data[b + 80..]));
        let dmin = f16_to_f32(read_u16(&data[b + 82..]));
        let mut scale_index = 0usize;
        for half in 0..2 {
            for field in 0..4 {
                let shift = field * 2;
                for group in 0..2 {
                    let packed_scale = scales[scale_index];
                    scale_index += 1;
                    let scale = d * (packed_scale & 15) as f32;
                    let minimum = dmin * (packed_scale >> 4) as f32;
                    let offset = base + half * 128 + field * 32 + group * 16;
                    let packed = &quants[half * 32 + group * 16..half * 32 + group * 16 + 16];
                    for lane in 0..16 {
                        let q = (packed[lane] >> shift) & 3;
                        total = (scale * q as f32 - minimum).mul_add(input[offset + lane], total);
                    }
                }
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

pub(crate) fn dot_q3_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 110;
        let high = &data[b..b + 32];
        let low = &data[b + 32..b + 96];
        let mut scales = [0i8; 16];
        // Q3_K packs its 16 six-bit scales into 12 bytes: 96..108. The `d`
        // half sits immediately after, at 108..110.
        unpack_q3_scales(&data[b + 96..b + 108], &mut scales);
        let d = f16_to_f32(read_u16(&data[b + 108..]));
        let mut group = 0usize;
        let mut high_bit: u8 = 1;
        for half in 0..2 {
            for field in 0..4 {
                let shift = field * 2;
                for lane in 0..32 {
                    let q = ((low[half * 32 + lane] >> shift) & 3) as i32
                        - if high[lane] & high_bit != 0 { 0 } else { 4 };
                    let scale = scales[group * 2 + lane / 16] as i32;
                    let offset = base + half * 128 + field * 32 + lane;
                    total = (d * scale as f32 * q as f32).mul_add(input[offset], total);
                }
                group += 1;
                high_bit <<= 1;
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

pub(crate) fn dot_q4_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 144;
        let d = f16_to_f32(read_u16(&data[b..]));
        let dmin = f16_to_f32(read_u16(&data[b + 2..]));
        let scales = &data[b + 4..b + 16];
        let mut quants = b + 16;
        let mut scale_index = 0usize;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(scale_index, scales);
            let (scale1, min1) = scale_min_k4(scale_index + 1, scales);
            let d0 = d * scale0 as f32;
            let m0 = dmin * min0 as f32;
            let d1 = d * scale1 as f32;
            let m1 = dmin * min1 as f32;
            let x = base + chunk * 64;
            for i in 0..32 {
                total = (d0 * (data[quants + i] & 15) as f32 - m0).mul_add(input[x + i], total);
            }
            for i in 0..32 {
                total =
                    (d1 * (data[quants + i] >> 4) as f32 - m1).mul_add(input[x + 32 + i], total);
            }
            quants += 32;
            scale_index += 2;
        }
        base += BLOCK;
        block += 1;
    }
    total
}

pub(crate) fn dot_q5_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 176;
        let d = f16_to_f32(read_u16(&data[b..]));
        let dmin = f16_to_f32(read_u16(&data[b + 2..]));
        let scales = &data[b + 4..b + 16];
        let high = &data[b + 16..b + 48];
        let low = &data[b + 48..b + 176];
        let mut scale_index = 0usize;
        let mut high0: u8 = 1;
        let mut high1: u8 = 2;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(scale_index, scales);
            let (scale1, min1) = scale_min_k4(scale_index + 1, scales);
            let d0 = d * scale0 as f32;
            let m0 = dmin * min0 as f32;
            let d1 = d * scale1 as f32;
            let m1 = dmin * min1 as f32;
            let x = base + chunk * 64;
            for i in 0..32 {
                let q =
                    (low[chunk * 32 + i] & 15) as i32 + if high[i] & high0 != 0 { 16 } else { 0 };
                total = (d0 * q as f32 - m0).mul_add(input[x + i], total);
            }
            for i in 0..32 {
                let q =
                    (low[chunk * 32 + i] >> 4) as i32 + if high[i] & high1 != 0 { 16 } else { 0 };
                total = (d1 * q as f32 - m1).mul_add(input[x + 32 + i], total);
            }
            scale_index += 2;
            high0 <<= 2;
            high1 <<= 2;
        }
        base += BLOCK;
        block += 1;
    }
    total
}

pub(crate) fn dot_q6_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 210;
        let d = f16_to_f32(read_u16(&data[b + 208..]));
        let scales = &data[b + 192..b + 208];
        for half in 0..2 {
            let x = base + half * 128;
            for i in 0..32 {
                let si = half * 8 + i / 16;
                let low_i = data[b + half * 64 + i];
                let low_i2 = data[b + half * 64 + i + 32];
                let h = data[b + 128 + half * 32 + i];
                let q0 = ((low_i & 15) | ((h & 3) << 4)) as i32 - 32;
                let q1 = ((low_i2 & 15) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q2 = ((low_i >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q3 = ((low_i2 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                total = (d * scales[si] as i8 as f32 * q0 as f32).mul_add(input[x + i], total);
                total =
                    (d * scales[si + 2] as i8 as f32 * q1 as f32).mul_add(input[x + 32 + i], total);
                total =
                    (d * scales[si + 4] as i8 as f32 * q2 as f32).mul_add(input[x + 64 + i], total);
                total =
                    (d * scales[si + 6] as i8 as f32 * q3 as f32).mul_add(input[x + 96 + i], total);
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

// ---- scalar dot products (Q8_K input) ----

pub(crate) fn dot_q2_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let scales = &data[block * 84..block * 84 + 16];
        let quants = &data[block * 84 + 16..block * 84 + 80];
        let d = f16_to_f32(read_u16(&data[block * 84 + 80..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 84 + 82..])) * b.scale;
        let mut weighted = 0i32;
        let mut minimum = 0i32;
        for half in 0..2 {
            for field in 0..4 {
                let scale_index = half * 8 + field * 2;
                let scale0 = (scales[scale_index] & 15) as i32;
                let scale1 = (scales[scale_index + 1] & 15) as i32;
                let shift = field * 2;
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for lane in 0..16 {
                    dot0 += ((quants[half * 32 + lane] >> shift) & 3) as i32
                        * b.quants[half * 128 + field * 32 + lane] as i32;
                    dot1 += ((quants[half * 32 + 16 + lane] >> shift) & 3) as i32
                        * b.quants[half * 128 + field * 32 + 16 + lane] as i32;
                }
                weighted += scale0 * dot0 + scale1 * dot1;
                minimum += (scales[scale_index] >> 4) as i32 * b.sums[scale_index] as i32;
                minimum += (scales[scale_index + 1] >> 4) as i32 * b.sums[scale_index + 1] as i32;
            }
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

pub(crate) fn dot_q3_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let high = &data[block * 110..block * 110 + 32];
        let low = &data[block * 110 + 32..block * 110 + 96];
        let mut scales = [0i8; 16];
        unpack_q3_scales(&data[block * 110 + 96..block * 110 + 108], &mut scales);
        let d = f16_to_f32(read_u16(&data[block * 110 + 108..])) * b.scale;
        let mut weighted = 0i32;
        let mut group = 0usize;
        for half in 0..2 {
            for field in 0..4 {
                let high_bit: u8 = 1 << group;
                let scale0 = scales[group * 2] as i32;
                let scale1 = scales[group * 2 + 1] as i32;
                for lane in 0..32 {
                    let qv = ((low[half * 32 + lane] >> (field * 2)) & 3) as i32
                        - if high[lane] & high_bit != 0 { 0 } else { 4 };
                    let scale = if lane < 16 { scale0 } else { scale1 };
                    weighted += scale * qv * b.quants[half * 128 + field * 32 + lane] as i32;
                }
                group += 1;
            }
        }
        total += d * weighted as f32;
    }
    total
}

pub(crate) fn dot_q4_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 144..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 144 + 2..])) * b.scale;
        let scales = &data[block * 144 + 4..block * 144 + 16];
        let quants = &data[block * 144 + 16..block * 144 + 144];
        let mut weighted = 0i32;
        let mut minimum = 0i32;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let mut dot0 = 0i32;
            let mut dot1 = 0i32;
            for i in 0..32 {
                dot0 += (quants[chunk * 32 + i] & 15) as i32 * b.quants[chunk * 64 + i] as i32;
                dot1 += (quants[chunk * 32 + i] >> 4) as i32 * b.quants[chunk * 64 + 32 + i] as i32;
            }
            weighted += scale0 as i32 * dot0 + scale1 as i32 * dot1;
            minimum += min0 as i32 * (b.sums[chunk * 4] + b.sums[chunk * 4 + 1]) as i32;
            minimum += min1 as i32 * (b.sums[chunk * 4 + 2] + b.sums[chunk * 4 + 3]) as i32;
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

pub(crate) fn dot_q5_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 176..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 176 + 2..])) * b.scale;
        let scales = &data[block * 176 + 4..block * 176 + 16];
        let high = &data[block * 176 + 16..block * 176 + 48];
        let low = &data[block * 176 + 48..block * 176 + 176];
        let mut minimum = 0i32;
        let mut weighted = 0i32;
        let mut high0: u8 = 1;
        let mut high1: u8 = 2;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let mut dot0 = 0i32;
            let mut dot1 = 0i32;
            for i in 0..32 {
                let qv0 =
                    (low[chunk * 32 + i] & 15) as i32 + if high[i] & high0 != 0 { 16 } else { 0 };
                let qv1 =
                    (low[chunk * 32 + i] >> 4) as i32 + if high[i] & high1 != 0 { 16 } else { 0 };
                dot0 += qv0 * b.quants[chunk * 64 + i] as i32;
                dot1 += qv1 * b.quants[chunk * 64 + 32 + i] as i32;
            }
            weighted += scale0 as i32 * dot0 + scale1 as i32 * dot1;
            minimum += min0 as i32 * (b.sums[chunk * 4] + b.sums[chunk * 4 + 1]) as i32;
            minimum += min1 as i32 * (b.sums[chunk * 4 + 2] + b.sums[chunk * 4 + 3]) as i32;
            high0 <<= 2;
            high1 <<= 2;
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

pub(crate) fn dot_q6_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 210 + 208..])) * b.scale;
        let scales = &data[block * 210 + 192..block * 210 + 208];
        let mut weighted = 0i32;
        for half in 0..2 {
            for i in 0..32 {
                let si = half * 8 + i / 16;
                let base = half * 128;
                let low_i = data[block * 210 + half * 64 + i];
                let low_i2 = data[block * 210 + half * 64 + i + 32];
                let h = data[block * 210 + 128 + half * 32 + i];
                let q0 = ((low_i & 15) | ((h & 3) << 4)) as i32 - 32;
                let q1 = ((low_i2 & 15) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q2 = ((low_i >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q3 = ((low_i2 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                weighted += scales[si] as i8 as i32 * q0 * b.quants[base + i] as i32;
                weighted += scales[si + 2] as i8 as i32 * q1 * b.quants[base + 32 + i] as i32;
                weighted += scales[si + 4] as i8 as i32 * q2 * b.quants[base + 64 + i] as i32;
                weighted += scales[si + 6] as i8 as i32 * q3 * b.quants[base + 96 + i] as i32;
            }
        }
        total += d * weighted as f32;
    }
    total
}
