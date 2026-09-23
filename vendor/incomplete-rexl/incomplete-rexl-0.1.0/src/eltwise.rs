use crate::cpu_features::{HAS_AVX512DQ, HAS_AVX512IFMA};
use crate::number_theory::{
    add_uint_mod, multiply_mod, barrett_reduce64, log2, maximum_value, multiply_mod_precon,
    reduce_mod, MultiplyFactor,
};
use crate::util::multiply_u64_full;

#[cfg(target_arch = "x86_64")]
use crate::avx512_util::{
    mm512_hexl_barrett_reduce64, mm512_hexl_mulhi_approx_epi, mm512_hexl_mulhi_epi,
    mm512_hexl_mullo_add_lo_epi, mm512_hexl_mullo_epi, mm512_hexl_shrdi_epi64,
    mm512_hexl_shrdi_epi64_runtime, mm512_hexl_small_add_mod_epi64, mm512_hexl_small_mod_epu64,
    mm512_hexl_small_sub_mod_epi64,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn reduce_mod_input<const INPUT_MOD_FACTOR: i32>(
    x: u64,
    modulus: u64,
    twice_modulus: Option<&u64>,
    four_times_modulus: Option<&u64>,
) -> u64 {
    match INPUT_MOD_FACTOR {
        1 => reduce_mod::<1>(x, modulus, None, None),
        2 => reduce_mod::<2>(x, modulus, None, None),
        4 => reduce_mod::<4>(x, modulus, twice_modulus, None),
        8 => reduce_mod::<8>(x, modulus, twice_modulus, four_times_modulus),
        _ => x,
    }
}

fn eltwise_mult_mod_native_dispatch<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    match INPUT_MOD_FACTOR {
        1 => eltwise_mult_mod_native::<1>(result, operand1, operand2, modulus),
        2 => eltwise_mult_mod_native::<2>(result, operand1, operand2, modulus),
        4 => eltwise_mult_mod_native::<4>(result, operand1, operand2, modulus),
        _ => {}
    }
}

fn eltwise_fma_mod_native_dispatch<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    match INPUT_MOD_FACTOR {
        1 => eltwise_fma_mod_native::<1>(result, arg1, arg2, arg3, modulus),
        2 => eltwise_fma_mod_native::<2>(result, arg1, arg2, arg3, modulus),
        4 => eltwise_fma_mod_native::<4>(result, arg1, arg2, arg3, modulus),
        8 => eltwise_fma_mod_native::<8>(result, arg1, arg2, arg3, modulus),
        _ => {}
    }
}

/// Public fused multiply-add: result = arg1 * arg2 + arg3 (elementwise)
pub fn eltwise_fma_mod(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    let n = result.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512IFMA {
            unsafe {
                // call AVX-512 variant if available; fall back to native if missing
                eltwise_fma_mod_avx512::<1>(result, arg1, arg2, arg3, modulus);
                return;
            }
        }
    }

    // Default to native implementation
    eltwise_fma_mod_native_dispatch::<1>(result, arg1, arg2, arg3, modulus);
}

// Generic native implementation for fused multiply-add. Uses INPUT_MOD_FACTOR
fn eltwise_fma_mod_native<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    let twice = modulus.wrapping_mul(2);
    for i in 0..result.len() {
        let a = reduce_mod_input::<INPUT_MOD_FACTOR>(arg1[i], modulus, Some(&twice), None);
        let prod = (a as u128).wrapping_mul(arg2 as u128);
        let mut v = (prod % (modulus as u128)) as u64;
        if let Some(c) = arg3 {
            let cval = reduce_mod_input::<INPUT_MOD_FACTOR>(c[i], modulus, Some(&twice), None);
            v = add_uint_mod(v, cval, modulus);
        }
        result[i] = v;
    }
}

// AVX-512 stub wrappers that delegate to native implementations when intrinsics
// were removed for stable builds.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eltwise_fma_mod_avx512<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    arg1: &[u64],
    arg2: u64,
    arg3: Option<&[u64]>,
    modulus: u64,
) {
    eltwise_fma_mod_native::<INPUT_MOD_FACTOR>(result, arg1, arg2, arg3, modulus);
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eltwise_sub_mod_avx512(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    eltwise_sub_mod_native(result, operand1, operand2, modulus);
}

#[cfg(target_arch = "x86_64")]
unsafe fn eltwise_mult_mod_avx512_float<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    eltwise_mult_mod_native::<INPUT_MOD_FACTOR>(result, operand1, operand2, modulus);
}

#[cfg(target_arch = "x86_64")]
unsafe fn eltwise_mult_mod_avx512_dq_int<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    eltwise_mult_mod_native::<INPUT_MOD_FACTOR>(result, operand1, operand2, modulus);
}

/// Fallback portable implementation of fused incomplete NTT multiplication.
pub fn fused_incomplete_ntt_mult_inner(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    shift_factors: &[u64],
    _shift_factors_f64: &[f64],
    n: usize,
    modulus: u64,
) {
    // Expect result and operands to be length >= 2*n
    for i in 0..n {
        let a0 = operand1[i];
        let b0 = operand2[i];
        let a1 = operand1[n + i];
        let b1 = operand2[n + i];

        let t0 = multiply_mod(a0, b0, modulus);
        let t1 = multiply_mod(a1, b1, modulus);
        let shifted = multiply_mod(shift_factors[i], t1, modulus);
        result[i] = add_uint_mod(t0, shifted, modulus);

        let t2 = multiply_mod(a1, b0, modulus);
        let t3 = multiply_mod(a0, b1, modulus);
        result[n + i] = add_uint_mod(t2, t3, modulus);
    }
}

pub fn eltwise_add_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            unsafe {
                eltwise_add_mod_avx512(result, operand1, operand2, modulus);
                return;
            }
        }
    }

    eltwise_add_mod_native(result, operand1, operand2, modulus);
}

fn eltwise_add_mod_native(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    for i in 0..result.len() {
        let sum = operand1[i] + operand2[i];
        result[i] = if sum >= modulus { sum - modulus } else { sum };
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eltwise_add_mod_avx512(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    let mut n = result.len();
    let mut idx = 0usize;
    let n_mod_8 = n % 8;
    if n_mod_8 != 0 {
        eltwise_add_mod_native(
            &mut result[..n_mod_8],
            &operand1[..n_mod_8],
            &operand2[..n_mod_8],
            modulus,
        );
        idx += n_mod_8;
        n -= n_mod_8;
    }

    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let mut vp_result = result[idx..].as_mut_ptr() as *mut __m512i;
    let mut vp_operand1 = operand1[idx..].as_ptr() as *const __m512i;
    let mut vp_operand2 = operand2[idx..].as_ptr() as *const __m512i;

    for _ in (0..n).step_by(8) {
        let v_operand1 = _mm512_loadu_si512(vp_operand1 as *const i32);
        let v_operand2 = _mm512_loadu_si512(vp_operand2 as *const i32);
        let v_result = mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);
        _mm512_storeu_si512(vp_result, v_result);
        vp_result = vp_result.add(1);
        vp_operand1 = vp_operand1.add(1);
        vp_operand2 = vp_operand2.add(1);
    }
}

pub fn eltwise_sub_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            unsafe {
                eltwise_sub_mod_avx512(result, operand1, operand2, modulus);
                return;
            }
        }
    }

    eltwise_sub_mod_native(result, operand1, operand2, modulus);
}

fn eltwise_sub_mod_native(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    for i in 0..result.len() {
        if operand1[i] >= operand2[i] {
            result[i] = operand1[i] - operand2[i];
        } else {
            result[i] = operand1[i] + modulus - operand2[i];
        }
    }
}

pub fn eltwise_reduce_mod(result: &mut [u64], operand: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }

    let input_mod_factor = modulus;
    let output_mod_factor = 1u64;

    if input_mod_factor == output_mod_factor && !core::ptr::eq(result.as_ptr(), operand.as_ptr()) {
        result.copy_from_slice(operand);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512IFMA
            && ((modulus < (1u64 << 51)) || (modulus < (1u64 << 52) && input_mod_factor <= 4))
        {
            unsafe {
                eltwise_reduce_mod_avx512::<52>(
                    result,
                    operand,
                    modulus,
                    input_mod_factor,
                    output_mod_factor,
                );
                return;
            }
        }

        if *HAS_AVX512DQ {
            unsafe {
                eltwise_reduce_mod_avx512::<64>(
                    result,
                    operand,
                    modulus,
                    input_mod_factor,
                    output_mod_factor,
                );
                return;
            }
        }
    }

    eltwise_reduce_mod_native(
        result,
        operand,
        modulus,
        input_mod_factor,
        output_mod_factor,
    );
}

fn eltwise_reduce_mod_native(
    result: &mut [u64],
    operand: &[u64],
    modulus: u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(input_mod_factor == modulus || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);
    debug_assert!(input_mod_factor != output_mod_factor);

    let barrett_factor = MultiplyFactor::new(1, 64, modulus).barrett_factor();
    let twice_modulus = modulus << 1;

    if input_mod_factor == modulus {
        if output_mod_factor == 2 {
            for i in 0..result.len() {
                result[i] = if operand[i] >= twice_modulus {
                    barrett_reduce64::<2>(operand[i], modulus, barrett_factor)
                } else {
                    operand[i]
                };
            }
        } else {
            for i in 0..result.len() {
                result[i] = if operand[i] >= modulus {
                    barrett_reduce64::<1>(operand[i], modulus, barrett_factor)
                } else {
                    operand[i]
                };
            }
        }
        return;
    }

    if input_mod_factor == 2 {
        for i in 0..result.len() {
            result[i] = reduce_mod::<2>(operand[i], modulus, None, None);
        }
        return;
    }

    if input_mod_factor == 4 {
        if output_mod_factor == 1 {
            for i in 0..result.len() {
                result[i] = reduce_mod::<4>(operand[i], modulus, Some(&twice_modulus), None);
            }
        } else {
            for i in 0..result.len() {
                result[i] = reduce_mod::<2>(operand[i], twice_modulus, None, None);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn eltwise_reduce_mod_avx512<const BITSHIFT: i32>(
    result: &mut [u64],
    operand: &[u64],
    modulus: u64,
    input_mod_factor: u64,
    output_mod_factor: u64,
) {
    debug_assert!(input_mod_factor == modulus || input_mod_factor == 2 || input_mod_factor == 4);
    debug_assert!(output_mod_factor == 1 || output_mod_factor == 2);
    debug_assert!(input_mod_factor != output_mod_factor);

    let mut n_tmp = result.len();

    let alpha = BITSHIFT as i64 - 2;
    let beta = -2i64;
    let ceil_log_mod = log2(modulus) + 1;
    let prod_right_shift = (ceil_log_mod as i64 + beta) as u64;
    let v_neg_mod = _mm512_set1_epi64(-(modulus as i64));

    let mut barrett_factor = MultiplyFactor::new(
        1u64 << (ceil_log_mod + alpha as u64 - BITSHIFT as u64),
        BITSHIFT as u64,
        modulus,
    )
    .barrett_factor();
    let barrett_factor_52 = MultiplyFactor::new(1, 52, modulus).barrett_factor();

    if BITSHIFT == 64 {
        barrett_factor = MultiplyFactor::new(1, 64, modulus).barrett_factor();
    }

    let v_bf = _mm512_set1_epi64(barrett_factor as i64);
    let v_bf_52 = _mm512_set1_epi64(barrett_factor_52 as i64);

    let n_mod_8 = n_tmp % 8;
    let mut op_ptr = operand;
    let mut res_ptr = result;
    if n_mod_8 != 0 {
        eltwise_reduce_mod_native(
            &mut res_ptr[..n_mod_8],
            &op_ptr[..n_mod_8],
            modulus,
            input_mod_factor,
            output_mod_factor,
        );
        op_ptr = &op_ptr[n_mod_8..];
        res_ptr = &mut res_ptr[n_mod_8..];
        n_tmp -= n_mod_8;
    }

    let twice_mod = modulus << 1;
    let mut v_operand = op_ptr.as_ptr() as *const __m512i;
    let mut v_result = res_ptr.as_mut_ptr() as *mut __m512i;
    let v_modulus = _mm512_set1_epi64(modulus as i64);
    let v_twice_mod = _mm512_set1_epi64(twice_mod as i64);

    if input_mod_factor == modulus {
        if output_mod_factor == 2 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand as *const i32);
                v_op = mm512_hexl_barrett_reduce64::<BITSHIFT, 2>(
                    v_op,
                    v_modulus,
                    v_bf,
                    v_bf_52,
                    prod_right_shift,
                    v_neg_mod,
                );
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        } else {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand as *const i32);
                v_op = mm512_hexl_barrett_reduce64::<BITSHIFT, 1>(
                    v_op,
                    v_modulus,
                    v_bf,
                    v_bf_52,
                    prod_right_shift,
                    v_neg_mod,
                );
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
    }

    if input_mod_factor == 2 {
        for _ in (0..n_tmp).step_by(8) {
            let mut v_op = _mm512_loadu_si512(v_operand as *const i32);
            v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_modulus, None, None);
            _mm512_storeu_si512(v_result, v_op);
            v_operand = v_operand.add(1);
            v_result = v_result.add(1);
        }
    }

    if input_mod_factor == 4 {
        if output_mod_factor == 1 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand as *const i32);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_twice_mod, None, None);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_modulus, None, None);
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
        if output_mod_factor == 2 {
            for _ in (0..n_tmp).step_by(8) {
                let mut v_op = _mm512_loadu_si512(v_operand as *const i32);
                v_op = mm512_hexl_small_mod_epu64::<2>(v_op, v_twice_mod, None, None);
                _mm512_storeu_si512(v_result, v_op);
                v_operand = v_operand.add(1);
                v_result = v_result.add(1);
            }
        }
    }
}

pub fn eltwise_mult_mod(result: &mut [u64], operand1: &[u64], operand2: &[u64], modulus: u64) {
    let n = result.len();
    if n == 0 {
        return;
    }
    let input_mod_factor = 1u64;

    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_AVX512DQ {
            if modulus < (1u64 << 50) {
                match input_mod_factor {
                    1 => unsafe {
                        eltwise_mult_mod_avx512_float::<1>(result, operand1, operand2, modulus)
                    },
                    2 => unsafe {
                        eltwise_mult_mod_avx512_float::<2>(result, operand1, operand2, modulus)
                    },
                    4 => unsafe {
                        eltwise_mult_mod_avx512_float::<4>(result, operand1, operand2, modulus)
                    },
                    _ => {}
                }
            } else {
                match input_mod_factor {
                    1 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<1>(result, operand1, operand2, modulus)
                    },
                    2 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<2>(result, operand1, operand2, modulus)
                    },
                    4 => unsafe {
                        eltwise_mult_mod_avx512_dq_int::<4>(result, operand1, operand2, modulus)
                    },
                    _ => {}
                }
            }
            return;
        }
    }

    match input_mod_factor {
        1 => eltwise_mult_mod_native::<1>(result, operand1, operand2, modulus),
        2 => eltwise_mult_mod_native::<2>(result, operand1, operand2, modulus),
        4 => eltwise_mult_mod_native::<4>(result, operand1, operand2, modulus),
        _ => {}
    }
}

fn eltwise_mult_mod_native<const INPUT_MOD_FACTOR: i32>(
    result: &mut [u64],
    operand1: &[u64],
    operand2: &[u64],
    modulus: u64,
) {
    debug_assert!(INPUT_MOD_FACTOR == 1 || INPUT_MOD_FACTOR == 2 || INPUT_MOD_FACTOR == 4);
    debug_assert!(modulus < (1u64 << 62));

    let beta = -2i64;
    let alpha = 62i64;
    let ceil_log_mod = log2(modulus) + 1;
    let prod_right_shift = (ceil_log_mod as i64 + beta) as u64;

    let barr_lo = MultiplyFactor::new(1u64 << (ceil_log_mod + alpha as u64 - 64), 64, modulus)
        .barrett_factor();

    let twice_modulus = 2 * modulus;

    for i in 0..result.len() {
        let x = reduce_mod_input::<INPUT_MOD_FACTOR>(operand1[i], modulus, Some(&twice_modulus), None);
        let y = reduce_mod_input::<INPUT_MOD_FACTOR>(operand2[i], modulus, Some(&twice_modulus), None);

        let (prod_hi, prod_lo) = multiply_u64_full(x, y);
        let c1 = (prod_lo >> prod_right_shift) + (prod_hi << (64 - prod_right_shift));
        let (c2_hi, _c2_lo) = multiply_u64_full(c1, barr_lo);
        let q_hat = c2_hi;
        let z = prod_lo.wrapping_sub(q_hat.wrapping_mul(modulus));
        result[i] = if z >= modulus { z - modulus } else { z };
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn eltwise_mult_mod_avx512_dq_int_loop_const<
    const PROD_RIGHT_SHIFT: i32,
    const INPUT_MOD_FACTOR: i32,
>(
    vp_result: *mut __m512i,
    vp_operand1: *const __m512i,
    vp_operand2: *const __m512i,
    v_barr_lo: __m512i)
{
    // AVX-512 specific optimized loop removed for stable toolchain compatibility.
    // Fallback to native implementation is used instead.
    let _ = (vp_result, vp_operand1, vp_operand2, v_barr_lo, PROD_RIGHT_SHIFT, INPUT_MOD_FACTOR);
}