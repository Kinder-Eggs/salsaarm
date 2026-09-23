#[cfg(not(feature = "incomplete-rexl"))]
#[link(name = "hexl_wrapper")]
extern "C" {
    pub fn multiply_mod(a: u64, b: u64, modulus: u64) -> u64;
    pub fn power_mod(a: u64, b: u64, modulus: u64) -> u64;
    pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64;
    pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64;
    pub fn eltwise_mult_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );
    pub fn get_roots(n: u64, modulus: u64) -> *const u64;
    pub fn inv_mod(a: u64, modulus: u64) -> u64;
    pub fn get_inv_roots(n: u64, modulus: u64) -> *const u64;
    pub fn sum_sq(operand1: *const u64, n: u64, modulus: u64) -> u64;
    pub fn sum_sq_fast(operand1: *const u64, n: u64, modulus: u64) -> u64;
    pub fn sum(operand1: *const u64, n: u64, modulus: u64) -> u64;
    pub fn sum_fast(operand1: *const u64, n: u64, modulus: u64) -> u64;
    pub fn eltwise_fma_mod(
        result: *mut u64,
        operand1: *const u64,
        scalar: u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );
    pub fn eltwise_add_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );
    pub fn eltwise_sub_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );
    pub fn multiply_poly(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        n: u64,
        modulus: u64,
    );
    pub fn eltwise_reduce_mod(result: *mut u64, operand: *const u64, n: u64, modulus: u64);
    pub fn eltwise_reduce_mod_naive(result: *mut u64, operand: *const u64, n: u64, modulus: u64);
    pub fn polynomial_multiply_cyclotomic_mod(
        result: *mut u64,
        operand1: *const u64,
        operand2: *const u64,
        phi: u64,
        mod_q: u64,
    );
    pub fn ntt_forward_in_place(operand: *mut u64, n: usize, modulus: u64);
    pub fn ntt_inverse_in_place(operand: *mut u64, n: usize, modulus: u64);
}

#[cfg(feature = "incomplete-rexl")]
pub fn multiply_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128) * (b as u128) % modulus as u128) as u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn power_mod(a: u64, b: u64, modulus: u64) -> u64 {
    let mut base = a % modulus;
    let mut exp = b;
    let mut result = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }
    result
}

#[cfg(feature = "incomplete-rexl")]
pub fn add_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 + b as u128) % modulus as u128) as u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn sub_mod(a: u64, b: u64, modulus: u64) -> u64 {
    let modulus = modulus as u128;
    let diff = (a as u128 + modulus - b as u128) % modulus;
    diff as u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_mult_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64) {
    if n == 0 {
        return;
    }
    let len = n as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let modulus = modulus as u128;
    for i in 0..len {
        out[i] = ((lhs[i] as u128 * rhs[i] as u128) % modulus) as u64;
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_add_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64) {
    if n == 0 {
        return;
    }
    let len = n as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let modulus = modulus as u128;
    for i in 0..len {
        out[i] = ((lhs[i] as u128 + rhs[i] as u128) % modulus) as u64;
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_sub_mod(result: *mut u64, operand1: *const u64, operand2: *const u64, n: u64, modulus: u64) {
    if n == 0 {
        return;
    }
    let len = n as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let modulus = modulus as u128;
    for i in 0..len {
        let diff = if lhs[i] >= rhs[i] {
            lhs[i] as u128 - rhs[i] as u128
        } else {
            modulus - (rhs[i] as u128 - lhs[i] as u128)
        };
        out[i] = (diff % modulus) as u64;
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_reduce_mod(result: *mut u64, operand: *const u64, n: u64, modulus: u64) {
    if n == 0 {
        return;
    }
    let len = n as usize;
    let src = unsafe { std::slice::from_raw_parts(operand, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    for i in 0..len {
        out[i] = src[i] % modulus;
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_reduce_mod_naive(result: *mut u64, operand: *const u64, n: u64, modulus: u64) {
    eltwise_reduce_mod(result, operand, n, modulus);
}

#[cfg(feature = "incomplete-rexl")]
pub fn sum_sq(operand1: *const u64, n: u64, modulus: u64) -> u64 {
    let len = n as usize;
    let src = unsafe { std::slice::from_raw_parts(operand1, len) };
    let modulus = modulus as u128;
    let mut total = 0u128;
    for &x in src {
        total = (total + (x as u128 * x as u128)) % modulus;
    }
    total as u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn sum_sq_fast(operand1: *const u64, n: u64, modulus: u64) -> u64 {
    sum_sq(operand1, n, modulus)
}

#[cfg(feature = "incomplete-rexl")]
pub fn sum(operand1: *const u64, n: u64, modulus: u64) -> u64 {
    let len = n as usize;
    let src = unsafe { std::slice::from_raw_parts(operand1, len) };
    let modulus = modulus as u128;
    let mut total = 0u128;
    for &x in src {
        total = (total + x as u128) % modulus;
    }
    total as u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn sum_fast(operand1: *const u64, n: u64, modulus: u64) -> u64 {
    sum(operand1, n, modulus)
}

#[cfg(feature = "incomplete-rexl")]
pub fn get_roots(n: u64, modulus: u64) -> *const u64 {
    let len = n as usize;
    let mut roots = vec![0u64; len];
    for i in 0..len {
        roots[i] = power_mod(i as u64 + 1, modulus - 2, modulus);
    }
    let ptr = roots.into_boxed_slice();
    let raw = Box::into_raw(ptr);
    raw as *const u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn inv_mod(a: u64, modulus: u64) -> u64 {
    if a == 0 { 0 } else { power_mod(a, modulus - 2, modulus) }
}

#[cfg(feature = "incomplete-rexl")]
pub fn get_inv_roots(n: u64, modulus: u64) -> *const u64 {
    let len = n as usize;
    let mut roots = vec![0u64; len];
    for i in 0..len {
        roots[i] = inv_mod((i as u64 + 1) % modulus, modulus);
    }
    let ptr = roots.into_boxed_slice();
    let raw = Box::into_raw(ptr);
    raw as *const u64
}

#[cfg(feature = "incomplete-rexl")]
pub fn eltwise_fma_mod(
    result: *mut u64,
    operand1: *const u64,
    scalar: u64,
    operand2: *const u64,
    n: u64,
    modulus: u64,
) {
    if n == 0 {
        return;
    }
    let len = n as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let modulus = modulus as u128;
    for i in 0..len {
        out[i] = (((lhs[i] as u128 * scalar as u128) + rhs[i] as u128) % modulus) as u64;
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn multiply_poly(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    n: u64,
    modulus: u64,
) {
    let len = n as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let modulus = modulus as u128;
    for i in 0..len {
        out[i] = 0;
    }
    for i in 0..len {
        for j in 0..len {
            let idx = (i + j) % len;
            let value = ((lhs[i] as u128 * rhs[j] as u128) % modulus) as u64;
            out[idx] = ((out[idx] as u128 + value as u128) % modulus) as u64;
        }
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn polynomial_multiply_cyclotomic_mod(
    result: *mut u64,
    operand1: *const u64,
    operand2: *const u64,
    phi: u64,
    mod_q: u64,
) {
    let len = phi as usize;
    let lhs = unsafe { std::slice::from_raw_parts(operand1, len) };
    let rhs = unsafe { std::slice::from_raw_parts(operand2, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(result, len) };
    let mod_q = mod_q as u128;

    for i in 0..len {
        out[i] = 0;
    }
    for i in 0..len {
        for j in 0..len {
            let idx = (i + j) % len;
            let value = ((lhs[i] as u128 * rhs[j] as u128) % mod_q) as u64;
            out[idx] = ((out[idx] as u128 + value as u128) % mod_q) as u64;
        }
    }
}

#[cfg(feature = "incomplete-rexl")]
pub fn ntt_forward_in_place(operand: *mut u64, n: usize, modulus: u64) {
    if n == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(operand, n) };
    incomplete_rexl::ntt_forward_in_place(slice, n, modulus);
}

#[cfg(feature = "incomplete-rexl")]
pub fn ntt_inverse_in_place(operand: *mut u64, n: usize, modulus: u64) {
    if n == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(operand, n) };
    incomplete_rexl::ntt_inverse_in_place(slice, n, modulus);
}

pub fn cpp_multiply_mod(a: u64, b: u64, modulus: u64) -> u64 {
    unsafe { multiply_mod(a, b, modulus) }
}

pub fn cpp_eltwise_mult_mod(result: &mut [u64], a: &[u64], b: &[u64], modulus: u64) {
    assert_eq!(result.len(), a.len());
    assert_eq!(a.len(), b.len());
    unsafe {
        eltwise_mult_mod(
            result.as_mut_ptr(),
            a.as_ptr(),
            b.as_ptr(),
            a.len() as u64,
            modulus,
        );
    }
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_mod() {
        let a: u64 = 12345;
        let b: u64 = 67890;
        let modulus: u64 = 1000000007;
        let result = cpp_multiply_mod(a, b, modulus);
        assert_eq!(result, 838102050);
    }

    #[test]
    fn test_eltwise_mult_mod() {
        let a: Vec<u64> = vec![1, 2, 3, 4, 5];
        let b: Vec<u64> = vec![6, 7, 8, 9, 10];
        let modulus: u64 = 100;
        let mut result: Vec<u64> = vec![0; a.len()];

        cpp_eltwise_mult_mod(&mut result, &a, &b, modulus);

        assert_eq!(result, vec![6, 14, 24, 36, 50]);
    }

    #[test]
    fn test_multiply_poly() {
        let n = 8;
        let modulus = 65537;
        let operand1: Vec<u64> = vec![1, 2, 3, 1, 0, 0, 0, 0];
        let operand2: Vec<u64> = vec![8, 7, 6, 1, 0, 0, 0, 0];
        let mut result = vec![0u64; n];

        let mut expected_result = vec![0u64; n];
        for i in 0..n / 2 {
            for j in 0..n / 2 {
                expected_result[i + j] =
                    (expected_result[i + j] + operand1[i] * operand2[j]) % modulus;
            }
        }

        unsafe {
            multiply_poly(
                result.as_mut_ptr(),
                operand1.as_ptr(),
                operand2.as_ptr(),
                n as u64,
                modulus,
            );
        }

        assert_eq!(result, expected_result);
    }
}
