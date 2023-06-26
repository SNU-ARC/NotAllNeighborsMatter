/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#include <cmath>
#include <type_traits>
#endif
#include <tensorview/core/all.h>
#include <tensorview/gemm/dtypes/uint128.h>
/**
 * \file
 * \brief Math utilities
 */

namespace tv {
namespace math {
/////////////////////////////////////////////////////////////////////////////////////////////////

/******************************************************************************
 * Static math utilities
 ******************************************************************************/

/**
 * Statically determine if N is a power-of-two
 */
template <int N>
struct is_pow2 {
  static bool const value = ((N & (N - 1)) == 0);
};

/**
 * Statically determine log2(N), rounded down
 */
template <int N, int CurrentVal = N, int Count = 0>
struct log2_down {
  /// Static logarithm value
  enum { value = log2_down<N, (CurrentVal >> 1), Count + 1>::value };
};

// Base case
template <int N, int Count>
struct log2_down<N, 1, Count> {
  enum { value = Count };
};

/**
 * Statically determine log2(N), rounded up
 */
template <int N, int CurrentVal = N, int Count = 0>
struct log2_up {
  /// Static logarithm value
  enum { value = log2_up<N, (CurrentVal >> 1), Count + 1>::value };
};

// Base case
template <int N, int Count>
struct log2_up<N, 1, Count> {
  enum { value = ((1 << Count) < N) ? Count + 1 : Count };
};

/**
 * Statically estimate sqrt(N) to the nearest power-of-two
 */
template <int N>
struct sqrt_est {
  enum { value = 1 << (log2_up<N>::value / 2) };
};

/**
 * For performing a constant-division with a compile-time assertion that the
 * Divisor evenly-divides the Dividend.
 */
template <int Dividend, int Divisor>
struct divide_assert {
  enum { value = Dividend / Divisor };

  static_assert((Dividend % Divisor == 0), "Not an even multiple");
};

/******************************************************************************
 * Rounding
 ******************************************************************************/

/**
 * Round dividend up to the nearest multiple of divisor
 */
template <typename dividend_t, typename divisor_t>
TV_HOST_DEVICE_INLINE dividend_t round_nearest(dividend_t dividend, divisor_t divisor) {
  return ((dividend + divisor - 1) / divisor) * divisor;
}

/**
 * Greatest common divisor
 */
template <typename value_t>
TV_HOST_DEVICE_INLINE value_t gcd(value_t a, value_t b) {
  for (;;) {
    if (a == 0) return b;
    b %= a;
    if (b == 0) return a;
    a %= b;
  }
}

/**
 * Least common multiple
 */
template <typename value_t>
TV_HOST_DEVICE_INLINE value_t lcm(value_t a, value_t b) {
  value_t temp = gcd(a, b);

  return temp ? (a / temp * b) : 0;
}

/// Returns the smallest value in the half-open range [a, a+b) that is a multiple of b
TV_HOST_DEVICE_INLINE
constexpr int round_up(int a, int b) {
  return ((a + b - 1) / b) * b;
}

/// Returns the ceiling of (a / b)
TV_HOST_DEVICE_INLINE
constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * log2 computation, what's the
 * difference between the below codes and
 * log2_up/down codes?
 */
template <typename value_t>
TV_HOST_DEVICE_INLINE value_t clz(value_t x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x) return 31 - i;
  }
  return 32;
}

template <typename value_t>
TV_HOST_DEVICE_INLINE value_t find_log2(value_t x) {
  int a = int(31 - clz(x));
  a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
  return a;
}


/**
 * Find divisor, using find_log2
 */
TV_HOST_DEVICE_INLINE 
void find_divisor(unsigned int& mul, unsigned int& shr, unsigned int denom) {
  if (denom == 1) {
    mul = 0;
    shr = 0;
  } else {
    unsigned int p = 31 + find_log2(denom);
    unsigned m = unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

    mul = m;
    shr = p - 32;
  }
}

/**
 * Find quotient and remainder using device-side intrinsics
 */
TV_HOST_DEVICE_INLINE 
void fast_divmod(int& quo, int& rem, int src, int div, unsigned int mul, unsigned int shr) {

  #if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
  #else
  quo = int((div != 1) ? int(((int64_t)src * mul) >> 32) >> shr : src);
  #endif

  // The remainder.
  rem = src - (quo * div);
}

// For long int input
TV_HOST_DEVICE_INLINE
void fast_divmod(int& quo, int64_t& rem, int64_t src, int div, unsigned int mul, unsigned int shr) {

  #if defined(__CUDA_ARCH__)
  // Use IMUL.HI if div != 1, else simply copy the source.
  quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
  #else
  quo = int((div != 1) ? ((src * mul) >> 32) >> shr : src);
  #endif
  // The remainder.
  rem = src - (quo * div);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);  
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod {

  int divisor;
  unsigned int multiplier;
  unsigned int shift_right;

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.

  TV_HOST_DEVICE_INLINE
  FastDivmod(): divisor(0), multiplier(0), shift_right(0) { }

  TV_HOST_DEVICE_INLINE
  FastDivmod(int divisor_): divisor(divisor_) {
    find_divisor(multiplier, shift_right, divisor);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  TV_HOST_DEVICE_INLINE
  void operator()(int &quotient, int &remainder, int dividend) const {
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  TV_HOST_DEVICE_INLINE
  void operator()(int &quotient, int64_t &remainder, int64_t dividend) const {
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Object to encapsulate the fast division+modulus operation for 64b integer division.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshalled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   uint64_t quotient, remainder, dividend, divisor;
///
///   FastDivmodU64 divmod(divisor);
///
///   divmod(quotient, remainder, dividend);  
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmodU64 {

  uint64_t divisor;
  uint64_t multiplier;
  unsigned int shift_right;
  unsigned int round_up;

  //
  // Static methods
  //

  /// Computes b, where 2^b is the greatest power of two that is less than or equal to x
  TV_HOST_DEVICE_INLINE
  static uint32_t integer_log2(uint64_t x) {
    uint32_t n = 0;
    while (x >>= 1) {
      ++n;
    }
    return n;
  }

  /// Default ctor
  TV_HOST_DEVICE_INLINE
  FastDivmodU64(): divisor(0), multiplier(0), shift_right(0), round_up(0) { }

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.
  TV_HOST_DEVICE_INLINE
  FastDivmodU64(uint64_t divisor_): divisor(divisor_), multiplier(1), shift_right(0), round_up(0) {

    if (divisor) {
      shift_right = integer_log2(divisor);

      if ((divisor & (divisor - 1)) == 0) {
        multiplier = 0;
      }
      else {
        uint64_t power_of_two = (uint64_t(1) << shift_right);
        uint64_t multiplier_lo = uint128_t(0, power_of_two) / divisor;
        multiplier = uint128_t(power_of_two, power_of_two) / divisor;
        round_up = (multiplier_lo == multiplier ? 1 : 0);
      }
    }
  }

  /// Returns the quotient of floor(dividend / divisor)
  TV_HOST_DEVICE_INLINE
  uint64_t divide(uint64_t dividend) const {
    uint64_t quotient = 0;

    #ifdef __CUDA_ARCH__
      uint64_t x = dividend;
      if (multiplier) {
        x = __umul64hi(dividend + round_up, multiplier);
      }
      quotient = (x >> shift_right);
    #else
      // TODO - use proper 'fast' division here also. No reason why x86-code shouldn't be optimized.
      quotient = dividend / divisor;
    #endif

    return quotient;
  }

  /// Computes the remainder given a computed quotient and dividend
  TV_HOST_DEVICE_INLINE
  uint64_t modulus(uint64_t quotient, uint64_t dividend) const {
    return uint32_t(dividend - quotient * divisor);
  }

  /// Returns the quotient of floor(dividend / divisor) and computes the remainder
  TV_HOST_DEVICE_INLINE
  uint64_t divmod(uint64_t &remainder, uint64_t dividend) const {
    uint64_t quotient = divide(dividend);
    remainder = modulus(quotient, dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  TV_HOST_DEVICE_INLINE
  void operator()(uint64_t &quotient, uint64_t &remainder, uint64_t dividend) const {
    quotient = divmod(remainder, dividend);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes the coordinate decomposition from a linear index.
///
/// This decomposition is accelerated by the FastDivmodU64 object. It is assumed that
/// a coordinate of <Rank> indices can be decomposed by <Rank - 1> div/mod operations.
/// Note, is assumed that element divmod[0] divides by extent[1].
///
/// For example, assume 4-D coordinate (n, p, q, c) is mapped to a linear index `npqc`. This
/// can be decomposed via three divide and modulus operations:
///
///      c = npqc % C;         |  divmod[2] = FastDivmodU64(C)
///    npq = npqc / C;         |   coord[3] = c
///
///      q =  npq % Q;         |  divmod[1] = FastDivmodU64(Q)
///     np =  npq / Q;         |   coord[2] = q
///
///      p =   np % P;         |  divmod[0] = FastDivmodU64(P)
///      n =   np / P;         |   coord[1] = p
///
///                            |   coord[0] = n
///
template <size_t Rank>
TV_HOST_DEVICE_INLINE array<int, Rank> CoordinateDecomposition(
  uint64_t linear_idx,                    ///< Linear index to decompose
  FastDivmodU64 const *divmod) {          ///< Pointer to array of Rank-1 FastDivmodU64 objects

  static_assert(Rank > 0, "CoordinateDecomposition requires Rank=1 or greater.");

  array<int, Rank> coord;

  TV_PRAGMA_UNROLL
  for (int i = Rank; i > 1; --i) {
    uint64_t remainder;
    linear_idx = divmod[i - 2].divmod(remainder, linear_idx);
    coord[i - 1] = int(remainder);
  }

  coord[0] = int(linear_idx);

  return coord;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Min/Max
/////////////////////////////////////////////////////////////////////////////////////////////////

template <int A, int B>
struct Min {
  static int const kValue = (A < B) ? A : B;
};

template <int A, int B>
struct Max {
  static int const kValue = (A > B) ? A : B;
};

TV_HOST_DEVICE_INLINE
constexpr int const_min(int a, int b) {
    return (b < a ? b : a);
}

TV_HOST_DEVICE_INLINE
constexpr int const_max(int a, int b) {
    return (b > a ? b : a);
}

template <typename T>
TV_HOST_DEVICE_INLINE
T fast_min(T a, T b) {
  return (b < a ? b : a);
}

template <>
TV_HOST_DEVICE_INLINE
float fast_min(float a, float b) {
  return fminf(a, b);
}

template <typename T>
TV_HOST_DEVICE_INLINE
T fast_max(T a, T b) {
  return (a < b ? b : a);
}

template <>
TV_HOST_DEVICE_INLINE
float fast_max(float a, float b) {
  return fmaxf(a, b);
}

TV_HOST_DEVICE_INLINE
float fast_cos(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::cosf(theta);
  #else
  return std::cos(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_cos(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::cos(theta);
  #else
  return std::cos(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_sin(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::sinf(theta);
  #else
  return std::sin(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_sin(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::sin(theta);
  #else
  return std::sin(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_acos(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::acosf(theta);
  #else
  return std::acos(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_acos(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::acos(theta);
  #else
  return std::acos(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_asin(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::asinf(theta);
  #else
  return std::asin(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_asin(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::asin(theta);
  #else
  return std::asin(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_sqrt(float theta) {
  #if defined(__CUDA_ARCH__)
  return ::sqrtf(theta);
  #else
  return std::sqrt(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_sqrt(double theta) {
  #if defined(__CUDA_ARCH__)
  return ::sqrt(theta);
  #else
  return std::sqrt(theta);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_log(float x) {
  #if defined(__CUDA_ARCH__)
  return ::logf(x);
  #else
  return std::log(x);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_log(double x) {
  #if defined(__CUDA_ARCH__)
  return ::log(x);
  #else
  return std::log(x);
  #endif
}

TV_HOST_DEVICE_INLINE
float fast_tanh(float x) {
  #if defined(__CUDA_ARCH__)
  return ::tanhf(x);
  #else
  return std::tanh(x);
  #endif
}

TV_HOST_DEVICE_INLINE
double fast_tanh(double x) {
  #if defined(__CUDA_ARCH__)
  return ::tanh(x);
  #else
  return std::tanh(x);
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

