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
/*! 
  \file
  \brief Defines an unsigned 128b integer with several operators to support 64-bit integer division.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace tv {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Optionally enable GCC's built-in type
#if defined(__x86_64) && !defined(__CUDA_ARCH__)
#if defined(__GNUC__)
#define TV_UINT128_NATIVE
#elif defined(_MSC_VER)
#define CUTLASS_INT128_ARITHMETIC
#include <intrin.h>
#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

///! Unsigned 128b integer type
struct uint128_t {

  /// Size of one part of the uint's storage in bits
  int const kPartSize = sizeof(uint64_t) * 8;

  // Use a union to store either low and high parts or, if present, a built-in 128b integer type.
  union {

    struct {
      uint64_t lo;
      uint64_t hi;
    };

    #if defined(TV_UINT128_NATIVE)
    unsigned __int128 native;
    #endif // defined(TV_UINT128_NATIVE)
  };

  //
  // Methods
  //

  /// Default ctor
  TV_HOST_DEVICE_INLINE
  uint128_t(): lo(0), hi(0) { }

  /// Constructor from uint64
  TV_HOST_DEVICE_INLINE
  uint128_t(uint64_t lo_): lo(lo_), hi(0) { }

  /// Constructor from two 64b unsigned integers
  TV_HOST_DEVICE_INLINE
  uint128_t(uint64_t lo_, uint64_t hi_): lo(lo_), hi(hi_) {

  }

  /// Optional constructor from native value
  #if defined(TV_UINT128_NATIVE)
  uint128_t(unsigned __int128 value): native(value) { }
  #endif

  /// Lossily cast to uint64
  TV_HOST_DEVICE_INLINE
  explicit operator uint64_t() const {
    return lo;
  }

  TV_HOST_DEVICE_INLINE
  static void exception() {
#if defined(__CUDA_ARCH__)
  asm volatile ("  brkpt;\n");
#else
  throw std::runtime_error("Not yet implemented.");
#endif
  }

  /// Add
  TV_HOST_DEVICE_INLINE
  uint128_t operator+(uint128_t const &rhs) const {
    uint128_t y;
#if defined(TV_UINT128_NATIVE)
    y.native = native + rhs.native;
#else
    y.lo = lo + rhs.lo;
    y.hi = hi + rhs.hi + (!y.lo && (rhs.lo));
#endif
    return y;
  }

  /// Subtract
  TV_HOST_DEVICE_INLINE
  uint128_t operator-(uint128_t const &rhs) const {
    uint128_t y;
#if defined(TV_UINT128_NATIVE)
    y.native = native - rhs.native;
#else
    y.lo = lo - rhs.lo;
    y.hi = hi - rhs.hi - (rhs.lo && y.lo > lo);
#endif
    return y;
  }

  /// Multiply by unsigned 64b integer yielding 128b integer
  TV_HOST_DEVICE_INLINE
  uint128_t operator*(uint64_t const &rhs) const {
    uint128_t y;
#if defined(TV_UINT128_NATIVE)
    y.native = native * rhs;
#elif defined(CUTLASS_INT128_ARITHMETIC)
    // Multiply by the low part
    y.lo = _umul128(lo, rhs, &y.hi);

    // Add the high part and ignore the overflow
    uint64_t overflow;
    y.hi += _umul128(hi, rhs, &overflow);
#else
    // TODO - not implemented
    exception();
#endif
    return y;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  TV_HOST_DEVICE_INLINE
  uint64_t operator/(uint64_t const &divisor) const {
    uint64_t quotient = 0;
#if defined(TV_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    uint64_t remainder = 0;
    quotient = _udiv128(hi, lo, divisor, &remainder);
#else
    // TODO - not implemented
    exception();
#endif
    return quotient;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  TV_HOST_DEVICE_INLINE
  uint64_t operator%(uint64_t const &divisor) const {
    uint64_t remainder = 0;
#if defined(TV_UINT128_NATIVE)
    remainder = uint64_t(native % divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    (void)_udiv128(hi, lo, divisor, &remainder);
#else
    // TODO - not implemented
    exception();
#endif
    return remainder;
  }

  /// Computes the quotient and remainder in a single method.
  TV_HOST_DEVICE_INLINE
  uint64_t divmod(uint64_t &remainder, uint64_t divisor) const {
    uint64_t quotient = 0;
#if defined(TV_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
    remainder = uint64_t(native % divisor);
#elif defined(CUTLASS_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    quotient = _udiv128(hi, lo, divisor, &remainder);
#else
    // TODO - not implemented
    exception();
#endif
    return quotient;
  }

  /// Left-shifts a 128b unsigned integer
  TV_HOST_DEVICE_INLINE
  uint128_t operator<<(int sh) const {
    if (sh == 0) {
      return *this;
    }
    else if (sh >= kPartSize) {
      return uint128_t(0, lo << (sh - kPartSize));
    }
    else {
      return uint128_t(
        (lo << sh),
        (hi << sh) | uint64_t(lo >> (kPartSize - sh))
      );
    }
  }

  /// Right-shifts a 128b unsigned integer
  TV_HOST_DEVICE_INLINE
  uint128_t operator>>(int sh) const {
    if (sh == 0) {
      return *this;
    }
    else if (sh >= kPartSize) {
      return uint128_t((hi >> (sh - kPartSize)), 0);
    }
    else {
      return uint128_t(
        (lo >> sh) | (hi << (kPartSize - sh)),
        (hi >> sh)
      );
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
