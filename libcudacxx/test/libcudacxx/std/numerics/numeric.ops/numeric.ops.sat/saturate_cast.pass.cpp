//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11, c++14

// <cuda/std/numeric>

// template<class R, class T>
// constexpr R saturate_cast(T x) noexcept;                     // freestanding

#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>

#include "test_macros.h"

using IMIN = signed char;
using UMIN = signed char;

#if !defined(TEST_HAS_NO_INT128_T)
using IMAX = __int128_t;
using UMAX = __uint128_t;
#else // ^^^ !TEST_HAS_NO_INT128_T ^^^ / vvv TEST_HAS_NO_INT128_T vvv
using IMAX = signed long long;
using UMAX = unsigned long long;
#endif // ^^^ TEST_HAS_NO_INT128_T ^^^

template <class Ret, class T>
__host__ __device__ constexpr bool test_sat_cast(T x, Ret res, int zero_value)
{
  assert(cuda::std::saturate_cast<Ret>(static_cast<T>(zero_value + x)) == res);
  return true;
}

template <class S>
__host__ __device__ constexpr bool test_type(int zero_value)
{
  static_assert(cuda::std::is_integral<S>::value && cuda::std::is_signed<S>::value, "");

  using U = cuda::std::make_unsigned_t<S>;

  constexpr auto small_smax  = cuda::std::numeric_limits<IMIN>::max();
  constexpr auto small_szero = IMIN{0};
  constexpr auto small_smin  = cuda::std::numeric_limits<IMIN>::min();
  constexpr auto small_umax  = cuda::std::numeric_limits<UMIN>::max();
  constexpr auto small_uzero = UMIN{0};

  constexpr auto big_smax  = cuda::std::numeric_limits<IMAX>::max();
  constexpr auto big_szero = IMAX{0};
  constexpr auto big_smin  = cuda::std::numeric_limits<IMAX>::min();
  constexpr auto big_umax  = cuda::std::numeric_limits<UMAX>::max();
  constexpr auto big_uzero = UMAX{0};

  constexpr S smax  = cuda::std::numeric_limits<S>::max();
  constexpr S szero = S{0};
  constexpr S smin  = cuda::std::numeric_limits<S>::min();
  constexpr U umax  = cuda::std::numeric_limits<U>::max();
  constexpr U uzero = U{0};

  // test signed

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(small_smax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(small_smax)), "");
  test_sat_cast<S>(small_smin, static_cast<S>(small_smin), zero_value);
  test_sat_cast<S>(small_szero, szero, zero_value);
  test_sat_cast<S>(small_smax, static_cast<S>(small_smax), zero_value);

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(small_umax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(small_umax)), "");
  test_sat_cast<S>(small_uzero, szero, zero_value);
  test_sat_cast<S>(small_umax, static_cast<S>(std::is_same<S, IMIN>::value ? small_smax : small_umax), zero_value);

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(smax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(smax)), "");
  test_sat_cast<S>(smin, smin, zero_value);
  test_sat_cast<S>(szero, szero, zero_value);
  test_sat_cast<S>(smax, smax, zero_value);

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(umax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(umax)), "");
  test_sat_cast<S>(uzero, szero, zero_value);
  test_sat_cast<S>(umax, smax, zero_value); // saturated

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(big_smax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(big_smax)), "");
  test_sat_cast<S>(big_smin, smin, zero_value); // saturated
  test_sat_cast<S>(big_szero, szero, zero_value);
  test_sat_cast<S>(big_smax, smax, zero_value); // saturated

  ASSERT_SAME_TYPE(S, decltype(cuda::std::saturate_cast<S>(big_umax)));
  static_assert(noexcept(cuda::std::saturate_cast<S>(big_umax)), "");
  test_sat_cast<S>(big_uzero, szero, zero_value);
  test_sat_cast<S>(big_umax, smax, zero_value); // saturated

  // test unsigned

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(small_smax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(small_smax)), "");
  test_sat_cast<U>(small_smin, uzero, zero_value);
  test_sat_cast<U>(small_szero, uzero, zero_value);
  test_sat_cast<U>(small_smax, static_cast<U>(small_smax), zero_value);

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(small_umax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(small_umax)), "");
  test_sat_cast<U>(small_uzero, uzero, zero_value);
  test_sat_cast<U>(small_umax, static_cast<U>(small_umax), zero_value);

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(smax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(smax)), "");
  test_sat_cast<U>(smin, uzero, zero_value);
  test_sat_cast<U>(szero, uzero, zero_value);
  test_sat_cast<U>(smax, static_cast<U>(smax), zero_value);

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(umax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(umax)), "");
  test_sat_cast<U>(uzero, uzero, zero_value);
  test_sat_cast<U>(umax, umax, zero_value);

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(big_smax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(big_smax)), "");
  test_sat_cast<U>(big_smin, uzero, zero_value); // saturated
  test_sat_cast<U>(big_szero, uzero, zero_value);
  test_sat_cast<U>(big_smax, (std::is_same<U, UMAX>::value ? static_cast<U>(smax) : umax), zero_value); // saturated

  ASSERT_SAME_TYPE(U, decltype(cuda::std::saturate_cast<U>(big_umax)));
  static_assert(noexcept(cuda::std::saturate_cast<U>(big_umax)), "");
  test_sat_cast<U>(big_uzero, uzero, zero_value);
  test_sat_cast<U>(big_umax, umax, zero_value); // saturated

  return true;
}

__host__ __device__ constexpr bool test(int zero_value)
{
  test_type<signed char>(zero_value);
  test_type<signed short>(zero_value);
  test_type<signed int>(zero_value);
  test_type<signed long>(zero_value);
  test_type<signed long long>(zero_value);
#ifndef TEST_HAS_NO_INT128_T
  test_type<__int128_t>(zero_value);
#endif // !TEST_HAS_NO_INT128_T

  return true;
}

__global__ void test_global_kernel(int* zero_value)
{
  test(*zero_value);
  static_assert(test(0), "");
}

int main(int, char**)
{
  volatile int zero_value = 0;

  test(zero_value);
  static_assert(test(0), "");

  return 0;
}
