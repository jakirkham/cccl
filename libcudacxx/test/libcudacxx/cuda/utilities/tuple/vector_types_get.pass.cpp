//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 0)
__NV_SILENCE_DEPRECATION_BEGIN
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <class VType, class BaseType, size_t VSize>
struct get_val;

template <class VType, class BaseType>
struct get_val<VType, BaseType, 1>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 2>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42), static_cast<BaseType>(1337)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 3>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{static_cast<BaseType>(42), static_cast<BaseType>(1337), static_cast<BaseType>(-1)};
  }
};
template <class VType, class BaseType>
struct get_val<VType, BaseType, 4>
{
  __host__ __device__ static constexpr VType create()
  {
    return VType{
      static_cast<BaseType>(42), static_cast<BaseType>(1337), static_cast<BaseType>(-1), static_cast<BaseType>(0)};
  }
};

template <class BaseType, size_t Size>
struct get_expected;

template <class BaseType>
struct get_expected<BaseType, 0>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(42)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 1>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(1337)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 2>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(-1)};
  }
};
template <class BaseType>
struct get_expected<BaseType, 3>
{
  __host__ __device__ static constexpr BaseType create()
  {
    return BaseType{static_cast<BaseType>(0)};
  }
};

template <class VType, class BaseType, size_t VSize, size_t Index, cuda::std::enable_if_t<(Index < VSize), int> = 0>
__host__ __device__ constexpr void test()
{
  { // & overload
    VType val  = get_val<VType, BaseType, VSize>::create();
    auto&& ret = cuda::std::get<Index>(val);
    //    static_assert(cuda::std::is_same<decltype(ret), BaseType&>::value, "");

    const BaseType expected = get_expected<BaseType, Index>::create();
    assert(ret == expected);
  }

  { // const& overload
    const VType val = get_val<const VType, BaseType, VSize>::create();
    auto&& ret      = cuda::std::get<Index>(val);
    // static_assert(cuda::std::is_same<decltype(ret), const BaseType&>::value,"");

    const BaseType expected = get_expected<BaseType, Index>::create();
    assert(ret == expected);
  }

  { // && overload
    VType val  = get_val<VType, BaseType, VSize>::create();
    auto&& ret = cuda::std::get<Index>(cuda::std::move(val));
    //    static_assert(cuda::std::is_same<decltype(ret), BaseType&&>::value, "");

    const BaseType expected = get_expected<BaseType, Index>::create();
    assert(ret == expected);
  }

  { // const && overload
    const VType val = get_val<const VType, BaseType, VSize>::create();
    auto&& ret      = cuda::std::get<Index>(cuda::std::move(val));
    //    static_assert(cuda::std::is_same<decltype(ret), const BaseType&&>::value,"");

    const BaseType expected = get_expected<const BaseType, Index>::create();
    assert(ret == expected);
  }
}

template <class VType, class BaseType, size_t VSize, size_t Index, cuda::std::enable_if_t<(Index >= VSize), int> = 0>
__host__ __device__ constexpr void test()
{}

template <class VType, class BaseType, size_t VSize>
__host__ __device__ constexpr void test()
{
  test<VType, BaseType, VSize, 0>();
  test<VType, BaseType, VSize, 1>();
  test<VType, BaseType, VSize, 2>();
  test<VType, BaseType, VSize, 3>();
}

#define EXPAND_VECTOR_TYPE(Type, BaseType) \
  test<Type##1, BaseType, 1>();            \
  test<Type##2, BaseType, 2>();            \
  test<Type##3, BaseType, 3>();            \
  test<Type##4, BaseType, 4>();

__host__ __device__ constexpr bool test()
{
  EXPAND_VECTOR_TYPE(char, signed char);
  EXPAND_VECTOR_TYPE(uchar, unsigned char);
  EXPAND_VECTOR_TYPE(short, short);
  EXPAND_VECTOR_TYPE(ushort, unsigned short);
  EXPAND_VECTOR_TYPE(int, int);
  EXPAND_VECTOR_TYPE(uint, unsigned int);
  EXPAND_VECTOR_TYPE(long, long);
  EXPAND_VECTOR_TYPE(ulong, unsigned long);
  EXPAND_VECTOR_TYPE(longlong, long long);
  EXPAND_VECTOR_TYPE(ulonglong, unsigned long long);
  EXPAND_VECTOR_TYPE(float, float);
  EXPAND_VECTOR_TYPE(double, double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, long, 4>();
  test<long4_32a, long, 4>();
  test<ulong4_16a, unsigned long, 4>();
  test<ulong4_32a, unsigned long, 4>();
  test<longlong4_16a, long long, 4>();
  test<longlong4_32a, long long, 4>();
  test<ulonglong4_16a, unsigned long long, 4>();
  test<ulonglong4_32a, unsigned long long, 4>();
  test<double4_16a, double, 4>();
  test<double4_32a, double, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  return true;
}

__host__ __device__
#if !TEST_COMPILER(MSVC)
  constexpr
#endif // !TEST_COMPILER(MSVC)
  bool
  test_dim3()
{
  test<dim3, unsigned int, 3, 0>();
  test<dim3, unsigned int, 3, 1>();
  test<dim3, unsigned int, 3, 2>();
  return true;
}

int main(int arg, char** argv)
{
  test();
  test_dim3();
  static_assert(test(), "");
#if !TEST_COMPILER(MSVC)
  static_assert(test_dim3(), "");
#endif // !TEST_COMPILER(MSVC)

  return 0;
}
