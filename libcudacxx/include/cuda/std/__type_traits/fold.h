//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_FOLD_H
#define _LIBCUDACXX___TYPE_TRAITS_FOLD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/negation.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !defined(_CCCL_NO_FOLD_EXPRESSIONS)

// Use fold expressions when possible to implement __fold_and[_v] and
// __fold_or[_v].
template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_and_v = (_Preds && ...);

template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_or_v = (_Preds || ...);

template <bool... _Preds>
using __fold_and = bool_constant<bool((_Preds && ...))>; // cast to bool to avoid error from gcc < 8

template <bool... _Preds>
using __fold_or = bool_constant<bool((_Preds || ...))>; // cast to bool to avoid error from gcc < 8

#else // ^^^ !_CCCL_NO_FOLD_EXPRESSIONS / _CCCL_NO_FOLD_EXPRESSIONS vvv

// Otherwise, we can use a helper class to implement __fold_and and __fold_or.
template <bool... _Preds>
struct __fold_helper;

template <bool... _Preds>
using __fold_and = _IsSame<__fold_helper<true, _Preds...>, __fold_helper<_Preds..., true>>;

template <bool... _Preds>
using __fold_or = _Not<_IsSame<__fold_helper<false, _Preds...>, __fold_helper<_Preds..., false>>>;

#  if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_and_v = __fold_and<_Preds...>::value;

template <bool... _Preds>
_CCCL_INLINE_VAR constexpr bool __fold_or_v = __fold_or<_Preds...>::value;
#  endif // !_CCCL_NO_VARIABLE_TEMPLATES

#endif // _CCCL_NO_FOLD_EXPRESSIONS

#if _CCCL_STD_VER >= 2017
#  define _CCCL_FOLD_AND(__pred)           (__pred && ... && true)
#  define _CCCL_FOLD_OR(__pred)            (__pred || ...)
#  define _CCCL_FOLD_PLUS(__init, __args)  (__args + ... + __init)
#  define _CCCL_FOLD_TIMES(__init, __args) (__args * ... * __init)

#else // ^^^ _CCCL_STD_VER >= 2017 ^^^ / vvv _CCCL_STD_VER <= 2014 vvv

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __variadic_sum(_Tp __init) noexcept
{
  return __init;
}

template <class _Tp, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __variadic_sum(_Tp __init, _Args... __args)
{
  const _Tp __arr[sizeof...(_Args)] = {static_cast<_Tp>(__args)...};
  for (size_t __i = 0; __i < sizeof...(_Args); ++__i)
  {
    __init += __arr[__i];
  }
  return __init;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __variadic_times(_Tp __init) noexcept
{
  return __init;
}

template <class _Tp, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __variadic_times(_Tp __init, _Args... __args)
{
  const _Tp __arr[sizeof...(_Args)] = {static_cast<_Tp>(__args)...};
  for (size_t __i = 0; __i < sizeof...(_Args); ++__i)
  {
    __init *= __arr[__i];
  }
  return __init;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __variadic_and() noexcept
{
  return true;
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __variadic_and(_Args... __args)
{
  const bool __arr[sizeof...(_Args)] = {static_cast<bool>(__args)...};
  for (size_t __i = 0; __i < sizeof...(_Args); ++__i)
  {
    if (!__arr[__i])
    {
      return false;
    }
  }
  return true;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __variadic_or() noexcept
{
  return false;
}

template <class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __variadic_or(_Args... __args)
{
  const bool __arr[sizeof...(_Args)] = {static_cast<bool>(__args)...};
  for (size_t __i = 0; __i < sizeof...(_Args); ++__i)
  {
    if (__arr[__i])
    {
      return true;
    }
  }
  return false;
}

#  define _CCCL_FOLD_AND(__pred)           _CUDA_VSTD::__variadic_and(__pred...)
#  define _CCCL_FOLD_OR(__pred)            _CUDA_VSTD::__variadic_or(__pred...)
#  define _CCCL_FOLD_PLUS(__init, __args)  _CUDA_VSTD::__variadic_sum(__init, __args...)
#  define _CCCL_FOLD_TIMES(__init, __args) _CUDA_VSTD::__variadic_times(__init, __args...)

#endif // _CCCL_STD_VER <= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_FOLD_H
