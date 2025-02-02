# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations  # TODO: required for Python 3.7 docs env

import ctypes
from typing import Callable

import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import get_bindings, get_paths
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike, GpuStruct


class _Scan:
    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: Callable,
        h_init: np.ndarray | GpuStruct,
    ):
        # Referenced from __del__:
        self.build_result = None

        d_in_cccl = cccl.to_cccl_iter(d_in)
        d_out_cccl = cccl.to_cccl_iter(d_out)
        cc_major, cc_minor = cuda.get_current_device().compute_capability
        cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_paths()
        bindings = get_bindings()
        if isinstance(h_init, np.ndarray):
            value_type = numba.from_dtype(h_init.dtype)
        else:
            value_type = numba.typeof(h_init)
        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)
        self.build_result = cccl.DeviceScanBuildResult()

        error = bindings.cccl_device_scan_build(
            ctypes.byref(self.build_result),
            d_in_cccl,
            d_out_cccl,
            self.op_wrapper,
            cccl.to_cccl_value(h_init),
            cc_major,
            cc_minor,
            ctypes.c_char_p(cub_path),
            ctypes.c_char_p(thrust_path),
            ctypes.c_char_p(libcudacxx_path),
            ctypes.c_char_p(cuda_include_path),
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building scan")

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        h_init: np.ndarray | GpuStruct,
        stream=None,
    ):
        d_in_cccl = cccl.to_cccl_iter(d_in)
        d_out_cccl = cccl.to_cccl_iter(d_out)
        if d_in_cccl.type.value == cccl.IteratorKind.ITERATOR:
            assert num_items is not None
        else:
            assert d_in_cccl.type.value == cccl.IteratorKind.POINTER
            if num_items is None:
                num_items = d_in.size
            else:
                assert num_items == d_in.size
        if d_out_cccl.type.value == cccl.IteratorKind.POINTER:
            assert num_items == d_out.size

        stream_handle = protocols.validate_and_get_stream(stream)
        bindings = get_bindings()
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]
        error = bindings.cccl_device_scan(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            d_in_cccl,
            d_out_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            cccl.to_cccl_value(h_init),
            stream_handle,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error reducing")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        bindings = get_bindings()
        bindings.cccl_device_scan_cleanup(ctypes.byref(self.build_result))


def make_cache_key(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    d_in_key = (
        d_in.kind if isinstance(d_in, IteratorBase) else protocols.get_dtype(d_in)
    )
    d_out_key = (
        d_out.kind if isinstance(d_out, IteratorBase) else protocols.get_dtype(d_out)
    )
    op_key = CachableFunction(op)
    h_init_key = h_init.dtype
    return (d_in_key, d_out_key, op_key, h_init_key)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_key(make_cache_key)
def scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable,
    h_init: np.ndarray,
):
    return _Scan(d_in, d_out, op, h_init)
