# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
import numba.types
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms


def exclusive_scan(inp: np.ndarray, op, init=0):
    result = inp.copy()
    result[0] = init
    for i in range(1, len(result)):
        result[i] = op(result[i - 1], inp[i - 1])
    return result


def scan_test_helper(d_input, d_output, num_items, op, h_init):
    scan = algorithms.scan(d_input, d_output, op, h_init)
    temp_storage_size = scan(None, d_input, d_output, None, h_init)
    d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=np.uint8)

    scan(d_temp_storage, d_input, d_output, None, h_init)

    expected = exclusive_scan(d_input.get(), op, init=h_init)
    got = d_output.get()
    np.testing.assert_allclose(expected, got)


def test_device_scan(input_array):
    def op(a, b):
        return a + b

    d_input = input_array
    dtype = d_input.dtype
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(len(d_input), dtype=dtype)
    scan_test_helper(d_input, d_output, len(d_input), op, h_init)
