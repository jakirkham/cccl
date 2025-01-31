# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import cupy as cp
import numba.cuda
import numba.types
import numpy as np

import cuda.parallel.experimental.algorithms as algorithms
import cuda.parallel.experimental.iterators as iterators


def exclusive_scan_host(h_input: np.ndarray, op, h_init=0):
    result = h_input.copy()
    result[0] = h_init[0]
    for i in range(1, len(result)):
        result[i] = op(result[i - 1], h_input[i - 1])
    return result


def exclusive_scan_device(d_input, d_output, num_items, op, h_init):
    scan = algorithms.scan(d_input, d_output, op, h_init)
    temp_storage_size = scan(None, d_input, d_output, num_items, h_init)
    d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=np.uint8)

    scan(d_temp_storage, d_input, d_output, num_items, h_init)


def test_scan_array_input(input_array):
    def op(a, b):
        return a + b

    d_input = input_array
    dtype = d_input.dtype
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(len(d_input), dtype=dtype)

    exclusive_scan_device(d_input, d_output, len(d_input), op, h_init)

    got = d_output.get()
    expected = exclusive_scan_host(d_input.get(), op, h_init)

    np.testing.assert_allclose(expected, got)


def test_scan_iterator_input():
    def op(a, b):
        return a + b

    d_input = iterators.CountingIterator(np.int32(1))
    num_items = 1024
    dtype = np.dtype("int32")
    h_init = np.array([42], dtype=dtype)
    d_output = cp.empty(num_items, dtype=dtype)

    exclusive_scan_device(d_input, d_output, num_items, op, h_init)

    got = d_output.get()
    expected = exclusive_scan_host(np.arange(1, num_items + 1, dtype=dtype), op, h_init)

    np.testing.assert_allclose(expected, got)
