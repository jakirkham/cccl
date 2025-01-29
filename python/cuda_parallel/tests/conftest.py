import cupy as cp
import numpy as np
import pytest


# Define a pytest fixture that returns random arrays with different dtypes
@pytest.fixture(
    params=[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        np.complex128,
    ]
)
def input_array(request):
    dtype = request.param

    # Generate random values based on the dtype
    if np.issubdtype(dtype, np.integer):
        # For integer types, use np.random.randint for random integers
        array = cp.random.randint(low=0, high=100, size=10, dtype=dtype)
    elif np.issubdtype(dtype, np.floating):
        # For floating-point types, use np.random.random and cast to the required dtype
        array = cp.random.random(10).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # For complex types, generate random real and imaginary parts
        real_part = cp.random.random(10)
        imag_part = cp.random.random(10)
        array = (real_part + 1j * imag_part).astype(dtype)

    return array
