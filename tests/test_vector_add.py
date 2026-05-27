import numpy as np
import pytest


@pytest.fixture(scope="module")
def cuda_vector_add():
    pytest.importorskip("cs179._native")
    from cs179 import _native

    if not hasattr(_native, "vector_add"):
        pytest.skip("_native built without CUDA (vector_add unavailable); use ./scripts/build_native.sh with nvcc")
    return _native.vector_add


def test_vector_add(cuda_vector_add) -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    out = np.zeros_like(a)

    cuda_vector_add(a, b, out)

    np.testing.assert_array_equal(out, np.array([5.0, 7.0, 9.0], dtype=np.float32))


def test_vector_add_python_wrapper(cuda_vector_add) -> None:
    from cs179 import vector_add

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    out = np.zeros_like(a)

    vector_add(a, b, out)

    np.testing.assert_array_equal(out, np.array([5.0, 7.0, 9.0], dtype=np.float32))


def test_vector_add_requires_native_extension(has_cuda_vector_add) -> None:
    if not has_cuda_vector_add:
        pytest.skip("_native built without CUDA")
