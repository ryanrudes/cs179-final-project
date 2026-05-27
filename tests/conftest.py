import importlib.util

import pytest


def _native_module():
    spec = importlib.util.find_spec("cs179._native")
    if spec is None:
        return None
    from cs179 import _native

    return _native


@pytest.fixture(scope="session")
def native_module():
    return _native_module()


@pytest.fixture(scope="session")
def has_native_loader(native_module):
    return native_module is not None and hasattr(native_module, "RldsObservationLoader")


@pytest.fixture(scope="session")
def has_cuda_vector_add(native_module):
    return native_module is not None and hasattr(native_module, "vector_add")
