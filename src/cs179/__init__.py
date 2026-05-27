"""CS179 CUDA + Python project."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from cs179 import _native

__all__ = ["vector_add"]


def vector_add(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    out: NDArray[np.floating],
) -> None:
    """Add two 1-D float arrays on the GPU, writing the result into `out`."""
    from cs179 import _native

    if not hasattr(_native, "vector_add"):
        raise RuntimeError(
            "CUDA was not built into _native. Install nvcc and run ./scripts/build_native.sh, "
            "or use ./scripts/build_loader.sh for RLDS-only builds."
        )

    _native.vector_add(
        np.ascontiguousarray(a, dtype=np.float32),
        np.ascontiguousarray(b, dtype=np.float32),
        np.ascontiguousarray(out, dtype=np.float32),
    )
