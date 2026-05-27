"""Iterate proprioception demos from an on-disk RLDS cache."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from .utils import DEFAULT_OBSERVATION_KEYS

if TYPE_CHECKING:
    from .loader import RldsObservationLoader

ProprioDemo = tuple[np.ndarray, np.ndarray, np.ndarray]


def iter_proprio_demos(
    loader: RldsObservationLoader,
    *,
    start_demo: int = 0,
    end_demo: int | None = None,
) -> Iterator[ProprioDemo]:
    """
    Yield ``(joint_position, gripper_position, cartesian_position)`` per demo.

    Arrays are float64: joints ``(T, 7)``, gripper ``(T,)``, cartesian ``(T, 6)``.
    """
    missing = set(DEFAULT_OBSERVATION_KEYS) - set(loader.observation_keys)
    if missing:
        raise KeyError(
            f"Cache at {loader.data_dir} is missing DROID-style keys {sorted(missing)}. "
            f"Available: {list(loader.observation_keys)}. "
            "Re-download with the DROID adapter enabled."
        )

    n_demos = len(loader)
    stop = n_demos if end_demo is None else min(end_demo, n_demos)
    if start_demo < 0 or start_demo >= stop:
        raise ValueError(f"Invalid demo range [{start_demo}, {stop}) for {n_demos} demos")

    for demo_id in range(start_demo, stop):
        demo = loader.get_demo(demo_id)
        joint_positions = np.asarray(demo["joint_position"], dtype=np.float64)
        gripper_positions = np.asarray(demo["gripper_position"], dtype=np.float64).reshape(-1)
        cartesian_positions = np.asarray(demo["cartesian_position"], dtype=np.float64)
        yield joint_positions, gripper_positions, cartesian_positions
