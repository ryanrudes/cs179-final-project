"""Retarget joint cache save/load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retarget.cache import (
    default_retarget_output_dir,
    load_joint_trajectory,
    load_metadata,
    save_joint_trajectory,
    write_metadata,
)


def test_default_output_dir() -> None:
    p = default_retarget_output_dir(Path("data"), "gs://gresearch/robotics/droid_100/1.0.0")
    assert p == Path("data/retargeted/droid_100")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "cache"
    q = np.arange(24, dtype=np.float32).reshape(8, 3)
    from retarget.cache import RetargetDemoRecord

    rel = save_joint_trajectory(out, 3, q)
    write_metadata(
        out,
        dataset_url="gs://gresearch/robotics/droid_100/1.0.0",
        robot_description="ur3e_description",
        use_gpu=True,
        reach_safety=0.9,
        demos=[RetargetDemoRecord(3, 8, rel)],
    )
    loaded = load_joint_trajectory(out, 3)
    assert np.allclose(loaded, q)
    meta = load_metadata(out)
    assert meta.use_gpu is True
    assert meta.joint_file(3).is_file()


def test_metadata_lists_demos(tmp_path: Path) -> None:
    out = tmp_path / "cache"
    save_joint_trajectory(out, 0, np.zeros((4, 6), dtype=np.float32))
    from retarget.cache import RetargetDemoRecord

    write_metadata(
        out,
        dataset_url="DROID_100",
        robot_description="ur3e_description",
        use_gpu=True,
        reach_safety=0.9,
        demos=[RetargetDemoRecord(0, 4, "joints/00000.npy")],
        skipped_gpu_length=[{"demo_idx": 9, "num_frames": 900}],
    )
    payload = json.loads((out / "metadata.json").read_text())
    assert payload["skipped_gpu_length"][0]["demo_idx"] == 9
