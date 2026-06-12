"""On-disk cache for retargeted joint trajectories (replay / SSH visualization)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlds.utils import dataset_name_from_url, normalize_dataset_url

JOINTS_SUBDIR = "joints"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class RetargetDemoRecord:
    demo_idx: int
    num_frames: int
    joint_path: str


@dataclass(frozen=True)
class RetargetOutputIndex:
    output_dir: Path
    dataset_url: str
    dataset_name: str
    robot_description: str
    use_gpu: bool
    reach_safety: float
    demos: tuple[RetargetDemoRecord, ...]

    def joint_file(self, demo_idx: int) -> Path:
        for rec in self.demos:
            if rec.demo_idx == demo_idx:
                return self.output_dir / rec.joint_path
        raise KeyError(f"demo {demo_idx} not in retarget cache")


def default_retarget_output_dir(data_dir: Path, dataset_url: str) -> Path:
    """``data/retargeted/<dataset>/`` when *data_dir* is the RLDS cache root or dataset path."""
    dataset_name = dataset_name_from_url(normalize_dataset_url(dataset_url))
    base = Path(data_dir)
    if base.name == dataset_name:
        return base.parent / "retargeted" / dataset_name
    return base / "retargeted" / dataset_name


def joints_dir(output_dir: Path) -> Path:
    return output_dir / JOINTS_SUBDIR


def joint_demo_path(output_dir: Path, demo_idx: int) -> Path:
    return joints_dir(output_dir) / f"{demo_idx:05d}.npy"


def save_joint_trajectory(output_dir: Path, demo_idx: int, joint_traj: np.ndarray) -> str:
    """Write ``(T, nv)`` joint trajectory; return relative path for metadata."""
    out_path = joint_demo_path(output_dir, demo_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(joint_traj, dtype=np.float32))
    return str(out_path.relative_to(output_dir))


def write_metadata(
    output_dir: Path,
    *,
    dataset_url: str,
    robot_description: str,
    use_gpu: bool,
    reach_safety: float,
    demos: list[RetargetDemoRecord],
    skipped_gpu_length: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_url": normalize_dataset_url(dataset_url),
        "dataset_name": dataset_name_from_url(normalize_dataset_url(dataset_url)),
        "robot_description": robot_description,
        "use_gpu": use_gpu,
        "reach_safety": reach_safety,
        "demos": [
            {
                "demo_idx": d.demo_idx,
                "num_frames": d.num_frames,
                "joint_path": d.joint_path,
            }
            for d in demos
        ],
        "skipped_gpu_length": skipped_gpu_length or [],
    }
    (output_dir / METADATA_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_metadata(output_dir: Path) -> RetargetOutputIndex:
    meta_path = output_dir / METADATA_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"retarget cache metadata not found: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    demos = tuple(
        RetargetDemoRecord(
            demo_idx=int(d["demo_idx"]),
            num_frames=int(d["num_frames"]),
            joint_path=str(d["joint_path"]),
        )
        for d in payload["demos"]
    )
    return RetargetOutputIndex(
        output_dir=output_dir.resolve(),
        dataset_url=str(payload["dataset_url"]),
        dataset_name=str(payload["dataset_name"]),
        robot_description=str(payload["robot_description"]),
        use_gpu=bool(payload["use_gpu"]),
        reach_safety=float(payload["reach_safety"]),
        demos=demos,
    )


def load_joint_trajectory(output_dir: Path, demo_idx: int) -> np.ndarray:
    index = load_metadata(output_dir)
    path = index.joint_file(demo_idx)
    q = np.load(path)
    if q.ndim != 2:
        raise ValueError(f"expected joint trajectory (T, nv), got shape {q.shape} at {path}")
    return q
