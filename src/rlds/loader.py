import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .datasets import RoboticsRldsDatasetUrl
from .utils import (
    dataset_metadata_dir,
    metadata_dir,
    normalize_dataset_url,
    observation_shard_path,
    resolve_data_dir,
)


class RldsObservationLoader:
    def __init__(
        self,
        data_dir: str | Path = "data",
        dataset_url: str | RoboticsRldsDatasetUrl | None = None,
    ):
        data_dir_path = Path(data_dir)
        meta_path = metadata_dir(data_dir_path) / "metadata.json"
        if meta_path.is_file():
            self.data_dir = data_dir_path
        else:
            resolved_url = (
                dataset_url
                if dataset_url is not None
                else RoboticsRldsDatasetUrl.DROID_100
            )
            self.data_dir = resolve_data_dir(
                data_dir_path,
                normalize_dataset_url(resolved_url),
            )

        self.metadata_dir = dataset_metadata_dir(self.data_dir)
        if not (self.metadata_dir / "metadata.json").is_file():
            raise FileNotFoundError(
                f"No cache metadata at {self.metadata_dir / 'metadata.json'}. "
                "Run `uv run cs179 download` for this dataset first."
            )
        self.demo_lengths = np.load(self.metadata_dir / "demo_lengths.npy", mmap_mode="r")
        self.demo_offsets = np.load(self.metadata_dir / "demo_offsets.npy", mmap_mode="r")
        self.shard_lengths = np.load(self.metadata_dir / "shard_lengths.npy", mmap_mode="r")
        self.shard_offsets = np.load(self.metadata_dir / "shard_offsets.npy", mmap_mode="r")
        self.total_steps = int(np.load(self.metadata_dir / "total_steps.npy"))

        with open(self.metadata_dir / "metadata.json") as f:
            metadata = json.load(f)
        self.observation_keys = tuple(metadata["observation_keys"])
        self.field_shapes = {
            key: tuple(shape)
            for key, shape in metadata["field_shapes"].items()
        }
        self.dataset_url = metadata["dataset_url"]
        self.control_hz = float(metadata["control_hz"])
        if self.control_hz <= 0:
            raise ValueError(f"Invalid control_hz in metadata: {self.control_hz}")

    @lru_cache(maxsize=64)
    def _open_shard(self, observation_key: str, shard_id: int) -> np.ndarray:
        return np.load(
            observation_shard_path(self.data_dir, observation_key, shard_id),
            mmap_mode="r",
        )

    def __len__(self) -> int:
        return len(self.demo_lengths)

    def get_demo(self, demo_id: int) -> dict[str, np.ndarray]:
        if demo_id < 0:
            demo_id += len(self)
        if demo_id < 0 or demo_id >= len(self):
            raise IndexError(f"demo_id {demo_id} out of range for {len(self)} demos")

        start = int(self.demo_offsets[demo_id])
        end = int(self.demo_offsets[demo_id + 1])
        return self.get_step_range(start, end)

    def get_step_range(self, start: int, end: int) -> dict[str, np.ndarray]:
        if start < 0 or end < start or end > self.total_steps:
            raise IndexError(f"Invalid step range [{start}, {end}) for {self.total_steps} total steps")

        if start == end:
            return {
                key: np.empty((0, *self.field_shapes[key]), dtype=np.float32)
                for key in self.observation_keys
            }

        length = end - start
        output = {
            key: np.empty((length, *self.field_shapes[key]), dtype=np.float32)
            for key in self.observation_keys
        }

        out_offset = 0
        shard_id = int(np.searchsorted(self.shard_offsets, start, side="right") - 1)
        cursor = start

        while cursor < end:
            shard_start = int(self.shard_offsets[shard_id])
            shard_end = int(self.shard_offsets[shard_id + 1])
            take_end = min(end, shard_end)

            local_start = cursor - shard_start
            local_end = take_end - shard_start
            take = take_end - cursor
            out_slice = slice(out_offset, out_offset + take)
            shard_slice = slice(local_start, local_end)

            for key in self.observation_keys:
                output[key][out_slice] = self._open_shard(key, shard_id)[shard_slice]

            cursor = take_end
            out_offset += take
            shard_id += 1

        return output

    def iter_demos(self):
        for demo_id in range(len(self)):
            yield self.get_demo(demo_id)

    def iter_demo_views_or_copies(self):
        for demo_id in range(len(self)):
            demo = self.get_demo_views(demo_id)
            yield demo if demo is not None else self.get_demo(demo_id)

    def get_demo_views(self, demo_id: int) -> dict[str, np.ndarray] | None:
        if demo_id < 0:
            demo_id += len(self)
        if demo_id < 0 or demo_id >= len(self):
            raise IndexError(f"demo_id {demo_id} out of range for {len(self)} demos")

        start = int(self.demo_offsets[demo_id])
        end = int(self.demo_offsets[demo_id + 1])
        shard_id = int(np.searchsorted(self.shard_offsets, start, side="right") - 1)

        if end > int(self.shard_offsets[shard_id + 1]):
            return None

        shard_start = int(self.shard_offsets[shard_id])
        local_start = start - shard_start
        local_end = end - shard_start
        demo_slice = slice(local_start, local_end)

        return {
            key: self._open_shard(key, shard_id)[demo_slice]
            for key in self.observation_keys
        }
