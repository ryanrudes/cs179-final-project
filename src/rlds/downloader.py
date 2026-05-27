import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from rich.progress import (
    FileSizeColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TimeElapsedColumn,
)
from rich.table import Column
from rich.text import Text

from .adapters import RldsDatasetAdapter, require_adapter
from .datasets import RoboticsRldsDatasetUrl
from .timing import resolve_control_hz
from .utils import (
    DEFAULT_OBSERVATION_KEYS,
    DEFAULT_SHARD_SIZE_MB,
    build_observation_decoders,
    iter_episode_step_batches,
    metadata_dir,
    normalize_dataset_url,
    observation_shard_path,
    resolve_data_dir,
    shard_size_mb_to_step_capacity,
    step_size_bytes,
    trim_observation_shard,
    validate_demo_field_shapes,
    validate_demo_lengths,
)

import tensorflow as tf
import tensorflow_datasets as tfds


class TimestepsColumn(ProgressColumn):
    """Shows total timesteps processed (via ``progress.update(..., timesteps=...)``)."""

    def __init__(self, table_column: Column | None = None) -> None:
        super().__init__(
            table_column=table_column or Column(no_wrap=True, justify="right"),
        )

    def render(self, task: Task) -> Text:
        timesteps = task.fields.get("timesteps")
        if timesteps is None:
            return Text("")
        return Text(f"{int(timesteps):,} timesteps", style="progress.description")


class BytesWrittenColumn(FileSizeColumn):
    """``FileSizeColumn`` backed by ``bytes_written`` in task fields (not ``completed``)."""

    def __init__(self) -> None:
        super().__init__(table_column=Column(no_wrap=True, justify="right"))

    def render(self, task: Task) -> Text:
        nbytes = task.fields.get("bytes_written")
        if nbytes is None:
            return Text("")
        return super().render(replace(task, completed=float(nbytes)))


class RldsObservationDownloader:
    def __init__(
        self,
        dataset_url: str | RoboticsRldsDatasetUrl = RoboticsRldsDatasetUrl.DROID_100,
        data_dir: str | Path = "data",
        shard_size_mb: int | float = DEFAULT_SHARD_SIZE_MB,
        observation_keys: tuple[str, ...] | None = None,
        trim_partial_shards: bool = True,
        use_adapter: bool = True,
        adapter: RldsDatasetAdapter | None = None,
        control_hz: float | None = None,
        max_demos: int | None = None,
    ):
        self.dataset_url = normalize_dataset_url(dataset_url)
        self.data_dir = resolve_data_dir(data_dir, self.dataset_url)
        self.shard_size_mb = shard_size_mb
        self.trim_partial_shards = trim_partial_shards
        self.use_adapter = use_adapter

        parsed_url = dataset_url if isinstance(dataset_url, RoboticsRldsDatasetUrl) else None
        if use_adapter:
            self._adapter = adapter if adapter is not None else require_adapter(
                parsed_url if parsed_url is not None else dataset_url
            )
            self.source_keys = self._adapter.source_keys
            self.observation_keys = self._adapter.output_keys
            self.field_shapes: dict[str, tuple[int, ...]] | None = dict(
                self._adapter.output_field_shapes
            )
        else:
            self._adapter = None
            self.observation_keys = (
                observation_keys
                if observation_keys is not None
                else DEFAULT_OBSERVATION_KEYS
            )
            self.source_keys = self.observation_keys
            self.field_shapes = None

        resolve_url = parsed_url if parsed_url is not None else self.dataset_url
        if use_adapter and self._adapter is not None and control_hz is None:
            self.control_hz = self._adapter.resolved_control_hz
        else:
            self.control_hz = resolve_control_hz(resolve_url, override=control_hz)

        if max_demos is not None and max_demos <= 0:
            raise ValueError(f"max_demos must be positive, got {max_demos}")
        self.max_demos = max_demos

        self.shard_step_capacity: int | None = None

        self.shard_id = 0
        self.shard_offset = 0
        self.total_steps = 0
        self.shard_lengths: list[int] = []
        self.current_shard: dict[str, np.ndarray] | None = None

    def download(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            TimestepsColumn(),
            BytesWrittenColumn(),
        )

        with progress:
            fetch_task = progress.add_task("Fetching dataset", total=None)
            builder = tfds.builder_from_directory(self.dataset_url)
            decoders = build_observation_decoders(builder, self.source_keys)
            split = f"train[:{self.max_demos}]" if self.max_demos is not None else "train"
            dataset = builder.as_dataset(split=split, decoders=decoders)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            progress.update(fetch_task, total=1, completed=1)

            enumerate_task = progress.add_task("Counting demos", total=1)
            num_demos = len(dataset)
            progress.update(
                enumerate_task,
                completed=1,
                description=f"Counted {num_demos:,} demos",
            )

            if num_demos == 0:
                raise ValueError(f"Dataset has no episodes: {self.dataset_url}")

            demo_lengths = np.empty(num_demos, dtype=np.int32)
            download_task = progress.add_task("Downloading data", unit="demo", total=num_demos)

            for i, episode in enumerate(dataset):
                demo_lengths[i] = self._stream_episode(episode, progress, download_task)
                progress.advance(download_task)

            finalize_task = progress.add_task("Finalizing shards", total=3)
            self._finalize_current_shard()
            progress.advance(finalize_task)

            demo_offsets = np.concatenate(([0], np.cumsum(demo_lengths, dtype=np.int64)))
            shard_lengths = np.asarray(self.shard_lengths, dtype=np.int64)
            shard_offsets = np.concatenate(([0], np.cumsum(shard_lengths, dtype=np.int64)))
            progress.advance(finalize_task)

            out_metadata_dir = metadata_dir(self.data_dir)
            out_metadata_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_metadata_dir / "demo_lengths.npy", demo_lengths)
            np.save(out_metadata_dir / "demo_offsets.npy", demo_offsets)
            np.save(out_metadata_dir / "shard_lengths.npy", shard_lengths)
            np.save(out_metadata_dir / "shard_offsets.npy", shard_offsets)
            np.save(out_metadata_dir / "total_steps.npy", np.asarray(self.total_steps, dtype=np.int64))
            assert self.field_shapes is not None
            metadata = {
                "dataset_url": self.dataset_url,
                "control_hz": self.control_hz,
                "max_demos": self.max_demos,
                "observation_keys": list(self.observation_keys),
                "field_shapes": {
                    key: list(shape)
                    for key, shape in self.field_shapes.items()
                },
                "bytes_per_step": step_size_bytes(self.field_shapes),
                "shard_size_mb": self.shard_size_mb,
                "shard_step_capacity": self.shard_step_capacity,
                "trim_partial_shards": self.trim_partial_shards,
                "use_adapter": self.use_adapter,
            }
            if self._adapter is not None:
                metadata["adapter"] = self._adapter.name
                metadata["adapter_source_keys"] = list(self.source_keys)
                metadata["adapter_metadata"] = dict(self._adapter.metadata)
            with open(out_metadata_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            progress.advance(finalize_task)

    def _create_shard(self, shard_id: int) -> dict[str, np.ndarray]:
        if self.field_shapes is None or self.shard_step_capacity is None:
            raise RuntimeError("field_shapes and shard_step_capacity must be initialized first")
        shard_arrays = {}
        for key in self.observation_keys:
            path = observation_shard_path(self.data_dir, key, shard_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            shard_arrays[key] = np.lib.format.open_memmap(
                path,
                dtype=np.float32,
                mode="w+",
                shape=(self.shard_step_capacity, *self.field_shapes[key]),
            )
        return shard_arrays

    def _finalize_current_shard(self) -> None:
        if self.current_shard is None:
            return

        used = self.shard_offset

        if used == 0:
            self._delete_current_empty_shard()
            self.current_shard = None
            return

        for array in self.current_shard.values():
            array.flush()

        shard_id = self.shard_id
        del self.current_shard
        self.current_shard = None
        self.shard_lengths.append(used)

        assert self.shard_step_capacity is not None
        if self.trim_partial_shards and used < self.shard_step_capacity:
            for key in self.observation_keys:
                trim_observation_shard(
                    observation_shard_path(self.data_dir, key, shard_id),
                    used,
                )

    def _delete_current_empty_shard(self) -> None:
        del self.current_shard
        for key in self.observation_keys:
            observation_shard_path(self.data_dir, key, self.shard_id).unlink(missing_ok=True)

    def _stream_episode(self, episode, progress: Progress, download_task) -> int:
        demo_steps = 0
        for batch in iter_episode_step_batches(episode, self.source_keys):
            raw_block = {
                key: batch[key].astype(np.float32, copy=False)
                for key in self.source_keys
            }
            step_block = (
                self._adapter.adapt_batch(raw_block)
                if self._adapter is not None
                else raw_block
            )
            if self.shard_step_capacity is None:
                if self.field_shapes is None:
                    self.field_shapes = {
                        key: tuple(value.shape[1:])
                        for key, value in step_block.items()
                    }
                self.shard_step_capacity = shard_size_mb_to_step_capacity(
                    self.shard_size_mb, self.field_shapes
                )
                self.current_shard = self._create_shard(self.shard_id)
            else:
                validate_demo_field_shapes(step_block, self.field_shapes)

            batch_steps = validate_demo_lengths(step_block)
            self._append_steps(step_block)
            demo_steps += batch_steps
            self._update_download_progress(progress, download_task)

        if demo_steps == 0:
            raise ValueError("Episode has zero steps")
        return demo_steps

    def _update_download_progress(self, progress: Progress, download_task) -> None:
        bytes_written = (
            self.total_steps * step_size_bytes(self.field_shapes)
            if self.field_shapes is not None
            else 0
        )
        progress.update(
            download_task,
            timesteps=self.total_steps,
            bytes_written=bytes_written,
        )

    def _append_steps(self, step_block: dict[str, np.ndarray]) -> None:
        remaining = step_block[self.observation_keys[0]].shape[0]
        source_offset = 0

        while remaining > 0:
            assert self.shard_step_capacity is not None
            available = self.shard_step_capacity - self.shard_offset
            if available == 0:
                self._finalize_current_shard()
                self.shard_id += 1
                self.shard_offset = 0
                self.current_shard = self._create_shard(self.shard_id)
                available = self.shard_step_capacity

            take = min(remaining, available)
            shard_slice = slice(self.shard_offset, self.shard_offset + take)
            source_slice = slice(source_offset, source_offset + take)

            assert self.current_shard is not None
            for key in self.observation_keys:
                self.current_shard[key][shard_slice] = step_block[key][source_slice]

            self.shard_offset += take
            source_offset += take
            remaining -= take
            self.total_steps += take
