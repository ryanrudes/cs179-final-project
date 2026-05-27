import os

# Must be set before TensorFlow is imported. Level 3 silences C++ INFO logs from
# tf.data (e.g. tf_record_dataset_op buffer_size) that otherwise trigger absl's
# "InitializeLog()" stderr warning and corrupt Rich progress output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_LEVEL", "3")

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)

from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .datasets import RoboticsRldsDatasetUrl

DEFAULT_OBSERVATION_KEYS = (
    "joint_position",
    "gripper_position",
    "cartesian_position",
)

DEFAULT_SHARD_SIZE_MB = 256


def _feature_children(feature) -> dict[str, object] | None:
    if isinstance(feature, dict):
        return feature

    if hasattr(feature, "keys") and hasattr(feature, "__getitem__"):
        try:
            return {key: feature[key] for key in feature.keys()}
        except TypeError:
            pass

    for attr in ("feature", "features"):
        child = getattr(feature, attr, None)
        if child is not None:
            children = _feature_children(child)
            if children is not None:
                return children

    return None


def _feature_leaf_paths(feature, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    children = _feature_children(feature)
    if children is None:
        return [prefix]

    paths = []
    for key, child in children.items():
        paths.extend(_feature_leaf_paths(child, prefix + (key,)))
    return paths


def _observation_leaf_paths(features) -> list[tuple[str, ...]]:
    return [
        path for path in _feature_leaf_paths(features)
        if len(path) >= 3 and path[-2] == "observation"
    ]


def _path_should_skip_decode(path: tuple[str, ...], keep_paths: set[tuple[str, ...]]) -> bool:
    if path in keep_paths:
        return False
    if len(path) >= 2 and path[0] == "steps" and path[1] == "observation":
        return True
    if path and path[0] == "steps":
        skip_tokens = frozenset(
            {"image", "images", "depth", "depth_image", "wrist_image", "exterior_image", "video"}
        )
        if any(part in skip_tokens for part in path):
            return True
    return False


def _build_observation_skip_decoders(
    feature,
    keep_paths: set[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
):
    children = _feature_children(feature)
    if children is None:
        if prefix in keep_paths:
            return None
        if _path_should_skip_decode(prefix, keep_paths):
            return tfds.decode.SkipDecoding()
        return None

    decoders = {}
    for key, child in children.items():
        child_prefix = prefix + (key,)
        child_decoder = _build_observation_skip_decoders(child, keep_paths, child_prefix)
        if child_decoder is not None:
            decoders[key] = child_decoder

    return decoders or None


def list_available_observation_keys(
    dataset_url: str | RoboticsRldsDatasetUrl,
) -> tuple[str, ...]:
    builder = tfds.builder_from_directory(normalize_dataset_url(dataset_url))
    paths = _observation_leaf_paths(builder.info.features)
    return tuple(sorted({path[-1] for path in paths}))


def build_observation_decoders(
    builder,
    observation_keys: tuple[str, ...] = DEFAULT_OBSERVATION_KEYS,
):
    observation_paths = _observation_leaf_paths(builder.info.features)
    observation_paths_by_key: dict[str, list[tuple[str, ...]]] = {}
    for path in observation_paths:
        observation_paths_by_key.setdefault(path[-1], []).append(path)

    keep_paths = set()
    missing_keys = []
    for key in observation_keys:
        matches = observation_paths_by_key.get(key, [])
        if not matches:
            missing_keys.append(key)
            continue

        preferred_matches = [path for path in matches if path[:2] == ("steps", "observation")]
        keep_paths.add(preferred_matches[0] if preferred_matches else matches[0])

    if missing_keys:
        available = ", ".join(sorted(observation_paths_by_key))
        raise KeyError(
            f"Missing requested observation keys: {missing_keys}. "
            f"Available observation keys: {available}"
        )

    decoders = _build_observation_skip_decoders(builder.info.features, keep_paths)
    return decoders if decoders is not None else {}


def step_size_bytes(
    field_shapes: dict[str, tuple[int, ...]],
    dtype: np.dtype = np.dtype(np.float32),
) -> int:
    return sum(
        int(np.prod(shape)) * dtype.itemsize
        for shape in field_shapes.values()
    )


def shard_size_mb_to_step_capacity(
    shard_size_mb: int | float,
    field_shapes: dict[str, tuple[int, ...]],
) -> int:
    bytes_per_step = step_size_bytes(field_shapes)
    shard_size_bytes = int(shard_size_mb * 1024 * 1024)
    step_capacity = shard_size_bytes // bytes_per_step
    if step_capacity <= 0:
        raise ValueError(
            f"shard_size_mb={shard_size_mb} is too small; "
            f"each step needs {bytes_per_step} bytes for field_shapes={field_shapes}."
        )
    return int(step_capacity)


def normalize_dataset_url(dataset_url: str | RoboticsRldsDatasetUrl) -> str:
    if isinstance(dataset_url, RoboticsRldsDatasetUrl):
        return dataset_url.value
    parsed = parse_dataset_url_arg(str(dataset_url))
    if isinstance(parsed, RoboticsRldsDatasetUrl):
        return parsed.value
    return str(parsed)


def parse_dataset_url_arg(dataset_url: str) -> str | RoboticsRldsDatasetUrl:
    if dataset_url in RoboticsRldsDatasetUrl.__members__:
        return RoboticsRldsDatasetUrl[dataset_url]
    upper = dataset_url.upper()
    if upper in RoboticsRldsDatasetUrl.__members__:
        return RoboticsRldsDatasetUrl[upper]
    return dataset_url


def dataset_name_from_url(dataset_url: str) -> str:
    parts = dataset_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse dataset name from URL: {dataset_url!r}")
    return parts[-2]


def resolve_data_dir(
    data_dir: str | Path,
    dataset_url: str | RoboticsRldsDatasetUrl,
) -> Path:
    base = Path(data_dir)
    dataset_name = dataset_name_from_url(normalize_dataset_url(dataset_url))
    if base.name == dataset_name:
        return base
    return base / dataset_name


def metadata_dir(data_dir: Path) -> Path:
    return data_dir / "metadata"


def dataset_metadata_dir(data_dir: Path) -> Path:
    """Metadata directory for an on-disk dataset cache."""
    return metadata_dir(data_dir)


def observation_shard_path(data_dir: Path, observation_key: str, shard_id: int) -> Path:
    return data_dir / observation_key / f"{shard_id:05d}.npy"


def trim_observation_shard(path: Path, used: int) -> None:
    """Drop preallocated tail rows so on-disk size matches ``used`` timesteps."""
    data = np.load(path, mmap_mode="r")
    if data.shape[0] == used:
        del data
        return

    trimmed = np.asarray(data[:used], dtype=np.float32)
    del data

    tmp_path = path.parent / f"{path.stem}.tmp.npy"
    np.save(tmp_path, trimmed)
    tmp_path.replace(path)


def select_observation_step(step, observation_keys: tuple[str, ...]):
    obs = step["observation"]
    return {key: obs[key] for key in observation_keys}


def validate_demo_lengths(demo: dict[str, np.ndarray]) -> int:
    lengths = {key: value.shape[0] for key, value in demo.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Observation fields have mismatched lengths: {lengths}")
    return next(iter(unique_lengths))


def validate_demo_field_shapes(
    demo: dict[str, np.ndarray],
    field_shapes: dict[str, tuple[int, ...]],
) -> None:
    mismatches = {
        key: (field_shapes[key], tuple(value.shape[1:]))
        for key, value in demo.items()
        if tuple(value.shape[1:]) != field_shapes[key]
    }
    if mismatches:
        raise ValueError(
            "Observation field shapes changed mid-dataset: "
            + ", ".join(
                f"{key} expected {expected} got {actual}"
                for key, (expected, actual) in mismatches.items()
            )
        )


STEP_BATCH_SIZE = 10_000


def iter_episode_step_batches(
    episode,
    observation_keys: tuple[str, ...],
    batch_size: int = STEP_BATCH_SIZE,
):
    """Yield observation step batches from an RLDS episode without concatenating."""

    @tf.autograph.experimental.do_not_convert
    def select_step(step):
        return select_observation_step(step, observation_keys)

    steps = episode["steps"].map(select_step, num_parallel_calls=tf.data.AUTOTUNE)
    yield from tfds.as_numpy(steps.batch(batch_size))


def stack_episode(episode, observation_keys: tuple[str, ...]):
    batches = list(iter_episode_step_batches(episode, observation_keys))
    if not batches:
        raise ValueError("Episode has zero steps")
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0).astype(
            np.float32, copy=False
        )
        for key in observation_keys
    }
