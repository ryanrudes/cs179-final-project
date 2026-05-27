from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..datasets import RoboticsRldsDatasetUrl
from ..timing import control_hz_for_dataset
from ..utils import DEFAULT_OBSERVATION_KEYS

Transform = Callable[[np.ndarray], np.ndarray]
"""Maps one RLDS field batch ``(T, *in)`` → ``(T, *out)``; see :mod:`rlds.adapters.transforms`."""

DROID_PROPRIO_SHAPES: dict[str, tuple[int, ...]] = {
    "joint_position": (7,),
    "gripper_position": (1,),
    "cartesian_position": (6,),
}


@dataclass(frozen=True)
class Field:
    """One DROID-style output column and how to build it from RLDS observations."""

    shape: tuple[int, ...]
    source: str | None = None
    transform: Transform | None = None
    fill: float = np.nan

    @classmethod
    def from_rlds(
        cls,
        key: str,
        *shape: int,
        transform: Transform | None = None,
    ) -> "Field":
        return cls(source=key, shape=tuple(shape), transform=transform)

    @classmethod
    def missing(cls, *shape: int, fill: float = np.nan) -> "Field":
        return cls(source=None, shape=tuple(shape), fill=fill)


@dataclass(frozen=True)
class DatasetAdapter:
    """
    Declarative RLDS → DROID-style proprio adapter.

    Example::

        KUKA = DatasetAdapter(
            name="kuka",
            datasets=[RoboticsRldsDatasetUrl.KUKA],
            fields={
                "joint_position": Field.missing(7),
                "gripper_position": Field.from_rlds("gripper_closed", 1, transform=to_column),
                "cartesian_position": Field.from_rlds(
                    "clip_function_input/base_pose_tool_reached",
                    6,
                    transform=pose7_xyzw_to_cartesian6,
                ),
            },
            notes={"joint_position_available": False},
        )
    """

    name: str
    datasets: tuple[RoboticsRldsDatasetUrl, ...]
    fields: dict[str, Field]
    notes: dict[str, object] = field(default_factory=dict)
    control_hz: float | None = None

    def __post_init__(self) -> None:
        unknown = set(self.fields) - set(DEFAULT_OBSERVATION_KEYS)
        if unknown:
            raise ValueError(
                f"Adapter {self.name!r} defines unknown output fields: {sorted(unknown)}. "
                f"Expected keys: {list(DEFAULT_OBSERVATION_KEYS)}"
            )

    @property
    def source_keys(self) -> tuple[str, ...]:
        seen: set[str] = set()
        keys: list[str] = []
        for spec in self.fields.values():
            if spec.source is not None and spec.source not in seen:
                seen.add(spec.source)
                keys.append(spec.source)
        return tuple(keys)

    @property
    def output_keys(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())

    @property
    def output_field_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: spec.shape for name, spec in self.fields.items()}

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self.notes)

    @property
    def resolved_control_hz(self) -> float:
        """Control rate (Hz) for velocity scaling; from ``control_hz`` or dataset registry."""
        if self.control_hz is not None:
            return self.control_hz

        rates: list[float] = []
        for dataset in self.datasets:
            hz = control_hz_for_dataset(dataset)
            if hz is None:
                raise ValueError(
                    f"Adapter {self.name!r} includes {dataset.name} with no entry in "
                    "rlds.timing.CONTROL_HZ. Set control_hz= on the adapter or extend the registry."
                )
            rates.append(hz)

        unique = set(rates)
        if len(unique) != 1:
            raise ValueError(
                f"Adapter {self.name!r} spans datasets with different control_hz values: {unique}"
            )
        return rates[0]

    def adapt_batch(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        num_steps = _infer_batch_length(batch, self.source_keys)
        adapted: dict[str, np.ndarray] = {}

        for name, spec in self.fields.items():
            if spec.source is None:
                adapted[name] = np.full(
                    (num_steps, *spec.shape),
                    spec.fill,
                    dtype=np.float32,
                )
                continue

            values = np.asarray(batch[spec.source], dtype=np.float32)
            if spec.transform is not None:
                values = spec.transform(values)
            else:
                values = values.reshape(num_steps, *spec.shape)
            adapted[name] = values

        return adapted


def droid_layout(
    name: str,
    *datasets: RoboticsRldsDatasetUrl,
    notes: dict[str, object] | None = None,
) -> DatasetAdapter:
    """Adapter for datasets that already expose the standard DROID proprio keys."""
    return DatasetAdapter(
        name=name,
        datasets=datasets,
        fields={
            key: Field.from_rlds(key, *DROID_PROPRIO_SHAPES[key])
            for key in DEFAULT_OBSERVATION_KEYS
        },
        notes=notes or {},
    )


def _infer_batch_length(
    batch: dict[str, np.ndarray],
    source_keys: Sequence[str],
) -> int:
    if not source_keys:
        raise ValueError("Cannot infer batch length without source keys")
    return int(np.asarray(batch[source_keys[0]]).shape[0])


# Back-compat alias used by the downloader type hints.
RldsDatasetAdapter = DatasetAdapter
