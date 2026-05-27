"""
Documented control rates for RLDS robotics datasets.

RLDS step tensors do not include timestamps or FPS. Values here come from dataset
papers / docs and are stored in ``metadata.json`` at download time.
"""

from __future__ import annotations

from .datasets import RoboticsRldsDatasetUrl
from .utils import normalize_dataset_url

# Hz for proprio / control steps (not camera FPS unless they coincide).
CONTROL_HZ: dict[RoboticsRldsDatasetUrl, float] = {
    # DROID: 15 Hz captures (https://droid-dataset.github.io, GH #13).
    RoboticsRldsDatasetUrl.DROID: 15.0,
    RoboticsRldsDatasetUrl.DROID_100: 15.0,
    # RT-1 KUKA RLDS: 3 Hz control (Brohan et al., RT-1).
    RoboticsRldsDatasetUrl.KUKA: 3.0,
}

_CONTROL_HZ_BY_URL: dict[str, float] = {
    member.value: hz for member, hz in CONTROL_HZ.items()
}


def control_hz_for_dataset(
    dataset: RoboticsRldsDatasetUrl,
) -> float | None:
    return CONTROL_HZ.get(dataset)


def control_hz_for_url(dataset_url: str) -> float | None:
    normalized = normalize_dataset_url(dataset_url)
    return _CONTROL_HZ_BY_URL.get(normalized)


def resolve_control_hz(
    dataset_url: str | RoboticsRldsDatasetUrl,
    *,
    override: float | None = None,
) -> float:
    """Return control rate in Hz, with optional CLI override."""
    if override is not None:
        if override <= 0:
            raise ValueError(f"control_hz must be positive, got {override}")
        return float(override)

    if isinstance(dataset_url, RoboticsRldsDatasetUrl):
        hz = control_hz_for_dataset(dataset_url)
    else:
        hz = control_hz_for_url(dataset_url)

    if hz is None:
        raise ValueError(
            f"No documented control_hz for {dataset_url!r}. "
            "Pass --control-hz on download/retarget, or add an entry to rlds.timing.CONTROL_HZ."
        )
    return hz
