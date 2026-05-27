"""Load proprio demos for retargeting from an on-disk RLDS cache."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rlds import RldsObservationLoader, RoboticsRldsDatasetUrl
from rlds.demos import ProprioDemo, iter_proprio_demos

__all__ = ["ProprioDemo", "iter_retarget_demos"]


def iter_retarget_demos(
    *,
    loader: RldsObservationLoader | None = None,
    data_dir: str | Path = "data",
    dataset_url: str | RoboticsRldsDatasetUrl = RoboticsRldsDatasetUrl.DROID_100,
    start_demo: int = 0,
    end_demo: int | None = None,
) -> Iterator[ProprioDemo]:
    """Yield proprio demos from ``RldsObservationLoader``."""
    if loader is None:
        loader = RldsObservationLoader(data_dir=data_dir, dataset_url=dataset_url)
    yield from iter_proprio_demos(loader, start_demo=start_demo, end_demo=end_demo)
