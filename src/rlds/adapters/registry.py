from ..datasets import RoboticsRldsDatasetUrl
from ..utils import normalize_dataset_url
from .base import DatasetAdapter
from .droid import DROID
from .kuka import KUKA

# Register adapters here (one line each).
ALL_ADAPTERS: tuple[DatasetAdapter, ...] = (
    DROID,
    KUKA,
)


def _index_adapters() -> tuple[
    dict[RoboticsRldsDatasetUrl, DatasetAdapter],
    dict[str, DatasetAdapter],
]:
    by_enum: dict[RoboticsRldsDatasetUrl, DatasetAdapter] = {}
    by_url: dict[str, DatasetAdapter] = {}
    for adapter in ALL_ADAPTERS:
        for dataset in adapter.datasets:
            if dataset in by_enum:
                raise ValueError(
                    f"Dataset {dataset.name} is registered on both "
                    f"{by_enum[dataset].name!r} and {adapter.name!r}"
                )
            by_enum[dataset] = adapter
            by_url[dataset.value] = adapter
    return by_enum, by_url


_ADAPTERS_BY_ENUM, _ADAPTERS_BY_URL = _index_adapters()


def list_adapter_dataset_names() -> tuple[str, ...]:
    return tuple(sorted(member.name for member in _ADAPTERS_BY_ENUM))


def resolve_adapter(
    dataset_url: str | RoboticsRldsDatasetUrl,
) -> DatasetAdapter | None:
    if isinstance(dataset_url, RoboticsRldsDatasetUrl):
        return _ADAPTERS_BY_ENUM.get(dataset_url)

    normalized = normalize_dataset_url(dataset_url)
    return _ADAPTERS_BY_URL.get(normalized)


def require_adapter(
    dataset_url: str | RoboticsRldsDatasetUrl,
) -> DatasetAdapter:
    adapter = resolve_adapter(dataset_url)
    if adapter is None:
        supported = ", ".join(list_adapter_dataset_names())
        raise ValueError(
            f"No dataset adapter registered for {dataset_url!r}. "
            f"Supported enum datasets: {supported}. "
            "Pass --no-adapter to download native observation keys instead."
        )
    return adapter
