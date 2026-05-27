from pathlib import Path

from rlds import (
    RldsObservationDownloader,
    list_available_observation_keys,
    parse_dataset_url_arg,
)


def run(
    *,
    dataset_url: str,
    data_dir: Path,
    observation_keys: tuple[str, ...] | None,
    list_observation_keys: bool,
    trim_partial_shards: bool,
    use_adapter: bool,
    control_hz: float | None,
    max_demos: int | None,
) -> None:
    url = parse_dataset_url_arg(dataset_url)

    if list_observation_keys:
        for key in list_available_observation_keys(url):
            print(key)
        return

    if use_adapter and observation_keys is not None:
        raise ValueError(
            "--observation-key can only be used with --no-adapter. "
            "With adapters enabled, output keys are defined by the dataset adapter."
        )

    RldsObservationDownloader(
        dataset_url=url,
        data_dir=data_dir,
        observation_keys=observation_keys,
        trim_partial_shards=trim_partial_shards,
        use_adapter=use_adapter,
        control_hz=control_hz,
        max_demos=max_demos,
    ).download()
