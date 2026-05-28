from pathlib import Path
from typing import Annotated, Optional

import typer

from app.download import run as run_download
from app.retarget import run as run_retarget
from cli_native import NoNativeOption, ReachSafetyOption
from reachability import REACH_SAFETY_MARGIN
from cli_reach_envelope import reach_envelope_app, reachability_legacy_app
from reachability.envelope import set_use_native_envelope
from retarget.config import default_config_path
from retarget.core import set_use_native_retarget
from rlds import DEFAULT_OBSERVATION_KEYS, RoboticsRldsDatasetUrl, list_adapter_dataset_names
from reachability import REACH_BINS_PHI, REACH_BINS_THETA, REACH_SAMPLE_COUNT_RETARGET

download_app = typer.Typer(help="Download RLDS observations to sharded on-disk .npy caches.")


@download_app.callback(invoke_without_command=True)
def download(
    dataset_url: Annotated[
        str,
        typer.Option(
            "--dataset-url",
            help="GCS RLDS directory URL, or enum name (e.g. DROID_100 or droid_100).",
        ),
    ] = str(RoboticsRldsDatasetUrl.DROID_100),
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            help="Base output directory; dataset name is appended unless already present.",
        ),
    ] = Path("data"),
    observation_key: Annotated[
        Optional[list[str]],
        typer.Option(
            "--observation-key",
            "--observation-keys",
            help=(
                "Native RLDS observation keys (requires --no-adapter). "
                f"Default naive keys: {' '.join(DEFAULT_OBSERVATION_KEYS)}. "
                "Repeat the flag for multiple keys."
            ),
        ),
    ] = None,
    no_adapter: Annotated[
        bool,
        typer.Option(
            "--no-adapter",
            help=(
                "Download native dataset keys without a format adapter. "
                f"Adapters are registered for: {', '.join(list_adapter_dataset_names())}."
            ),
        ),
    ] = False,
    list_observation_keys: Annotated[
        bool,
        typer.Option(
            "--list-observation-keys",
            help="Print observation leaf keys for --dataset-url and exit.",
        ),
    ] = False,
    no_trim_partial_shards: Annotated[
        bool,
        typer.Option(
            "--no-trim-partial-shards",
            help="Keep preallocated padding in the final partial shard (faster finalize).",
        ),
    ] = False,
    control_hz: Annotated[
        Optional[float],
        typer.Option(
            "--control-hz",
            help="Override documented control rate (Hz) stored in cache metadata.",
        ),
    ] = None,
    max_demos: Annotated[
        Optional[int],
        typer.Option(
            "--max-demos",
            help="Download at most this many episodes (TFDS split train[:N]).",
        ),
    ] = None,
) -> None:
    """Download observation tensors from an RLDS dataset on GCS."""
    run_download(
        dataset_url=dataset_url,
        data_dir=data_dir,
        observation_keys=tuple(observation_key) if observation_key else None,
        list_observation_keys=list_observation_keys,
        trim_partial_shards=not no_trim_partial_shards,
        use_adapter=not no_adapter,
        control_hz=control_hz,
        max_demos=max_demos,
    )


retarget_app = typer.Typer(help="Retarget cached DROID proprio demos onto UR3e.")


@retarget_app.callback(invoke_without_command=True)
def retarget(
    ctx: typer.Context,
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Base data directory or dataset cache path."),
    ] = Path("data"),
    dataset_url: Annotated[
        str,
        typer.Option(
            "--dataset-url",
            help="Dataset enum name or URL used to resolve cache dir under --data-dir.",
        ),
    ] = str(RoboticsRldsDatasetUrl.DROID_100),
    start_demo: Annotated[
        int,
        typer.Option("--start-demo", help="First demo index (inclusive)."),
    ] = 0,
    end_demo: Annotated[
        Optional[int],
        typer.Option("--end-demo", help="End demo index (exclusive). Default: all demos."),
    ] = None,
    robot_description: Annotated[
        str,
        typer.Option("--robot", help="Target robot description package name."),
    ] = "ur3e_description",
    panda_description: Annotated[
        str,
        typer.Option("--panda", help="Source-arm description for elbow-side hints."),
    ] = "panda_description",
    control_hz: Annotated[
        Optional[float],
        typer.Option(
            "--control-hz",
            help="Override control rate (Hz) for velocity costs; default reads cache metadata.",
        ),
    ] = None,
    disable_visualization: Annotated[
        bool,
        typer.Option(
            "--disable-visualization",
            help="Skip Meshcat robot playback during IK.",
        ),
    ] = False,
    no_plots: Annotated[
        bool,
        typer.Option(
            "--no-plots",
            help="Skip per-demo matplotlib error plots.",
        ),
    ] = False,
    reach_n_samples: Annotated[
        int,
        typer.Option(
            "--reach-n-samples",
            help="Monte Carlo FK samples for the target robot reach envelope cache.",
        ),
    ] = REACH_SAMPLE_COUNT_RETARGET,
    reach_n_theta: Annotated[
        int,
        typer.Option(
            "--reach-n-theta",
            help="Polar bins for the reach envelope used to scale Cartesian targets.",
        ),
    ] = REACH_BINS_THETA,
    reach_n_phi: Annotated[
        int,
        typer.Option(
            "--reach-n-phi",
            help="Azimuth bins for the reach envelope used to scale Cartesian targets.",
        ),
    ] = REACH_BINS_PHI,
    reach_force_rebuild: Annotated[
        bool,
        typer.Option(
            "--reach-force-rebuild",
            help="Rebuild reach envelope cache instead of loading existing NPZ.",
        ),
    ] = False,
    no_native: NoNativeOption = False,
    use_gpu: Annotated[
        bool,
        typer.Option(
            "--gpu",
            help=(
                "Retarget on GPU (CUDA build). Headless (--disable-visualization "
                "--no-plots) uses batched trajectories; add --live-stats for per-demo "
                "Rich progress instead."
            ),
        ),
    ] = False,
    live_stats: Annotated[
        bool,
        typer.Option(
            "--live-stats",
            help=(
                "Rich live stats panel (headless only). With --gpu, defaults to batched "
                "GPU without this flag."
            ),
        ),
    ] = False,
    reach_safety: ReachSafetyOption = REACH_SAFETY_MARGIN,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Retarget weights/solver YAML (default: config/default.yaml).",
        ),
    ] = Path("config/default.yaml"),
    save_joints: Annotated[
        bool,
        typer.Option(
            "--save-joints",
            help=(
                "Write joint trajectories under data/retargeted/<dataset>/ for later "
                "Meshcat replay (recommended with batched --gpu)."
            ),
        ),
    ] = False,
    save_joints_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--save-joints-dir",
            help="Override output directory for --save-joints.",
        ),
    ] = None,
) -> None:
    """Retarget proprio demos onto UR3e (reach envelope cache under data/reach_envelopes/)."""
    if ctx.invoked_subcommand is not None:
        return
    set_use_native_envelope(not no_native)
    set_use_native_retarget(not no_native)
    if use_gpu and no_native:
        raise typer.BadParameter("--gpu requires a CUDA native build; omit --no-native")
    config_path = config if config.is_file() else default_config_path()
    run_retarget(
        data_dir=data_dir,
        dataset_url=dataset_url,
        start_demo=start_demo,
        end_demo=end_demo,
        robot_description=robot_description,
        panda_description=panda_description,
        control_hz=control_hz,
        enable_visualization=not disable_visualization,
        show_plots=not no_plots,
        show_progress=True if live_stats else None,
        reach_n_samples=reach_n_samples,
        reach_n_theta=reach_n_theta,
        reach_n_phi=reach_n_phi,
        reach_force_rebuild=reach_force_rebuild,
        reach_safety=reach_safety,
        config_path=config_path,
        use_gpu=use_gpu,
        save_joints=save_joints,
        save_joints_dir=save_joints_dir,
    )


@retarget_app.command("replay")
def retarget_replay(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", help="Base data directory (used to resolve default cache path)."),
    ] = Path("data"),
    dataset_url: Annotated[
        str,
        typer.Option("--dataset-url", help="Dataset used for default cache path resolution."),
    ] = str(RoboticsRldsDatasetUrl.DROID_100),
    save_joints_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--save-joints-dir",
            help="Retarget output directory (default: data/retargeted/<dataset>).",
        ),
    ] = None,
    demo: Annotated[
        Optional[int],
        typer.Option("--demo", help="Demo index to replay (default: play --start-demo … range)."),
    ] = None,
    start_demo: Annotated[
        int,
        typer.Option("--start-demo", help="First demo index when replaying a range."),
    ] = 0,
    end_demo: Annotated[
        Optional[int],
        typer.Option("--end-demo", help="End demo index (exclusive) for range replay."),
    ] = None,
    fps: Annotated[
        Optional[float],
        typer.Option("--fps", help="Playback FPS (default: config runtime.display_fps)."),
    ] = None,
    loop: Annotated[
        bool,
        typer.Option("--loop", help="Loop the trajectory until Ctrl+C."),
    ] = False,
) -> None:
    """Replay cached joint trajectories in Meshcat (use SSH port 7000 forwarding)."""
    from retarget.cache import default_retarget_output_dir
    from retarget.replay import replay_demo, replay_range

    output_dir = save_joints_dir or default_retarget_output_dir(data_dir, dataset_url)
    if demo is not None:
        replay_demo(output_dir, demo, fps=fps, loop=loop)
    else:
        replay_range(output_dir, start_demo=start_demo, end_demo=end_demo, fps=fps)


app = typer.Typer(help="CS179 final project tools.", no_args_is_help=True)
app.add_typer(download_app, name="download")
app.add_typer(retarget_app, name="retarget")
app.add_typer(reach_envelope_app, name="reach-envelope")
app.add_typer(reachability_legacy_app, name="reachability")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
