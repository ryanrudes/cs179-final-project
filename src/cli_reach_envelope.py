"""Typer options and commands for `cs179 reach-envelope`."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from app.reach_envelope import run_build, run_visualize
from cli_native import NoNativeOption, ReachSafetyOption
from reachability import REACH_BINS_PHI, REACH_BINS_THETA, REACH_SAMPLE_COUNT_VIZ, REACH_SAFETY_MARGIN
from reachability.envelope import set_use_native_envelope

reach_envelope_app = typer.Typer(
    help="Build and visualize directional reach envelopes (Monte Carlo FK).",
    no_args_is_help=True,
)

reachability_legacy_app = typer.Typer(
    help="Deprecated: use `cs179 reach-envelope` (build | visualize).",
    invoke_without_command=True,
)


def _run_visualize_command(
    *,
    robot_description: str,
    compare_robot: Optional[str],
    n_samples: int,
    n_theta: int,
    n_phi: int,
    mesh_theta: Optional[int],
    mesh_phi: Optional[int],
    force_rebuild: bool,
    compare_samples: Optional[int],
    fk_samples: bool,
    no_robot: bool,
    no_block: bool,
    reach_safety: float,
) -> None:
    run_visualize(
        robot_description=robot_description,
        compare_robot=compare_robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        n_mesh_theta=mesh_theta if mesh_theta is not None else n_theta,
        n_mesh_phi=mesh_phi if mesh_phi is not None else n_phi,
        force_rebuild=force_rebuild,
        compare_samples=compare_samples,
        show_robot=not no_robot,
        show_samples=fk_samples,
        block=not no_block,
        reach_safety=reach_safety,
    )


@reach_envelope_app.command("build")
def reach_envelope_build(
    robot_description: Annotated[
        str,
        typer.Option("--robot", help="Robot description package to build."),
    ] = "ur3e_description",
    compare_robot: Annotated[
        Optional[str],
        typer.Option(
            "--compare-robot",
            help="Also build an envelope for a second robot (separate cache file).",
        ),
    ] = None,
    n_samples: Annotated[
        int,
        typer.Option("--n-samples", help="Monte Carlo FK samples."),
    ] = REACH_SAMPLE_COUNT_VIZ,
    n_theta: Annotated[
        int,
        typer.Option("--n-theta", help="Polar bins for the directional reach grid."),
    ] = REACH_BINS_THETA,
    n_phi: Annotated[
        int,
        typer.Option("--n-phi", help="Azimuth bins for the directional reach grid."),
    ] = REACH_BINS_PHI,
    force_rebuild: Annotated[
        bool,
        typer.Option(
            "--force-rebuild",
            help="Ignore cached envelope and rebuild.",
        ),
    ] = False,
    compare_samples: Annotated[
        Optional[int],
        typer.Option(
            "--compare-samples",
            help="FK samples for --compare-robot (default: same as --n-samples).",
        ),
    ] = None,
    no_native: NoNativeOption = False,
) -> None:
    """Build envelope grid(s) and save to data/reach_envelopes/ (no Viser)."""
    set_use_native_envelope(not no_native)
    run_build(
        robot_description=robot_description,
        compare_robot=compare_robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        force_rebuild=force_rebuild,
        compare_samples=compare_samples,
    )


@reach_envelope_app.command("visualize")
def reach_envelope_visualize(
    robot_description: Annotated[
        str,
        typer.Option("--robot", help="Robot to visualize."),
    ] = "ur3e_description",
    compare_robot: Annotated[
        Optional[str],
        typer.Option(
            "--compare-robot",
            help="Optional second robot envelope (e.g. panda_description).",
        ),
    ] = None,
    n_samples: Annotated[
        int,
        typer.Option(
            "--n-samples",
            help="Monte Carlo FK samples for envelope build (and point cloud if --fk-samples).",
        ),
    ] = REACH_SAMPLE_COUNT_VIZ,
    n_theta: Annotated[
        int,
        typer.Option("--n-theta", help="Polar bins for the directional reach grid."),
    ] = REACH_BINS_THETA,
    n_phi: Annotated[
        int,
        typer.Option("--n-phi", help="Azimuth bins for the directional reach grid."),
    ] = REACH_BINS_PHI,
    mesh_theta: Annotated[
        Optional[int],
        typer.Option(
            "--mesh-theta",
            help="Boundary-mesh polar tessellation (default: same as --n-theta).",
        ),
    ] = None,
    mesh_phi: Annotated[
        Optional[int],
        typer.Option(
            "--mesh-phi",
            help="Boundary-mesh azimuth tessellation (default: same as --n-phi).",
        ),
    ] = None,
    force_rebuild: Annotated[
        bool,
        typer.Option(
            "--force-rebuild",
            help="Ignore cached envelope and rebuild.",
        ),
    ] = False,
    compare_samples: Annotated[
        Optional[int],
        typer.Option(
            "--compare-samples",
            help="FK samples for --compare-robot (default: same as --n-samples).",
        ),
    ] = None,
    fk_samples: Annotated[
        bool,
        typer.Option(
            "--fk-samples",
            help="Sample FK point cloud for Viser (slow; off by default).",
        ),
    ] = False,
    no_robot: Annotated[
        bool,
        typer.Option("--no-robot", help="Hide URDF in the Viser scene."),
    ] = False,
    no_block: Annotated[
        bool,
        typer.Option("--no-block", help="Return immediately instead of blocking on Viser."),
    ] = False,
    no_native: NoNativeOption = False,
    reach_safety: ReachSafetyOption = REACH_SAFETY_MARGIN,
) -> None:
    """Open Viser with the directional reach envelope (builds or loads cache first)."""
    set_use_native_envelope(not no_native)
    _run_visualize_command(
        robot_description=robot_description,
        compare_robot=compare_robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        mesh_theta=mesh_theta,
        mesh_phi=mesh_phi,
        force_rebuild=force_rebuild,
        compare_samples=compare_samples,
        fk_samples=fk_samples,
        no_robot=no_robot,
        no_block=no_block,
        reach_safety=reach_safety,
    )


@reachability_legacy_app.callback()
def reachability_legacy(
    robot_description: Annotated[
        str,
        typer.Option("--robot", help="Robot to visualize."),
    ] = "ur3e_description",
    compare_robot: Annotated[
        Optional[str],
        typer.Option(
            "--compare-robot",
            help="Optional second robot envelope (e.g. panda_description).",
        ),
    ] = None,
    n_samples: Annotated[
        int,
        typer.Option(
            "--n-samples",
            help="Monte Carlo FK samples for envelope build (and point cloud if --fk-samples).",
        ),
    ] = REACH_SAMPLE_COUNT_VIZ,
    n_theta: Annotated[
        int,
        typer.Option("--n-theta", help="Polar bins for the directional reach grid."),
    ] = REACH_BINS_THETA,
    n_phi: Annotated[
        int,
        typer.Option("--n-phi", help="Azimuth bins for the directional reach grid."),
    ] = REACH_BINS_PHI,
    mesh_theta: Annotated[
        Optional[int],
        typer.Option(
            "--mesh-theta",
            help="Boundary-mesh polar tessellation (default: same as --n-theta).",
        ),
    ] = None,
    mesh_phi: Annotated[
        Optional[int],
        typer.Option(
            "--mesh-phi",
            help="Boundary-mesh azimuth tessellation (default: same as --n-phi).",
        ),
    ] = None,
    force_rebuild: Annotated[
        bool,
        typer.Option(
            "--force-rebuild",
            help="Ignore cached envelope and rebuild.",
        ),
    ] = False,
    compare_samples: Annotated[
        Optional[int],
        typer.Option(
            "--compare-samples",
            help="FK samples for --compare-robot (default: same as --n-samples).",
        ),
    ] = None,
    fk_samples: Annotated[
        bool,
        typer.Option(
            "--fk-samples",
            help="Sample FK point cloud for Viser (slow; off by default).",
        ),
    ] = False,
    no_robot: Annotated[
        bool,
        typer.Option("--no-robot", help="Hide URDF in the Viser scene."),
    ] = False,
    no_block: Annotated[
        bool,
        typer.Option("--no-block", help="Return immediately instead of blocking on Viser."),
    ] = False,
    no_native: NoNativeOption = False,
    reach_safety: ReachSafetyOption = REACH_SAFETY_MARGIN,
) -> None:
    """Deprecated alias for `cs179 reach-envelope visualize`."""
    set_use_native_envelope(not no_native)
    typer.secho(
        "Note: `cs179 reachability` is deprecated; use `cs179 reach-envelope visualize`.",
        fg=typer.colors.YELLOW,
        err=True,
    )
    _run_visualize_command(
        robot_description=robot_description,
        compare_robot=compare_robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        mesh_theta=mesh_theta,
        mesh_phi=mesh_phi,
        force_rebuild=force_rebuild,
        compare_samples=compare_samples,
        fk_samples=fk_samples,
        no_robot=no_robot,
        no_block=no_block,
        reach_safety=reach_safety,
    )
