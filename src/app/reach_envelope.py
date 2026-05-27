"""CLI backends for directional reach envelope build and visualization."""

from __future__ import annotations

from typing import Optional

from robot_descriptions.loaders.pinocchio import load_robot_description

from reachability import DirectionalReachEnvelope
from reachability.viz import _tool_frame_id, visualize_reachability_in_viser


def _build_envelope_for_robot(
    robot_description: str,
    *,
    n_samples: int,
    n_theta: int,
    n_phi: int,
    force_rebuild: bool,
) -> DirectionalReachEnvelope:
    pin_robot = load_robot_description(robot_description)
    model = pin_robot.model
    data = model.createData()
    frame_id = _tool_frame_id(model)
    return DirectionalReachEnvelope.from_robot_cached(
        model,
        data,
        frame_id,
        robot_key=robot_description,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        force_rebuild=force_rebuild,
    )


def run_build(
    *,
    robot_description: str,
    compare_robot: Optional[str],
    n_samples: int,
    n_theta: int,
    n_phi: int,
    force_rebuild: bool,
    compare_samples: Optional[int],
) -> None:
    """Monte Carlo FK envelope build; writes NPZ under data/reach_envelopes/."""
    _build_envelope_for_robot(
        robot_description,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        force_rebuild=force_rebuild,
    )
    if compare_robot is not None:
        cmp_n_samples = n_samples if compare_samples is None else compare_samples
        _build_envelope_for_robot(
            compare_robot,
            n_samples=cmp_n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            force_rebuild=force_rebuild,
        )


def run_visualize(
    *,
    robot_description: str,
    compare_robot: Optional[str],
    n_samples: int,
    n_theta: int,
    n_phi: int,
    n_mesh_theta: int,
    n_mesh_phi: int,
    force_rebuild: bool,
    compare_samples: Optional[int],
    show_robot: bool,
    show_samples: bool,
    block: bool,
    reach_safety: float,
) -> None:
    visualize_reachability_in_viser(
        robot_description=robot_description,
        compare_robot=compare_robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        n_mesh_theta=n_mesh_theta,
        n_mesh_phi=n_mesh_phi,
        force_rebuild=force_rebuild,
        compare_samples=compare_samples,
        show_robot=show_robot,
        show_samples=show_samples,
        block=block,
        safety=reach_safety,
    )
