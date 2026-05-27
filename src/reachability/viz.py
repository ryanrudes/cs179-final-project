"""Viser visualization for directional reach envelopes and workspace samples."""

from __future__ import annotations

import time

import numpy as np
import viser
import viser.extras
from robot_descriptions.loaders.pinocchio import load_robot_description
from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf

from .envelope import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    REACH_SAFETY_MARGIN,
    REACH_SAMPLE_COUNT_VIZ,
    DirectionalReachEnvelope,
    sample_tool_positions,
)


def _sphere_mesh(
    radius: float,
    pivot: np.ndarray,
    n_theta: int = 32,
    n_phi: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    dirs = np.column_stack(
        [
            np.sin(tt.ravel()) * np.cos(pp.ravel()),
            np.sin(tt.ravel()) * np.sin(pp.ravel()),
            np.cos(tt.ravel()),
        ]
    )
    vertices = pivot + radius * dirs
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            jn = (j + 1) % n_phi
            a = i * n_phi + j
            b = i * n_phi + jn
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + jn
            faces.append((a, b, c))
            faces.append((b, d, c))
    return vertices, np.asarray(faces, dtype=np.uint32)


def _tool_frame_id(model) -> int:
    if model.existFrame("tool0"):
        return model.getFrameId("tool0")
    return model.getFrameId("panda_hand_tcp")


def _frame_camera_on_envelope(
    server: viser.ViserServer,
    envelope: DirectionalReachEnvelope,
    pivot: np.ndarray,
    *,
    safety: float = REACH_SAFETY_MARGIN,
) -> None:
    """Point new clients at the workspace envelope (robot base at origin)."""
    max_r = max(envelope.max_radius() * safety, 0.15)
    center = np.asarray(pivot, dtype=np.float64)
    dist = max_r * 2.8
    server.initial_camera.position = center + np.array([dist, dist, dist * 0.55], dtype=np.float64)
    server.initial_camera.look_at = center + np.array([0.0, 0.0, max_r * 0.25], dtype=np.float64)


def add_reachability_to_viser(
    server: viser.ViserServer,
    envelope: DirectionalReachEnvelope,
    *,
    pivot: np.ndarray | None = None,
    prefix: str = "/reach",
    samples: np.ndarray | None = None,
    n_mesh_theta: int = 48,
    n_mesh_phi: int = 96,
    safety: float = REACH_SAFETY_MARGIN,
) -> dict[str, object]:
    """Add reachability meshes to an existing Viser server. Caller must keep `server` alive."""
    if pivot is None:
        pivot = np.zeros(3, dtype=float)

    boundary_verts, boundary_faces = envelope.boundary_mesh(
        pivot=pivot,
        n_theta=n_mesh_theta,
        n_phi=n_mesh_phi,
        safety=safety,
    )
    max_r = envelope.max_radius() * safety
    sphere_verts, sphere_faces = _sphere_mesh(max_r, pivot)

    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    handles: dict[str, object] = {}
    if samples is not None and len(samples) > 0:
        sample_colors = np.tile(np.array([[80, 180, 255]], dtype=np.uint8), (len(samples), 1))
        handles["samples"] = server.scene.add_point_cloud(
            f"{prefix}/samples",
            points=samples.astype(np.float32),
            colors=sample_colors,
            point_size=0.006,
            point_shape="circle",
            visible=False,
        )

    handles["envelope"] = server.scene.add_mesh_simple(
        f"{prefix}/envelope",
        vertices=boundary_verts.astype(np.float32),
        faces=boundary_faces,
        color=(90, 220, 140),
        opacity=0.45,
        wireframe=False,
        side="double",
    )
    handles["wire"] = server.scene.add_mesh_simple(
        f"{prefix}/envelope_wireframe",
        vertices=boundary_verts.astype(np.float32),
        faces=boundary_faces,
        color=(40, 160, 90),
        opacity=0.85,
        wireframe=True,
        side="double",
        visible=False,
    )
    handles["sphere"] = server.scene.add_mesh_simple(
        f"{prefix}/max_sphere",
        vertices=sphere_verts.astype(np.float32),
        faces=sphere_faces,
        color=(255, 120, 90),
        opacity=0.15,
        wireframe=True,
        side="double",
        visible=False,
    )

    _frame_camera_on_envelope(server, envelope, pivot, safety=safety)
    return handles


def start_reachability_viser(
    envelope: DirectionalReachEnvelope,
    *,
    robot_description: str = "ur3e_description",
    show_robot: bool = True,
    show_samples: bool = False,
    n_samples: int = REACH_SAMPLE_COUNT_VIZ,
    safety: float = REACH_SAFETY_MARGIN,
) -> viser.ViserServer:
    """Lightweight reachability-only Viser view (no Panda rebuild). Keeps server alive via return value."""
    server = viser.ViserServer()

    samples = None
    if show_samples:
        pin_robot = load_robot_description(robot_description)
        model = pin_robot.model
        data = model.createData()
        frame_id = _tool_frame_id(model)
        print(f"Sampling {n_samples} tool positions for point cloud...")
        samples = sample_tool_positions(model, data, frame_id, n_samples)

    add_reachability_to_viser(server, envelope, samples=samples, safety=safety)

    if show_robot:
        urdf = load_urdf(robot_description)
        viser.extras.ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/{robot_description}")

    max_r = envelope.max_radius() * safety
    with server.gui.add_folder("Reachability"):
        server.gui.add_text(
            "Info",
            initial_value=(
                f"{robot_description}: max reach ≈ {envelope.max_radius():.3f} m "
                f"(safety {safety:.2f})"
            ),
        )
        server.gui.add_markdown(
            "**Green mesh** = directional reach used for retarget scaling. "
            "Open this URL (not Meshcat) to see it."
        )

    print(f"Viser reachability view ready (max reach ≈ {max_r:.3f} m).")
    return server


def visualize_reachability_in_viser(
    robot_description: str = "ur3e_description",
    *,
    envelope: DirectionalReachEnvelope | None = None,
    n_samples: int = REACH_SAMPLE_COUNT_VIZ,
    n_theta: int | None = None,
    n_phi: int | None = None,
    n_mesh_theta: int = 48,
    n_mesh_phi: int = 96,
    pivot: np.ndarray | None = None,
    compare_robot: str | None = None,
    compare_samples: int | None = None,
    force_rebuild: bool = False,
    show_robot: bool = True,
    show_samples: bool = False,
    block: bool = True,
    safety: float = REACH_SAFETY_MARGIN,
) -> viser.ViserServer:
    """Full standalone viewer: directional envelope by default; optional FK cloud and comparison."""
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must be in (0, 1], got {safety}")

    if pivot is None:
        pivot = np.zeros(3, dtype=float)

    if n_theta is None:
        n_theta = REACH_BINS_THETA
    if n_phi is None:
        n_phi = REACH_BINS_PHI

    pin_robot = load_robot_description(robot_description)
    model = pin_robot.model
    data = model.createData()
    frame_id = _tool_frame_id(model)

    if envelope is None:
        envelope = DirectionalReachEnvelope.from_robot_cached(
            model,
            data,
            frame_id,
            robot_key=robot_description,
            n_samples=n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            force_rebuild=force_rebuild,
        )

    samples = None
    if show_samples:
        print(f"Sampling {n_samples} tool positions for point cloud...")
        samples = sample_tool_positions(model, data, frame_id, n_samples)

    server = viser.ViserServer()
    handles = add_reachability_to_viser(
        server,
        envelope,
        pivot=pivot,
        samples=samples,
        n_mesh_theta=n_mesh_theta,
        n_mesh_phi=n_mesh_phi,
        safety=safety,
    )

    if show_robot:
        urdf = load_urdf(robot_description)
        viser.extras.ViserUrdf(server, urdf_or_path=urdf, root_node_name=f"/{robot_description}")

    compare_mesh = None
    compare_wire = None
    if compare_robot is not None:
        cmp_n_samples = n_samples if compare_samples is None else compare_samples
        print(
            f"Building comparison envelope for {compare_robot} "
            f"({cmp_n_samples:,} samples, {n_theta}×{n_phi} bins)..."
        )
        cmp_pin = load_robot_description(compare_robot)
        cmp_model = cmp_pin.model
        cmp_data = cmp_model.createData()
        cmp_frame = _tool_frame_id(cmp_model)
        cmp_envelope = DirectionalReachEnvelope.from_robot_cached(
            cmp_model,
            cmp_data,
            cmp_frame,
            robot_key=compare_robot,
            n_samples=cmp_n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            force_rebuild=force_rebuild,
        )
        cmp_verts, cmp_faces = cmp_envelope.boundary_mesh(
            pivot=pivot,
            n_theta=n_mesh_theta,
            n_phi=n_mesh_phi,
            safety=safety,
        )
        compare_mesh = server.scene.add_mesh_simple(
            f"/reach/compare/{compare_robot}",
            vertices=cmp_verts.astype(np.float32),
            faces=cmp_faces,
            color=(255, 200, 80),
            opacity=0.25,
            wireframe=False,
            side="double",
            visible=False,
        )
        compare_wire = server.scene.add_mesh_simple(
            f"/reach/compare/{compare_robot}_wire",
            vertices=cmp_verts.astype(np.float32),
            faces=cmp_faces,
            color=(180, 120, 40),
            opacity=0.55,
            wireframe=True,
            side="double",
            visible=False,
        )

    with server.gui.add_folder("Reachability"):
        show_samples_cb = server.gui.add_checkbox(
            "FK samples",
            initial_value=False,
            disabled="samples" not in handles,
        )
        show_envelope = server.gui.add_checkbox("Directional envelope", initial_value=True)
        show_wire = server.gui.add_checkbox("Envelope wireframe", initial_value=False)
        show_sphere = server.gui.add_checkbox("Max-radius sphere", initial_value=False)
        show_compare = server.gui.add_checkbox(
            "Comparison envelope",
            initial_value=False,
            disabled=compare_robot is None,
        )
        server.gui.add_text(
            "Info",
            initial_value=(
                f"{robot_description}: max reach ≈ {envelope.max_radius():.3f} m "
                f"(safety {safety:.2f})"
            ),
        )

    @show_samples_cb.on_update
    def _on_samples(_):
        if "samples" in handles:
            handles["samples"].visible = show_samples_cb.value

    @show_envelope.on_update
    def _on_envelope(_):
        handles["envelope"].visible = show_envelope.value

    @show_wire.on_update
    def _on_wire(_):
        handles["wire"].visible = show_wire.value

    @show_sphere.on_update
    def _on_sphere(_):
        handles["sphere"].visible = show_sphere.value

    if compare_robot is not None:

        @show_compare.on_update
        def _on_compare(_):
            visible = show_compare.value
            compare_mesh.visible = visible
            compare_wire.visible = visible

    print("Open the Viser URL above.")
    if block:
        while True:
            time.sleep(1.0)
    return server


if __name__ == "__main__":
    visualize_reachability_in_viser()
