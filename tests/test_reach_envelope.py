"""Native reach envelope: query parity and build smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
from robot_descriptions.loaders.pinocchio import load_robot_description
from scipy.interpolate import RegularGridInterpolator

from reachability.envelope import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    DirectionalReachEnvelope,
    _NATIVE_ENVELOPE,
    _build_bin_radii_batched,
    _fill_empty_bins,
    set_use_native_envelope,
)

pytestmark = pytest.mark.skipif(_NATIVE_ENVELOPE is None, reason="native reach envelope not built")


def _python_interp_limits(envelope: DirectionalReachEnvelope, directions: np.ndarray) -> np.ndarray:
    """Reference reach_limits using the pure-Python interpolator only."""
    n_theta, n_phi = envelope.bin_radii.shape
    theta_pts = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi_pts = (np.arange(n_phi) + 0.5) * (2.0 * np.pi) / n_phi - np.pi
    fallback = float(np.max(envelope.bin_radii))
    interp = RegularGridInterpolator(
        (theta_pts, phi_pts),
        envelope.bin_radii,
        bounds_error=False,
        fill_value=fallback,
    )
    dirs = np.asarray(directions, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs.reshape(1, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-9)
    theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    return interp(np.column_stack([theta, phi]))


def test_native_reach_limits_match_python_on_same_bins() -> None:
    robot = load_robot_description("ur3_description")
    frame_id = robot.model.getFrameId("tool0")
    bins = _fill_empty_bins(
        _build_bin_radii_batched(
            robot.model,
            robot.data,
            frame_id,
            n_samples=3_000,
            n_theta=24,
            n_phi=48,
            batch_size=1_500,
            show_progress=False,
        )
    )

    envelope = DirectionalReachEnvelope(bins)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, -0.3, 0.4],
            [-0.5, 0.2, 0.7],
        ],
        dtype=float,
    )
    native_limits = envelope.reach_limits(directions)
    python_limits = _python_interp_limits(envelope, directions)
    assert np.allclose(native_limits, python_limits, rtol=1e-10, atol=1e-10)


def test_native_scale_positions_non_contiguous_xyz_slice() -> None:
    """scale_cartesian_to_robot passes (N,3) slices from (N,6) arrays — must not confuse C++."""
    robot = load_robot_description("ur3_description")
    frame_id = robot.model.getFrameId("tool0")
    bins = _build_bin_radii_batched(
        robot.model,
        robot.data,
        frame_id,
        n_samples=2_000,
        n_theta=16,
        n_phi=32,
        batch_size=1_000,
        show_progress=False,
    )
    set_use_native_envelope(True)
    try:
        envelope = DirectionalReachEnvelope(bins)
        assert envelope._native is not None
        cart6 = np.zeros((4, 6), dtype=np.float64)
        cart6[:, :3] = [
            [0.35, 0.0, 0.25],
            [0.40, 0.05, 0.20],
            [0.30, -0.1, 0.30],
            [0.45, 0.0, 0.15],
        ]
        _, scales_slice = envelope.scale_positions(cart6[:, :3])
        _, scales_contig = envelope.scale_positions(np.ascontiguousarray(cart6[:, :3]))
        assert np.allclose(scales_slice, scales_contig, rtol=1e-10, atol=1e-10)
    finally:
        set_use_native_envelope(True)


def test_no_native_skips_native_wrapper() -> None:
    robot = load_robot_description("ur3_description")
    frame_id = robot.model.getFrameId("tool0")
    bins = np.ones((8, 16), dtype=np.float64) * 0.5
    set_use_native_envelope(False)
    try:
        envelope = DirectionalReachEnvelope(bins)
        assert envelope._native is None
        limits = envelope.reach_limits(np.array([[1.0, 0.0, 0.0]], dtype=float))
        assert limits.shape == (1,)
    finally:
        set_use_native_envelope(True)


def test_native_build_smoke() -> None:
    robot = load_robot_description("ur3_description")
    frame_id = robot.model.getFrameId("tool0")
    bins = _build_bin_radii_batched(
        robot.model,
        robot.data,
        frame_id,
        n_samples=5_000,
        n_theta=REACH_BINS_THETA,
        n_phi=REACH_BINS_PHI,
        batch_size=2_500,
        show_progress=False,
    )
    envelope = DirectionalReachEnvelope(bins)
    max_r = envelope.max_radius()
    assert 0.35 < max_r < 0.95

    positions = np.array([[0.7, 0.0, 0.0], [0.0, 0.0, 0.15]], dtype=float)
    scaled, scales = envelope.scale_positions(positions)
    assert scaled.shape == positions.shape
    assert scales.shape == (2,)
    assert np.all(scales <= 1.0 + 1e-9)
