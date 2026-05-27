#!/usr/bin/env python3
"""Directional reach envelope — Python API walkthrough.

Build or load a Monte Carlo FK envelope, query limits, and scale Cartesian targets.
See REACH_ENVELOPE.md and: uv run cs179 reach-envelope --help
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description

from reachability import (
    REACH_BINS_PHI,
    REACH_BINS_THETA,
    REACH_CACHE_DIR,
    REACH_SAFETY_MARGIN,
    DirectionalReachEnvelope,
    envelope_cache_path,
    scale_cartesian_to_robot,
    set_use_native_envelope,
)


def _tool_frame_id(model) -> int:
    return model.getFrameId("tool0") if model.existFrame("tool0") else model.getFrameId("ee_link")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        default="ur3e_description",
        help="robot_descriptions package name",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer FK samples and bins for a fast demo",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="Force Python build/query path (ignore cs179._native)",
    )
    parser.add_argument(
        "--safety",
        type=float,
        default=REACH_SAFETY_MARGIN,
        help="Radial scale factor in (0, 1] when clamping positions",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REACH_CACHE_DIR,
        help="Directory for NPZ envelope caches",
    )
    args = parser.parse_args()

    if not 0.0 < args.safety <= 1.0:
        raise SystemExit("--safety must be in (0, 1]")

    set_use_native_envelope(not args.no_native)

    n_samples = 50_000 if args.quick else 1_000_000
    n_theta = 24 if args.quick else REACH_BINS_THETA
    n_phi = 48 if args.quick else REACH_BINS_PHI

    pin_robot = load_robot_description(args.robot)
    model = pin_robot.model
    data = model.createData()
    frame_id = _tool_frame_id(model)
    frame_name = model.frames[frame_id].name

    cache_path = envelope_cache_path(
        args.robot,
        frame_name,
        n_theta,
        n_phi,
        n_samples,
        cache_dir=args.cache_dir,
    )
    print(f"Cache path: {cache_path}")

    # --- Build or load cached envelope ---
    envelope = DirectionalReachEnvelope.from_robot_cached(
        model,
        data,
        frame_id,
        robot_key=args.robot,
        n_samples=n_samples,
        n_theta=n_theta,
        n_phi=n_phi,
        cache_dir=args.cache_dir,
        force_rebuild=False,
        show_progress=True,
    )
    print(f"Envelope grid: {envelope.n_theta}×{envelope.n_phi}, max radius ≈ {envelope.max_radius():.3f} m")

    # --- Query reach along a few directions ---
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.4, 0.5],
        ],
        dtype=float,
    )
    limits = envelope.reach_limits(directions)
    for d, lim in zip(directions, limits):
        print(f"  limit({d}) ≈ {lim:.3f} m")

    # --- Scale workspace positions (e.g. demo Cartesian xyz) ---
    # Use a contiguous (N, 3) array; slices from (N, 6) pose rows need np.ascontiguousarray for native.
    positions = np.array(
        [
            [0.45, 0.0, 0.25],
            [0.55, 0.1, 0.15],
            [0.20, -0.35, 0.40],
        ],
        dtype=float,
    )
    scaled, scales = envelope.scale_positions(positions, safety=args.safety)
    print("scale_positions (xyz only):")
    for orig, new, s in zip(positions, scaled, scales):
        print(f"  {orig} -> {new}  (scale={s:.3f})")

    # --- Scale full 6-DoF demo rows (xyz + euler); helper keeps orientation columns ---
    cartesian = np.hstack([positions, np.zeros((len(positions), 3))])
    cartesian_scaled, scales2 = scale_cartesian_to_robot(cartesian, envelope, safety=args.safety)
    assert np.allclose(scales, scales2)

    # --- Explicit save/load (same format as CLI cache) ---
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        demo_path = Path(tmp.name)
    try:
        envelope.save(demo_path)
        loaded = DirectionalReachEnvelope.load(demo_path)
        print(f"Saved and reloaded {demo_path.name} (max radius {loaded.max_radius():.3f} m)")
    finally:
        demo_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
