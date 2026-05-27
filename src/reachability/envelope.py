"""Directional reach envelope and workspace sampling."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter

REACH_SAFETY_MARGIN = 0.9
REACH_BINS_THETA = 64
REACH_BINS_PHI = 128

# Monte Carlo FK samples (statistical coverage of the bin grid).
REACH_SAMPLE_COUNT_RETARGET = 10_000_000
REACH_SAMPLE_COUNT_VIZ = 10_000_000

# FK samples processed per batch; bin updates use vectorized np.maximum.at.
REACH_BUILD_BATCH_SIZE = 50_000

REACH_CACHE_DIR = Path("data/reach_envelopes")

_NATIVE_ENVELOPE_CLASS = None
try:
    from cs179._native import DirectionalReachEnvelope as _NativeDirectionalReachEnvelope

    _NATIVE_ENVELOPE_CLASS = _NativeDirectionalReachEnvelope
except ImportError:
    pass

# Tests and docs refer to whether the extension was built (ignores --no-native).
_NATIVE_ENVELOPE = _NATIVE_ENVELOPE_CLASS

_use_native_envelope = True


def native_envelope_built() -> bool:
    """True when ``cs179._native.DirectionalReachEnvelope`` imported successfully."""
    return _NATIVE_ENVELOPE_CLASS is not None


def use_native_envelope() -> bool:
    """True when native reach-envelope code should run (built and not disabled)."""
    return _use_native_envelope and native_envelope_built()


def set_use_native_envelope(enabled: bool) -> None:
    """Enable or disable native reach-envelope acceleration for this process."""
    global _use_native_envelope
    _use_native_envelope = enabled


def _active_native_envelope_class():
    if use_native_envelope():
        return _NATIVE_ENVELOPE_CLASS
    return None


class DirectionalReachEnvelope:
    """Max tool reach per direction (theta, phi) from base; built via Monte Carlo FK samples.

    Resolution is set by ``n_theta`` × ``n_phi`` (the bin grid). ``n_samples`` controls how
    thoroughly random configurations explore that grid—not mesh tessellation density.
    """

    def __init__(self, bin_radii: np.ndarray):
        self.bin_radii = np.asarray(bin_radii, dtype=np.float64)
        self._native = None
        native_cls = _active_native_envelope_class()
        if native_cls is not None:
            self._native = native_cls(self.bin_radii)
        n_theta, n_phi = self.bin_radii.shape
        theta_pts = (np.arange(n_theta) + 0.5) * np.pi / n_theta
        phi_pts = (np.arange(n_phi) + 0.5) * (2.0 * np.pi) / n_phi - np.pi
        fallback = float(np.max(self.bin_radii))
        self._interp = RegularGridInterpolator(
            (theta_pts, phi_pts),
            self.bin_radii,
            bounds_error=False,
            fill_value=fallback,
        )

    @property
    def n_theta(self) -> int:
        return int(self.bin_radii.shape[0])

    @property
    def n_phi(self) -> int:
        return int(self.bin_radii.shape[1])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, bin_radii=self.bin_radii)

    @classmethod
    def load(cls, path: Path | str) -> DirectionalReachEnvelope:
        with np.load(path) as data:
            return cls(data["bin_radii"])

    @classmethod
    def from_robot(
        cls,
        model,
        data,
        frame_id: int,
        *,
        n_samples: int,
        n_theta: int = REACH_BINS_THETA,
        n_phi: int = REACH_BINS_PHI,
        batch_size: int = REACH_BUILD_BATCH_SIZE,
        show_progress: bool = True,
    ) -> DirectionalReachEnvelope:
        bin_radii = _build_bin_radii_batched(
            model,
            data,
            frame_id,
            n_samples=n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return cls(bin_radii)

    @classmethod
    def from_robot_cached(
        cls,
        model,
        data,
        frame_id: int,
        *,
        robot_key: str,
        frame_name: str | None = None,
        n_samples: int,
        n_theta: int = REACH_BINS_THETA,
        n_phi: int = REACH_BINS_PHI,
        cache_dir: Path | str = REACH_CACHE_DIR,
        force_rebuild: bool = False,
        batch_size: int = REACH_BUILD_BATCH_SIZE,
        show_progress: bool = True,
    ) -> DirectionalReachEnvelope:
        """Load envelope from disk cache, or build and save if missing."""
        if frame_name is None:
            frame_name = model.frames[frame_id].name

        cache_path = envelope_cache_path(
            robot_key,
            frame_name,
            n_theta,
            n_phi,
            n_samples,
            cache_dir=cache_dir,
        )

        if cache_path.exists() and not force_rebuild:
            t0 = time.perf_counter()
            envelope = cls.load(cache_path)
            if envelope.n_theta != n_theta or envelope.n_phi != n_phi:
                raise ValueError(
                    f"Cached envelope shape {envelope.bin_radii.shape} does not match "
                    f"requested ({n_theta}, {n_phi}). Delete {cache_path} or set force_rebuild=True."
                )
            print(
                f"Loaded cached reach envelope from {cache_path} "
                f"({envelope.n_theta}×{envelope.n_phi}, {time.perf_counter() - t0:.2f}s)"
            )
            return envelope

        print(
            f"Building reach envelope for {robot_key}/{frame_name} "
            f"({n_samples:,} FK samples, {n_theta}×{n_phi} bins)..."
        )
        t0 = time.perf_counter()
        envelope = cls.from_robot(
            model,
            data,
            frame_id,
            n_samples=n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        envelope.save(cache_path)
        print(f"Saved reach envelope to {cache_path} ({time.perf_counter() - t0:.1f}s)")
        return envelope

    def reach_limits(self, directions: np.ndarray) -> np.ndarray:
        """Unit directions (N, 3) -> max reach (N,) in each direction."""
        if self._native is not None:
            dirs = np.ascontiguousarray(directions, dtype=np.float64)
            return np.asarray(self._native.reach_limits(dirs), dtype=float)
        dirs = np.asarray(directions, dtype=float)
        if dirs.ndim == 1:
            dirs = dirs.reshape(1, 3)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-9)
        theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
        phi = np.arctan2(dirs[:, 1], dirs[:, 0])
        return self._interp(np.column_stack([theta, phi]))

    def scale_positions(
        self,
        positions: np.ndarray,
        pivot: np.ndarray | None = None,
        safety: float = REACH_SAFETY_MARGIN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp each point onto its radial reach limit; rotations unchanged."""
        if self._native is not None:
            pos_arr = np.ascontiguousarray(positions, dtype=np.float64)
            scaled, scales = self._native.scale_positions(pos_arr, pivot=pivot, safety=safety)
            return np.asarray(scaled, dtype=float), np.asarray(scales, dtype=float)
        pos = np.asarray(positions, dtype=float).copy()
        if pivot is None:
            pivot = np.zeros(3, dtype=float)
        offset = pos - pivot
        radii = np.linalg.norm(offset, axis=1)
        scale = np.ones(len(pos))
        valid = radii > 1e-9
        if np.any(valid):
            limits = self.reach_limits(offset[valid] / radii[valid, None]) * safety
            scale[valid] = np.minimum(1.0, limits / radii[valid])
            pos[valid] = pivot + scale[valid, None] * offset[valid]
        return pos, scale

    def boundary_mesh(
        self,
        pivot: np.ndarray | None = None,
        n_theta: int = 48,
        n_phi: int = 96,
        safety: float = REACH_SAFETY_MARGIN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangulated directional reach boundary as (vertices, faces)."""
        if pivot is None:
            pivot = np.zeros(3, dtype=float)
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
        radii = self.reach_limits(dirs) * safety
        vertices = pivot + dirs * radii[:, None]

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

    def max_radius(self) -> float:
        if self._native is not None:
            return float(self._native.max_radius())
        return float(np.max(self.bin_radii))


def envelope_cache_path(
    robot_key: str,
    frame_name: str,
    n_theta: int,
    n_phi: int,
    n_samples: int,
    *,
    cache_dir: Path | str = REACH_CACHE_DIR,
) -> Path:
    safe_robot = robot_key.replace("/", "_").replace(" ", "_")
    safe_frame = frame_name.replace("/", "_").replace(" ", "_")
    name = f"{safe_robot}_{safe_frame}_{n_theta}x{n_phi}_{n_samples}.npz"
    return Path(cache_dir) / name


def _fill_empty_bins(bin_radii: np.ndarray) -> np.ndarray:
    empty = bin_radii < 1e-6
    if not np.any(empty):
        return bin_radii
    spread = maximum_filter(bin_radii, size=3, mode="nearest")
    bin_radii = np.where(empty, spread, bin_radii)
    still_empty = bin_radii < 1e-6
    if np.any(still_empty):
        bin_radii = np.where(still_empty, float(np.max(bin_radii)), bin_radii)
    return bin_radii


def _accumulate_positions_into_bins(
    positions: np.ndarray,
    bin_radii: np.ndarray,
    n_theta: int,
    n_phi: int,
) -> None:
    """Scatter-max of radii into (theta, phi) bins."""
    radii = np.linalg.norm(positions, axis=1)
    valid = radii >= 1e-6
    if not np.any(valid):
        return

    p = positions[valid]
    r = radii[valid]
    theta = np.arccos(np.clip(p[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(p[:, 1], p[:, 0])
    ti = np.minimum((theta / np.pi * n_theta).astype(np.intp), n_theta - 1)
    pj = ((phi + np.pi) / (2.0 * np.pi) * n_phi).astype(np.intp) % n_phi
    np.maximum.at(bin_radii, (ti, pj), r)


def _build_bin_radii_batched(
    model,
    data,
    frame_id: int,
    *,
    n_samples: int,
    n_theta: int,
    n_phi: int,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    native_cls = _active_native_envelope_class()
    if native_cls is not None:
        native = native_cls.from_robot(
            model,
            data,
            frame_id,
            n_samples=n_samples,
            n_theta=n_theta,
            n_phi=n_phi,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return np.asarray(native.bin_radii, dtype=np.float64)

    bin_radii = np.zeros((n_theta, n_phi), dtype=np.float64)
    batch_size = max(1, min(batch_size, n_samples))
    done = 0
    t0 = time.perf_counter()
    report_every = max(1, n_samples // 20 // batch_size)

    batch_idx = 0
    while done < n_samples:
        n_batch = min(batch_size, n_samples - done)
        positions = np.empty((n_batch, 3), dtype=np.float64)
        for i in range(n_batch):
            q = pin.randomConfiguration(model)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            positions[i] = data.oMf[frame_id].translation

        _accumulate_positions_into_bins(positions, bin_radii, n_theta, n_phi)
        done += n_batch
        batch_idx += 1

        if show_progress and batch_idx % report_every == 0:
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (n_samples - done) / max(rate, 1e-9)
            print(f"  reach envelope: {done:,}/{n_samples:,} samples ({rate:,.0f}/s, ETA {eta:.0f}s)")

    return _fill_empty_bins(bin_radii)


def sample_tool_positions(
    model,
    data,
    frame_id: int,
    n_samples: int,
    *,
    batch_size: int = REACH_BUILD_BATCH_SIZE,
) -> np.ndarray:
    """Monte Carlo tool positions in the robot base frame."""
    points = np.empty((n_samples, 3), dtype=float)
    done = 0
    while done < n_samples:
        n_batch = min(batch_size, n_samples - done)
        for i in range(n_batch):
            q = pin.randomConfiguration(model)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            points[done + i] = data.oMf[frame_id].translation
        done += n_batch
    return points


def scale_cartesian_to_robot(
    cartesian: np.ndarray,
    envelope: DirectionalReachEnvelope,
    pivot: np.ndarray | None = None,
    safety: float = REACH_SAFETY_MARGIN,
) -> tuple[np.ndarray, np.ndarray]:
    out = cartesian.copy()
    out[:, :3], radial_scales = envelope.scale_positions(out[:, :3], pivot=pivot, safety=safety)
    return out, radial_scales
