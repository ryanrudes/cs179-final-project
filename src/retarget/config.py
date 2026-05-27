"""Load retargeting parameters from YAML configuration files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class CostUnits:
    position: float = 0.01
    rotation: float = math.radians(5.0)
    joint_velocity: np.ndarray = field(
        default_factory=lambda: np.array(
            [math.pi / 2, math.pi / 2, math.pi / 2, math.pi, math.pi, math.pi],
            dtype=float,
        )
    )
    joint_acceleration: float = math.radians(800.0)
    neutral_pose: float = 2.0 * math.pi
    elbow_side: float = 0.05


@dataclass(frozen=True)
class CostWeights:
    position: float = 1.0
    rotation: float = 1.0
    rotation_min_scale: float = 0.35
    joint_velocity: float = 0.05
    joint_acceleration: float = 0.05
    neutral_pose: float = 0.05
    elbow_branch: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    units: CostUnits = CostUnits()
    weights: CostWeights = CostWeights()
    elbow_margin: float = 0.02


@dataclass(frozen=True)
class IkSeedConfig:
    max_iterations: int = 200
    step: float = 0.5
    damping: float = 1e-4
    tolerance: float = 1e-4


@dataclass(frozen=True)
class OptimizerConfig:
    method: str = "L-BFGS-B"
    max_iterations: int = 400
    ftol: float = 1e-9
    gradient_step_scale: float = 1e-8


@dataclass(frozen=True)
class FramesConfig:
    tool: str = "tool0"
    ur3e_elbow: tuple[str, str, str] = ("shoulder_link", "forearm_link", "wrist_2_link")
    panda_elbow: tuple[str, str, str] = ("panda_link1", "panda_link3", "panda_link5")


@dataclass(frozen=True)
class RetargetConfig:
    display_fps: int = 60
    control_hz: float = 15.0
    cost: CostConfig = CostConfig()
    ik_seed: IkSeedConfig = IkSeedConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    frames: FramesConfig = FramesConfig()

    def to_native_dict(self) -> dict[str, Any]:
        """Parameter overrides for ``cs179._native.Retargeter`` (C++ internal names)."""
        u = self.cost.units
        w = self.cost.weights
        ik = self.ik_seed
        opt = self.optimizer
        return {
            "position_error_unit": u.position,
            "rotation_error_unit": u.rotation,
            "joint_velocity_error_unit": u.joint_velocity.tolist(),
            "joint_acceleration_error_unit": u.joint_acceleration,
            "neutral_pose_error_unit": u.neutral_pose,
            "neutral_pose_weight": w.neutral_pose,
            "pos_weight": w.position,
            "rot_weight": w.rotation,
            "rot_weight_min_scale": w.rotation_min_scale,
            "joint_vel_weight": w.joint_velocity,
            "joint_acc_weight": w.joint_acceleration,
            "elbow_branch_weight": w.elbow_branch,
            "elbow_branch_margin": self.cost.elbow_margin,
            "elbow_branch_error_unit": u.elbow_side,
            "seed_ik_n_iter": ik.max_iterations,
            "seed_ik_dt": ik.step,
            "seed_ik_damp": ik.damping,
            "seed_ik_convergence_tol": ik.tolerance,
            "solver_max_iter": opt.max_iterations,
            "solver_ftol": opt.ftol,
            "solver_grad_eps": opt.gradient_step_scale,
            "tool_frame": self.frames.tool,
        }


def _section(raw: dict, key: str) -> dict:
    block = raw.get(key) or {}
    return block if isinstance(block, dict) else {}


def _parse_cost_units(units_raw: dict) -> CostUnits:
    vel_deg_s = units_raw.get("joint_velocity", [90, 90, 90, 180, 180, 180])
    return CostUnits(
        position=float(units_raw.get("position", 0.01)),
        rotation=float(np.deg2rad(units_raw.get("rotation", 5.0))),
        joint_velocity=np.deg2rad(np.asarray(vel_deg_s, dtype=float)),
        joint_acceleration=float(np.deg2rad(units_raw.get("joint_acceleration", 800.0))),
        neutral_pose=float(units_raw.get("neutral_pose", 2.0 * math.pi)),
        elbow_side=float(units_raw.get("elbow_side", 0.05)),
    )


def _parse_cost_weights(weights_raw: dict) -> CostWeights:
    return CostWeights(
        position=float(weights_raw.get("position", 1.0)),
        rotation=float(weights_raw.get("rotation", 1.0)),
        rotation_min_scale=float(weights_raw.get("rotation_min_scale", 0.35)),
        joint_velocity=float(weights_raw.get("joint_velocity", 0.05)),
        joint_acceleration=float(weights_raw.get("joint_acceleration", 0.05)),
        neutral_pose=float(weights_raw.get("neutral_pose", 0.05)),
        elbow_branch=float(weights_raw.get("elbow_branch", 0.0)),
    )


def default_config_path() -> Path:
    """Preferred default config file (repo ``config/default.yaml`` or cwd)."""
    cwd_candidate = Path("config/default.yaml")
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(
        "No retarget config found. Expected config/default.yaml in the working directory "
        f"or at {DEFAULT_CONFIG_PATH}"
    )


def load_retarget_config(path: Path | str | None = None) -> RetargetConfig:
    """Load retarget parameters from YAML; uses :func:`default_config_path` when *path* is None."""
    config_path = default_config_path() if path is None else Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Retarget config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    runtime = _section(raw, "runtime")
    cost_raw = _section(raw, "cost")
    ik_raw = _section(raw, "ik_seed")
    opt_raw = _section(raw, "optimizer")
    frames_raw = _section(raw, "frames")

    units_raw = _section(cost_raw, "units")
    weights_raw = _section(cost_raw, "weights")

    ur3e_elbow = frames_raw.get("ur3e_elbow", ["shoulder_link", "forearm_link", "wrist_2_link"])
    panda_elbow = frames_raw.get("panda_elbow", ["panda_link1", "panda_link3", "panda_link5"])

    return RetargetConfig(
        display_fps=int(runtime.get("display_fps", 60)),
        control_hz=float(runtime.get("control_hz", 15.0)),
        cost=CostConfig(
            units=_parse_cost_units(units_raw),
            weights=_parse_cost_weights(weights_raw),
            elbow_margin=float(cost_raw.get("elbow_margin", 0.02)),
        ),
        ik_seed=IkSeedConfig(
            max_iterations=int(ik_raw.get("max_iterations", 200)),
            step=float(ik_raw.get("step", 0.5)),
            damping=float(ik_raw.get("damping", 1e-4)),
            tolerance=float(ik_raw.get("tolerance", 1e-4)),
        ),
        optimizer=OptimizerConfig(
            method=str(opt_raw.get("method", "L-BFGS-B")),
            max_iterations=int(opt_raw.get("max_iterations", 400)),
            ftol=float(opt_raw.get("ftol", 1e-9)),
            gradient_step_scale=float(opt_raw.get("gradient_step_scale", 1e-8)),
        ),
        frames=FramesConfig(
            tool=str(frames_raw.get("tool", "tool0")),
            ur3e_elbow=tuple(str(x) for x in ur3e_elbow),
            panda_elbow=tuple(str(x) for x in panda_elbow),
        ),
    )
