from __future__ import annotations

from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

import pinocchio as pin
import numpy as np

from .config import RetargetConfig, load_retarget_config

_NATIVE_RETARGETER_CLASS = None
try:
    from cs179._native import Retargeter as _NativeRetargeterClass

    _NATIVE_RETARGETER_CLASS = _NativeRetargeterClass
except ImportError:
    pass

_NATIVE_RETARGETER = _NATIVE_RETARGETER_CLASS

_use_native_retarget = True


def native_retarget_built() -> bool:
    """True when ``cs179._native.Retargeter`` imported successfully."""
    return _NATIVE_RETARGETER_CLASS is not None


def use_native_retarget() -> bool:
    """True when native retargeting should run (built and not disabled)."""
    return _use_native_retarget and native_retarget_built()


def set_use_native_retarget(enabled: bool) -> None:
    """Enable or disable native retargeting for this process."""
    global _use_native_retarget
    _use_native_retarget = enabled


def _default_config() -> RetargetConfig:
    try:
        return load_retarget_config()
    except FileNotFoundError:
        return RetargetConfig()


_DEFAULT_CONFIG = _default_config()
_C = _DEFAULT_CONFIG.cost
_U = _C.units
_W = _C.weights

# Backward-compatible module-level names (values from default YAML when present).
POSITION_ERROR_UNIT = _U.position
ROTATION_ERROR_UNIT = _U.rotation
JOINT_VELOCITY_ERROR_UNIT = _U.joint_velocity
JOINT_ACCELERATION_ERROR_UNIT = _U.joint_acceleration
NEUTRAL_POSE_ERROR_UNIT = _U.neutral_pose
NEUTRAL_POSE_WEIGHT = _W.neutral_pose
POS_WEIGHT = _W.position
ROT_WEIGHT = _W.rotation
ROT_WEIGHT_MIN_SCALE = _W.rotation_min_scale
JOINT_VEL_WEIGHT = _W.joint_velocity
JOINT_ACC_WEIGHT = _W.joint_acceleration
ELBOW_BRANCH_WEIGHT = _W.elbow_branch
ELBOW_BRANCH_MARGIN = _C.elbow_margin
ELBOW_BRANCH_ERROR_UNIT = _U.elbow_side
DEFAULT_CONTROL_HZ = _DEFAULT_CONFIG.control_hz
DISPLAY_FPS = _DEFAULT_CONFIG.display_fps

PANDA_ELBOW_FRAMES = _DEFAULT_CONFIG.frames.panda_elbow
UR3E_ELBOW_FRAMES = _DEFAULT_CONFIG.frames.ur3e_elbow


def expand_per_joint_units(units: np.ndarray, n: int) -> np.ndarray:
    """Broadcast YAML velocity/acceleration units to ``model.nv`` (pad with last entry)."""
    u = np.asarray(units, dtype=float).ravel()
    if u.size == n:
        return u
    if u.size == 1:
        return np.full(n, u[0], dtype=float)
    if u.size < n:
        return np.concatenate([u, np.full(n - u.size, u[-1], dtype=float)])
    return u[:n]


def resolve_tool_frame(model, tool_frame: str) -> str:
    """Pick a tool frame that exists on ``model`` (default YAML uses UR3e ``tool0``)."""
    if model.existFrame(tool_frame):
        return tool_frame
    if tool_frame == "tool0" and model.existFrame("panda_hand"):
        return "panda_hand"
    raise ValueError(
        f"Tool frame {tool_frame!r} not found on robot (nframes={model.nframes}). "
        "Set frames.tool in retarget config (e.g. panda_hand for Franka)."
    )


def tool_frame_id(model, tool_frame: str) -> int:
    """Frame id for reach envelope / FK; raises if Pinocchio would return an invalid id."""
    name = resolve_tool_frame(model, tool_frame)
    frame_id = model.getFrameId(name)
    if frame_id < 0 or frame_id >= model.nframes:
        raise ValueError(f"Invalid tool frame id {frame_id} for frame {name!r}")
    return frame_id


def native_retarget_params(config: RetargetConfig, model) -> dict:
    """C++ ``Retargeter`` overrides with tool/elbow frames and ``nv``-sized velocity units."""
    params = config.to_native_dict()
    params["tool_frame"] = resolve_tool_frame(model, params["tool_frame"])
    elbow = target_elbow_frames(config, model)
    params["elbow_shoulder_frame"] = elbow[0]
    params["elbow_mid_frame"] = elbow[1]
    params["elbow_wrist_frame"] = elbow[2]
    vel = np.asarray(params["joint_velocity_error_unit"], dtype=float)
    if vel.size != model.nv:
        params["joint_velocity_error_unit"] = expand_per_joint_units(vel, model.nv).tolist()
    return params


def target_elbow_frames(config: RetargetConfig, model) -> tuple[str, str, str]:
    """Elbow-branch frame triple on the retarget *target* robot model."""
    for frames in (config.frames.panda_elbow, config.frames.ur3e_elbow):
        if all(model.existFrame(n) for n in frames):
            return frames
    raise ValueError(
        "No elbow frame triple found on target robot. "
        f"Expected one of {config.frames.panda_elbow!r} or {config.frames.ur3e_elbow!r}."
    )


def target_to_se3(target: np.ndarray) -> pin.SE3:
    return pin.SE3(R.from_euler("xyz", target[3:]).as_matrix(), target[:3])


def unwrap_euler_targets(eulers: np.ndarray) -> np.ndarray:
    """Remove 2π jumps per axis (``numpy.unwrap``); values may lie outside [-π, π]."""
    out = np.asarray(eulers, dtype=float).copy()
    for axis in range(out.shape[1]):
        out[:, axis] = np.unwrap(out[:, axis])
    return out


def pose_log6_error(oMf: pin.SE3, oMdes: pin.SE3) -> np.ndarray:
    """6D pose error (translational, rotational) in tool frame — same convention as seed_ik."""
    return pin.log(oMf.inverse() * oMdes).vector


def elbow_side_scalar(
    model,
    data,
    q: np.ndarray,
    frame_names: tuple[str, str, str] = UR3E_ELBOW_FRAMES,
) -> float:
    """Signed elbow offset from the vertical plane through shoulder→wrist (+/- = branch)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    shoulder_id, elbow_id, wrist_id = (model.getFrameId(n) for n in frame_names)
    s = data.oMf[shoulder_id].translation
    e = data.oMf[elbow_id].translation
    w = data.oMf[wrist_id].translation
    sw = w - s
    sw_norm = float(np.linalg.norm(sw))
    if sw_norm < 1e-6:
        return 0.0
    up = np.array([0.0, 0.0, 1.0])
    n = np.cross(sw, up)
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(sw, np.array([0.0, 1.0, 0.0]))
    n /= np.linalg.norm(n)
    return float(np.dot(e - s, n))


def panda_q_from_demo(q_demo: np.ndarray, model) -> np.ndarray:
    q = pin.neutral(model)
    n = min(len(q_demo), 7)
    q[:n] = q_demo[:n]
    return q


def elbow_side_target_from_demo(
    q_demo: np.ndarray,
    panda_model,
    panda_data,
    frame_names: tuple[str, str, str] = PANDA_ELBOW_FRAMES,
    fallback: float = 1.0,
) -> float:
    scalar = elbow_side_scalar(
        panda_model, panda_data, panda_q_from_demo(q_demo, panda_model), frame_names
    )
    if abs(scalar) < 1e-3:
        return fallback
    return float(np.sign(scalar))


def demo_elbow_side_targets(
    joint_positions: np.ndarray,
    panda_model,
    panda_data,
    frame_names: tuple[str, str, str] = PANDA_ELBOW_FRAMES,
) -> np.ndarray:
    sides = np.empty(len(joint_positions))
    last = 1.0
    for i, q_demo in enumerate(joint_positions):
        last = elbow_side_target_from_demo(
            q_demo, panda_model, panda_data, frame_names=frame_names, fallback=last
        )
        sides[i] = last
    return sides


def clamp_configuration(model, q: np.ndarray) -> np.ndarray:
    q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    return pin.normalize(model, q)


def seed_ik(
    model,
    data,
    q0,
    target,
    frame_id,
    config: RetargetConfig | None = None,
):
    """Damped least-squares IK seed (Pinocchio tutorial convention)."""
    cfg = config or _DEFAULT_CONFIG
    q = clamp_configuration(model, q0.copy())
    oMdes = target_to_se3(target)
    ik = cfg.ik_seed
    for _ in range(ik.max_iterations):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        nu = pose_log6_error(data.oMf[frame_id], oMdes)
        if np.linalg.norm(nu) < ik.tolerance:
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id)
        v = J.T @ np.linalg.solve(J @ J.T + ik.damping * np.eye(6), nu)
        q = clamp_configuration(model, pin.integrate(model, q, v * ik.step))
    return q


def rotation_geodesic_error(R_current, euler_target, seq="xyz"):
    R_target = R.from_euler(seq, euler_target).as_matrix()
    R_err = R_target.T @ R_current
    return R.from_matrix(R_err).as_rotvec()


class _PythonRetargeter:
    def __init__(
        self,
        robot: pin.RobotWrapper,
        *,
        control_hz: float,
        config: RetargetConfig,
    ):
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.cfg = config
        self.control_hz = float(control_hz)

        self.robot = robot
        self.model = robot.model
        self.data = self.model.createData()

        self.q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.q_neutral)
        pin.updateFramePlacements(self.model, self.data)

        self.q = self.q_neutral.copy()
        self.q_prev = None
        self.position_scale = 1.0
        self.elbow_side_target = 0.0
        self._tool_frame_id = tool_frame_id(self.model, self.cfg.frames.tool)
        self._elbow_frames = target_elbow_frames(self.cfg, self.model)
        self._joint_velocity_unit = expand_per_joint_units(self.cfg.cost.units.joint_velocity, self.model.nv)

    def set_position_scale(self, scale: float) -> None:
        self.position_scale = float(np.clip(scale, 0.0, 1.0))

    def set_elbow_side_target(self, side: float) -> None:
        self.elbow_side_target = float(np.sign(side)) if abs(side) > 1e-6 else 0.0

    def reset_episode(self, target: np.ndarray):
        self.q = seed_ik(
            self.model,
            self.data,
            self.q_neutral,
            target,
            self._tool_frame_id,
            config=self.cfg,
        )
        self.q_prev = None

    def compute_cost(self, q: np.ndarray, target: np.ndarray) -> float:
        u = self.cfg.cost.units
        w = self.cfg.cost.weights
        q = pin.normalize(self.model, q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        err6 = pose_log6_error(self.data.oMf[self._tool_frame_id], target_to_se3(target))
        pos_residual = err6[:3] / u.position
        rot_residual = err6[3:] / u.rotation

        pos_error = np.dot(pos_residual, pos_residual)
        rot_error = np.dot(rot_residual, rot_residual)

        rot_scale = w.rotation_min_scale + (1.0 - w.rotation_min_scale) * self.position_scale
        rot_term_weight = w.rotation * rot_scale
        cost = w.position * pos_error + rot_term_weight * rot_error

        if self.q_prev is not None:
            dq = pin.difference(self.model, self.q, q)
            vel_residual = (dq * self.control_hz) / self._joint_velocity_unit
            vel_error = np.dot(vel_residual, vel_residual)
            cost += w.joint_velocity * vel_error

            dq_prev = pin.difference(self.model, self.q_prev, self.q)
            dacc = dq - dq_prev
            acc_residual = (dacc * self.control_hz * self.control_hz) / u.joint_acceleration
            acc_error = np.dot(acc_residual, acc_residual)
            cost += w.joint_acceleration * acc_error

        dq_neutral = pin.difference(self.model, self.q_neutral, q) / u.neutral_pose
        neutral_error = np.dot(dq_neutral, dq_neutral)
        cost += w.neutral_pose * neutral_error

        if self.elbow_side_target != 0.0:
            side_scalar = elbow_side_scalar(self.model, self.data, q, self._elbow_frames)
            violation = self.cfg.cost.elbow_margin - self.elbow_side_target * side_scalar
            if violation > 0.0:
                branch_residual = violation / u.elbow_side
                cost += w.elbow_branch * (branch_residual * branch_residual)

        return cost

    def __call__(self, target: np.ndarray):
        opt = self.cfg.optimizer
        result = minimize(
            self.compute_cost,
            self.q,
            args=(target,),
            method=opt.method,
            bounds=list(zip(self.robot.model.lowerPositionLimit, self.robot.model.upperPositionLimit)),
            options={
                "maxiter": opt.max_iterations,
                "ftol": opt.ftol,
            },
        )

        q_new = pin.normalize(self.model, result.x)

        self.q_prev = self.q.copy()
        self.q = q_new.copy()

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        frame_fk = self.data.oMf[self._tool_frame_id]
        err6 = pose_log6_error(frame_fk, target_to_se3(target))

        return (
            q_new,
            frame_fk.translation.copy(),
            R.from_matrix(frame_fk.rotation).as_euler("xyz"),
            float(np.linalg.norm(err6[:3])),
            float(np.linalg.norm(err6[3:])),
            bool(result.success),
            int(result.nit),
        )


class Retargeter:
    """UR3e retargeting; uses native NLopt L-BFGS-B when built, else SciPy."""

    def __init__(
        self,
        robot: pin.RobotWrapper,
        *,
        control_hz: float | None = None,
        config: RetargetConfig | None = None,
    ):
        self.cfg = config or _DEFAULT_CONFIG
        hz = self.cfg.control_hz if control_hz is None else control_hz
        if hz <= 0:
            raise ValueError(f"control_hz must be positive, got {hz}")
        self.control_hz = float(hz)
        self.robot = robot
        self.model = robot.model
        self._native = None
        self._python: _PythonRetargeter | None = None

        if use_native_retarget():
            self._native = _NATIVE_RETARGETER_CLASS(
                self.model, self.control_hz, native_retarget_params(self.cfg, self.model)
            )
        else:
            self._python = _PythonRetargeter(robot, control_hz=self.control_hz, config=self.cfg)

    @property
    def q(self) -> np.ndarray:
        if self._native is not None:
            return np.asarray(self._native.q, dtype=float)
        assert self._python is not None
        return self._python.q

    @property
    def q_prev(self) -> np.ndarray | None:
        if self._native is not None:
            q_prev = self._native.q_prev
            if q_prev is None:
                return None
            return np.asarray(q_prev, dtype=float)
        assert self._python is not None
        return self._python.q_prev

    def set_position_scale(self, scale: float) -> None:
        if self._native is not None:
            self._native.set_position_scale(scale)
            return
        assert self._python is not None
        self._python.set_position_scale(scale)

    def set_elbow_side_target(self, side: float) -> None:
        if self._native is not None:
            self._native.set_elbow_side_target(side)
            return
        assert self._python is not None
        self._python.set_elbow_side_target(side)

    def reset_episode(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        if self._native is not None:
            self._native.reset_episode(target)
            return
        assert self._python is not None
        self._python.reset_episode(target)

    def __call__(self, target: np.ndarray):
        target = np.asarray(target, dtype=float)
        if self._native is not None:
            return self._native(target)
        assert self._python is not None
        return self._python(target)
