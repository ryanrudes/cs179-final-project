"""
Per-step transforms for dataset adapters.

Each function in this module is an :data:`~rlds.adapters.base.Transform`: it
accepts one RLDS observation array for a single field, with leading dimension
``T`` (timesteps in the current batch), and returns a float32 array ready to be
written into a DROID-style shard.

Conventions
-----------
* **DROID ``cartesian_position``** uses ``(x, y, z, rx, ry, rz)``: translation in
  metres (or dataset units) plus intrinsic Euler angles in ``xyz`` order (same as
  :func:`scipy.spatial.transform.Rotation.as_euler` with default ``"xyz"``).
* **DROID ``gripper_position``** is shape ``(1,)`` per step (one scalar per
  timestep), not a bare 0-D value.
* **Quaternions** in RLDS pose fields are **scalar-last** ``(x, y, z, w)``, which
  matches :meth:`scipy.spatial.transform.Rotation.from_quat`.

Register transforms on :class:`~rlds.adapters.base.Field` via
``Field.from_rlds(..., transform=...)``. The adapter applies them after slicing
a batch from TensorFlow; you do not need to handle batching yourself beyond the
``T`` dimension.
"""

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "as_float32",
    "pose7_xyzw_to_cartesian6",
    "to_column",
]


def as_float32(array: np.ndarray) -> np.ndarray:
    """
    Cast an RLDS observation array to ``float32``.

    Parameters
    ----------
    array
        Any array-like value from an RLDS step batch.

    Returns
    -------
    numpy.ndarray
        Same shape as ``array``, dtype ``float32``. A copy is made only when the
        input dtype differs.
    """
    return np.asarray(array, dtype=np.float32)


def to_column(array: np.ndarray) -> np.ndarray:
    """
    Reshape a per-step scalar gripper signal to DROID ``gripper_position`` layout.

    DROID stores gripper state as shape ``(1,)`` per timestep. Some datasets
    (e.g. KUKA ``gripper_closed``) provide a scalar or shape ``(T,)`` array
    instead.

    Parameters
    ----------
    array
        Gripper values with leading dimension ``T``. Accepts shape ``(T,)``,
        ``(T, 1)``, or any layout that can be reshaped to ``(T, 1)`` without
        changing element order.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, 1)``, dtype ``float32``.

    Examples
    --------
    >>> to_column(np.array([0.0, 1.0], dtype=np.float32)).shape
    (2, 1)
    """
    return as_float32(array).reshape(-1, 1)


def pose7_xyzw_to_cartesian6(pose7: np.ndarray) -> np.ndarray:
    """
    Convert a 7-D tool pose into DROID ``cartesian_position`` format.

    Maps RT-1 / KUKA-style poses from
    ``clip_function_input/base_pose_tool_reached`` (position + unit quaternion)
    into the six-dimensional layout used by DROID proprioception caches:
    translation plus intrinsic Euler angles.

    Parameters
    ----------
    pose7
        Array of shape ``(T, 7)`` where each row is
        ``[x, y, z, qx, qy, qz, qw]`` (position then quaternion, scalar-last).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(T, 6)`` and dtype ``float32``, each row
        ``[x, y, z, rx, ry, rz]`` with ``(rx, ry, rz)`` from
        :meth:`~scipy.spatial.transform.Rotation.as_euler` with sequence
        ``"xyz"`` (intrinsic rotations about moving X, then Y, then Z).

    Raises
    ------
    ValueError
        If ``pose7`` is not two-dimensional or does not have seven columns.

    Notes
    -----
    * This is an **approximate** correspondence with DROID ``cartesian_position``:
      DROID uses the same 6-D layout, but the original frame and Euler convention
      may differ slightly from KUKA/RT-1. Downstream code should treat adapted
      KUKA poses as targets in the same *tensor layout*, not as guaranteed
      identical kinematic frames.
    * For datasets with no joint encodings, pair this with
      :meth:`~rlds.adapters.base.Field.missing` on ``joint_position``.

    See Also
    --------
    :data:`rlds.adapters.kuka.KUKA` : Adapter that uses this for
    ``cartesian_position``.
    """
    pose7 = as_float32(pose7)
    if pose7.ndim != 2 or pose7.shape[1] != 7:
        raise ValueError(f"Expected pose shape (T, 7); got {pose7.shape}")
    position = pose7[:, :3]
    euler = Rotation.from_quat(pose7[:, 3:7]).as_euler("xyz")
    return np.concatenate([position, euler], axis=1).astype(np.float32, copy=False)
