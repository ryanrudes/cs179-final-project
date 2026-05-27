from ..datasets import RoboticsRldsDatasetUrl
from .base import DatasetAdapter, Field
from .transforms import pose7_xyzw_to_cartesian6, to_column

KUKA = DatasetAdapter(
    name="kuka",
    datasets=[RoboticsRldsDatasetUrl.KUKA],
    fields={
        "joint_position": Field.missing(7),
        "gripper_position": Field.from_rlds(
            "gripper_closed",
            1,
            transform=to_column,
        ),
        "cartesian_position": Field.from_rlds(
            "clip_function_input/base_pose_tool_reached",
            6,
            transform=pose7_xyzw_to_cartesian6,
        ),
    },
    notes={
        "joint_position_available": False,
        "cartesian_position_format": "xyz_euler_xyz",
    },
)
