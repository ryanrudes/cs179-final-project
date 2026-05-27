from ..datasets import RoboticsRldsDatasetUrl
from .base import droid_layout

# Full DROID and the 100-demo subset share the same on-disk proprio layout.
DROID = droid_layout(
    "droid",
    RoboticsRldsDatasetUrl.DROID,
    RoboticsRldsDatasetUrl.DROID_100,
)
