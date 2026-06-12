import sys
from rlds import RldsObservationLoader
from retarget.gpu import gpu_retarget_built, load_gpu_fk_model, retarget_cartesian_trajectories, prepare_cartesian_for_gpu_batch, trajectory_fits_gpu_shmem
from retarget.config import load_retarget_config
from robot_descriptions.loaders.pinocchio import load_robot_description
from reachability import DirectionalReachEnvelope, REACH_SAFETY_MARGIN, tool_frame_id

loader = RldsObservationLoader(data_dir="data", dataset_url="droid")
gpu_fk_model = load_gpu_fk_model()
config = load_retarget_config()
robot = load_robot_description("ur3e_description")
reach_data = robot.model.createData()
ur3e_reach = DirectionalReachEnvelope.from_robot_cached(robot.model, reach_data, tool_frame_id(robot.model, config.frames.tool), "ur3e_description")

batch = []
for demo in loader:
    if len(batch) >= 10: break
    if trajectory_fits_gpu_shmem(len(demo[2]), gpu_fk_model.nv):
        batch.append((len(batch), demo[0], demo[2]))

cartesian_list, scales_list = prepare_cartesian_for_gpu_batch(batch, ur3e_reach, reach_safety=REACH_SAFETY_MARGIN)
retarget_cartesian_trajectories(gpu_fk_model, cartesian_list, config, position_scales_list=scales_list)
print("Done")
