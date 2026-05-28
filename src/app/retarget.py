from pathlib import Path
from typing import Optional

from retarget.run import run as run_retarget


def run(
    *,
    data_dir: Path,
    dataset_url: str,
    start_demo: int,
    end_demo: Optional[int],
    robot_description: str,
    panda_description: str,
    control_hz: Optional[float],
    enable_visualization: bool,
    show_plots: bool,
    show_progress: bool | None = None,
    reach_n_samples: int,
    reach_n_theta: int,
    reach_n_phi: int,
    reach_force_rebuild: bool,
    reach_safety: float,
    config_path: Optional[Path] = None,
    use_gpu: bool = False,
    save_joints: bool = False,
    save_joints_dir: Optional[Path] = None,
) -> None:
    run_retarget(
        data_dir=data_dir,
        dataset_url=dataset_url,
        start_demo=start_demo,
        end_demo=end_demo,
        robot_description=robot_description,
        panda_description=panda_description,
        control_hz=control_hz,
        enable_visualization=enable_visualization,
        show_plots=show_plots,
        show_progress=show_progress,
        reach_n_samples=reach_n_samples,
        reach_n_theta=reach_n_theta,
        reach_n_phi=reach_n_phi,
        reach_force_rebuild=reach_force_rebuild,
        reach_safety=reach_safety,
        config_path=config_path,
        use_gpu=use_gpu,
        save_joints=save_joints,
        save_joints_dir=save_joints_dir,
    )
