#pragma once

#include <Eigen/Core>

#include <string>

namespace cs179 {

struct RetargetParams {
    double position_error_unit = 0.01;
    double rotation_error_unit = 5.0 * M_PI / 180.0;
    Eigen::VectorXd joint_velocity_error_unit =
        (Eigen::VectorXd(6) << M_PI / 2, M_PI / 2, M_PI / 2, M_PI, M_PI, M_PI).finished();
    double joint_acceleration_error_unit = 800.0 * M_PI / 180.0;
    double neutral_pose_error_unit = 2.0 * M_PI;
    double neutral_pose_weight = 0.05;
    double pos_weight = 1.0;
    double rot_weight = 1.0;
    double rot_weight_min_scale = 0.35;
    double joint_vel_weight = 0.05;
    double joint_acc_weight = 0.05;
    double elbow_branch_weight = 0.0;
    double elbow_branch_margin = 0.02;
    double elbow_branch_error_unit = 0.05;

    int seed_ik_n_iter = 200;
    double seed_ik_dt = 0.5;
    double seed_ik_damp = 1e-4;
    double seed_ik_convergence_tol = 1e-4;

    int solver_max_iter = 400;
    double solver_ftol = 1e-9;
    double solver_grad_eps = 1e-8;

    std::string tool_frame = "tool0";
    std::string elbow_shoulder_frame = "shoulder_link";
    std::string elbow_mid_frame = "forearm_link";
    std::string elbow_wrist_frame = "wrist_2_link";
};

[[nodiscard]] RetargetParams default_retarget_params();

}  // namespace cs179
