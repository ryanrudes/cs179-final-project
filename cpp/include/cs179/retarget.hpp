#pragma once

#include "cs179/retarget_params.hpp"

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <Eigen/Core>

#include <optional>

namespace cs179 {

struct RetargetFrameResult {
    Eigen::VectorXd q_new;
    Eigen::Vector3d translation;
    Eigen::Vector3d euler_xyz;
    double position_error = 0.0;
    double rotation_error = 0.0;
    bool success = false;
    int nit = 0;
};

/// Damped least-squares IK seed (matches ``retarget.core.seed_ik``).
[[nodiscard]] Eigen::VectorXd seed_ik(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q0,
    const Eigen::Ref<const Eigen::VectorXd>& target6,
    pinocchio::FrameIndex frame_id,
    const RetargetParams& params = default_retarget_params());

/// UR3e retargeting optimizer (Pinocchio FK + NLopt L-BFGS-B), mirroring ``retarget.core.Retargeter``.
class Retargeter {
public:
    Retargeter(
        const pinocchio::Model& model,
        double control_hz,
        RetargetParams params = default_retarget_params());

    void set_position_scale(double scale);
    void set_elbow_side_target(double side);
    void reset_episode(const Eigen::Ref<const Eigen::VectorXd>& target6);
    RetargetFrameResult retarget_frame(const Eigen::Ref<const Eigen::VectorXd>& target6);

    [[nodiscard]] const Eigen::VectorXd& q() const { return q_; }
    [[nodiscard]] const Eigen::VectorXd* q_prev() const { return q_prev_.has_value() ? &(*q_prev_) : nullptr; }

    /// NLopt objective: returns cost; fills ``grad`` when non-null (central finite differences).
    double evaluate_objective(
        const Eigen::VectorXd& q,
        const Eigen::Ref<const Eigen::VectorXd>& target6,
        Eigen::VectorXd* grad);

private:
    double compute_cost(const Eigen::VectorXd& q, const Eigen::Ref<const Eigen::VectorXd>& target6);
    double elbow_side_scalar(const Eigen::VectorXd& q);

    const pinocchio::Model& model_;
    pinocchio::Data data_;
    RetargetParams params_;
    pinocchio::FrameIndex tool0_frame_id_;
    pinocchio::FrameIndex elbow_shoulder_id_;
    pinocchio::FrameIndex elbow_mid_id_;
    pinocchio::FrameIndex elbow_wrist_id_;

    double control_hz_ = 15.0;
    Eigen::VectorXd q_neutral_;
    Eigen::VectorXd q_;
    std::optional<Eigen::VectorXd> q_prev_;
    double position_scale_ = 1.0;
    double elbow_side_target_ = 0.0;
};

}  // namespace cs179
