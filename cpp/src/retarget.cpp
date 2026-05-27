#include "cs179/retarget.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <nlopt.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cs179 {
namespace {

Eigen::VectorXd damped_least_squares_velocity(
    const Eigen::Ref<const Eigen::Matrix<double, 6, Eigen::Dynamic>>& J,
    const Eigen::Matrix<double, 6, 1>& nu,
    double damping) {
    const Eigen::Matrix<double, 6, 6> damped =
        J * J.transpose() + damping * Eigen::Matrix<double, 6, 6>::Identity();
    // Match NumPy ``linalg.solve`` (unpivoted QR / dgesv); LDLT and col-piv QR diverge on ill-conditioned JJt.
    const Eigen::Matrix<double, 6, 1> x = damped.householderQr().solve(nu);
    return J.transpose() * x;
}

/// Intrinsic xyz Euler (matches ``scipy.spatial.transform.Rotation.from_euler("xyz")``): R = Rz * Ry * Rx.
Eigen::Matrix3d euler_xyz_to_matrix(const Eigen::Vector3d& euler) {
    return (Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX()))
        .toRotationMatrix();
}

Eigen::Vector3d matrix_to_euler_xyz(const Eigen::Matrix3d& rotation) {
    const Eigen::Vector3d zyx = rotation.eulerAngles(2, 1, 0);
    return Eigen::Vector3d(zyx[2], zyx[1], zyx[0]);
}

pinocchio::SE3 target_to_se3(const Eigen::Ref<const Eigen::VectorXd>& target6) {
    if (target6.size() != 6) {
        throw std::invalid_argument("target must have length 6");
    }
    return pinocchio::SE3(euler_xyz_to_matrix(target6.tail<3>()), target6.head<3>());
}

Eigen::Matrix<double, 6, 1> pose_log6_error(const pinocchio::SE3& oMf, const pinocchio::SE3& oMdes) {
    return pinocchio::log6(oMf.inverse() * oMdes).toVector();
}

Eigen::VectorXd clamp_configuration(const pinocchio::Model& model, const Eigen::VectorXd& q_in) {
    Eigen::VectorXd q = q_in;
    q = q.cwiseMax(model.lowerPositionLimit).cwiseMin(model.upperPositionLimit);
    pinocchio::normalize(model, q);
    return q;
}

struct NloptContext {
    Retargeter* retargeter = nullptr;
    Eigen::VectorXd target6;
};

double nlopt_objective(unsigned n, const double* x, double* grad, void* data) {
    auto* ctx = static_cast<NloptContext*>(data);
    Eigen::Map<const Eigen::VectorXd> q_map(x, static_cast<Eigen::Index>(n));
    Eigen::VectorXd grad_vec;
    Eigen::VectorXd* grad_out = nullptr;
    if (grad != nullptr) {
        grad_vec.resize(static_cast<Eigen::Index>(n));
        grad_out = &grad_vec;
    }
    const double cost = ctx->retargeter->evaluate_objective(q_map, ctx->target6, grad_out);
    if (grad != nullptr) {
        Eigen::Map<Eigen::VectorXd>(grad, static_cast<Eigen::Index>(n)) = grad_vec;
    }
    return cost;
}

}  // namespace

Eigen::VectorXd seed_ik(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q0,
    const Eigen::Ref<const Eigen::VectorXd>& target6,
    const pinocchio::FrameIndex frame_id,
    const RetargetParams& params) {
    Eigen::VectorXd q = clamp_configuration(model, q0);
    const pinocchio::SE3 oMdes = target_to_se3(target6);
    for (int iter = 0; iter < params.seed_ik_n_iter; ++iter) {
        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacements(model, data);
        const Eigen::Matrix<double, 6, 1> nu = pose_log6_error(data.oMf[frame_id], oMdes);
        if (nu.norm() < params.seed_ik_convergence_tol) {
            break;
        }
        Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model.nv);
        pinocchio::computeFrameJacobian(model, data, q, frame_id, pinocchio::LOCAL, J);
        const Eigen::VectorXd v = damped_least_squares_velocity(J, nu, params.seed_ik_damp);
        q = clamp_configuration(model, pinocchio::integrate(model, q, v * params.seed_ik_dt));
    }
    return q;
}

namespace {

Eigen::VectorXd expand_per_joint_units(const Eigen::VectorXd& units, int n) {
    if (units.size() == n) {
        return units;
    }
    if (units.size() == 1) {
        return Eigen::VectorXd::Constant(n, units[0]);
    }
    if (units.size() < n) {
        Eigen::VectorXd out(n);
        out.head(units.size()) = units;
        out.tail(n - units.size()).setConstant(units[units.size() - 1]);
        return out;
    }
    return units.head(n);
}

}  // namespace

Retargeter::Retargeter(const pinocchio::Model& model, double control_hz, RetargetParams params)
    : model_(model),
      data_(model),
      params_(std::move(params)),
      tool0_frame_id_(model.getFrameId(params_.tool_frame)),
      elbow_shoulder_id_(model.getFrameId(params_.elbow_shoulder_frame)),
      elbow_mid_id_(model.getFrameId(params_.elbow_mid_frame)),
      elbow_wrist_id_(model.getFrameId(params_.elbow_wrist_frame)) {
    if (control_hz <= 0.0) {
        throw std::invalid_argument("control_hz must be positive");
    }
    if (tool0_frame_id_ >= model_.nframes) {
        throw std::invalid_argument("tool frame not found: " + params_.tool_frame);
    }
    params_.joint_velocity_error_unit =
        expand_per_joint_units(params_.joint_velocity_error_unit, model_.nv);
    control_hz_ = control_hz;
    q_neutral_ = pinocchio::neutral(model_);
    q_ = q_neutral_;
}

void Retargeter::set_position_scale(const double scale) {
    position_scale_ = std::clamp(scale, 0.0, 1.0);
}

void Retargeter::set_elbow_side_target(const double side) {
    elbow_side_target_ = std::abs(side) > 1e-6 ? (side > 0.0 ? 1.0 : -1.0) : 0.0;
}

void Retargeter::reset_episode(const Eigen::Ref<const Eigen::VectorXd>& target6) {
    q_ = seed_ik(model_, data_, q_neutral_, target6, tool0_frame_id_, params_);
    q_prev_.reset();
}

double Retargeter::elbow_side_scalar(const Eigen::VectorXd& q) {
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);
    const Eigen::Vector3d s = data_.oMf[elbow_shoulder_id_].translation();
    const Eigen::Vector3d e = data_.oMf[elbow_mid_id_].translation();
    const Eigen::Vector3d w = data_.oMf[elbow_wrist_id_].translation();
    const Eigen::Vector3d sw = w - s;
    const double sw_norm = sw.norm();
    if (sw_norm < 1e-6) {
        return 0.0;
    }
    Eigen::Vector3d up(0.0, 0.0, 1.0);
    Eigen::Vector3d n = sw.cross(up);
    if (n.norm() < 1e-6) {
        n = sw.cross(Eigen::Vector3d(0.0, 1.0, 0.0));
    }
    n.normalize();
    return (e - s).dot(n);
}

double Retargeter::compute_cost(
    const Eigen::VectorXd& q_in,
    const Eigen::Ref<const Eigen::VectorXd>& target6) {
    Eigen::VectorXd q = q_in;
    pinocchio::normalize(model_, q);
    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    const pinocchio::SE3 oMdes = target_to_se3(target6);
    const Eigen::Matrix<double, 6, 1> err6 = pose_log6_error(data_.oMf[tool0_frame_id_], oMdes);
    const Eigen::Vector3d pos_residual = err6.head<3>() / params_.position_error_unit;
    const Eigen::Vector3d rot_residual = err6.tail<3>() / params_.rotation_error_unit;
    const double pos_error = pos_residual.squaredNorm();
    const double rot_error = rot_residual.squaredNorm();
    const double rot_weight = params_.rot_weight *
        (params_.rot_weight_min_scale + (1.0 - params_.rot_weight_min_scale) * position_scale_);
    double cost = params_.pos_weight * pos_error + rot_weight * rot_error;

    if (q_prev_.has_value()) {
        const Eigen::VectorXd dq = pinocchio::difference(model_, q_, q);
        const Eigen::VectorXd vel_residual =
            (dq * control_hz_).cwiseQuotient(params_.joint_velocity_error_unit);
        cost += params_.joint_vel_weight * vel_residual.squaredNorm();

        const Eigen::VectorXd dq_prev = pinocchio::difference(model_, *q_prev_, q_);
        const Eigen::VectorXd dacc = dq - dq_prev;
        const Eigen::VectorXd acc_residual =
            (dacc * control_hz_ * control_hz_) / params_.joint_acceleration_error_unit;
        cost += params_.joint_acc_weight * acc_residual.squaredNorm();
    }

    const Eigen::VectorXd dq_neutral =
        pinocchio::difference(model_, q_neutral_, q) / params_.neutral_pose_error_unit;
    cost += params_.neutral_pose_weight * dq_neutral.squaredNorm();

    if (elbow_side_target_ != 0.0 && params_.elbow_branch_weight > 0.0) {
        const double side_scalar = elbow_side_scalar(q);
        const double violation = params_.elbow_branch_margin - elbow_side_target_ * side_scalar;
        if (violation > 0.0) {
            const double branch_residual = violation / params_.elbow_branch_error_unit;
            cost += params_.elbow_branch_weight * branch_residual * branch_residual;
        }
    }

    return cost;
}

double Retargeter::evaluate_objective(
    const Eigen::VectorXd& q_in,
    const Eigen::Ref<const Eigen::VectorXd>& target6,
    Eigen::VectorXd* grad) {
    const double cost = compute_cost(q_in, target6);
    if (grad == nullptr) {
        return cost;
    }

    grad->resize(model_.nq);
    Eigen::VectorXd q = q_in;
    pinocchio::normalize(model_, q);
    for (Eigen::Index i = 0; i < q.size(); ++i) {
        const double h = params_.solver_grad_eps * std::max(1.0, std::abs(q[i]));
        Eigen::VectorXd q_plus = q;
        Eigen::VectorXd q_minus = q;
        q_plus[i] += h;
        q_minus[i] -= h;
        pinocchio::normalize(model_, q_plus);
        pinocchio::normalize(model_, q_minus);
        const double f_plus = compute_cost(q_plus, target6);
        const double f_minus = compute_cost(q_minus, target6);
        (*grad)[i] = (f_plus - f_minus) / (2.0 * h);
    }
    return cost;
}

RetargetFrameResult Retargeter::retarget_frame(const Eigen::Ref<const Eigen::VectorXd>& target6) {
    if (target6.size() != 6) {
        throw std::invalid_argument("target must have length 6");
    }

    nlopt::opt optimizer(nlopt::LD_LBFGS, static_cast<unsigned>(model_.nq));
    optimizer.set_lower_bounds(std::vector<double>(
        model_.lowerPositionLimit.data(),
        model_.lowerPositionLimit.data() + model_.nq));
    optimizer.set_upper_bounds(std::vector<double>(
        model_.upperPositionLimit.data(),
        model_.upperPositionLimit.data() + model_.nq));
    optimizer.set_maxeval(params_.solver_max_iter);
    optimizer.set_ftol_rel(params_.solver_ftol);

    NloptContext ctx{this, target6};
    optimizer.set_min_objective(nlopt_objective, &ctx);

    std::vector<double> x(model_.nq);
    Eigen::Map<Eigen::VectorXd>(x.data(), model_.nq) = q_;

    double minf = 0.0;
    nlopt::result status = nlopt::FAILURE;
    try {
        status = optimizer.optimize(x, minf);
    } catch (const std::exception&) {
        status = nlopt::FAILURE;
    }

    RetargetFrameResult out;
    out.q_new = Eigen::Map<const Eigen::VectorXd>(x.data(), model_.nq);
    pinocchio::normalize(model_, out.q_new);
    out.nit = static_cast<int>(optimizer.get_numevals());
    out.success = (status > 0);

    q_prev_ = q_;
    q_ = out.q_new;

    pinocchio::forwardKinematics(model_, data_, q_);
    pinocchio::updateFramePlacements(model_, data_);
    const pinocchio::SE3& frame_fk = data_.oMf[tool0_frame_id_];
    const Eigen::Matrix<double, 6, 1> err6 = pose_log6_error(frame_fk, target_to_se3(target6));
    out.translation = frame_fk.translation();
    out.euler_xyz = matrix_to_euler_xyz(frame_fk.rotation());
    out.position_error = err6.head<3>().norm();
    out.rotation_error = err6.tail<3>().norm();

    return out;
}

}  // namespace cs179
