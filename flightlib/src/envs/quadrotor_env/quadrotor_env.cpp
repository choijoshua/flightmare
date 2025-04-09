#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                   std::string("/flightpy/configs/control/config.yaml"),
                 0) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path, const int env_id)
  : EnvBase() {
  // check if configuration file exist
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }
  // load configuration file
  cfg_ = YAML::LoadFile(cfg_path);
  //
  init();
  env_id_ = env_id;
}

QuadrotorEnv::QuadrotorEnv(const YAML::Node &cfg_node, const int env_id)
  : EnvBase(), cfg_(cfg_node) {
  //
  init();
  env_id_ = env_id;
}

void QuadrotorEnv::init() {
  // load parameters
  loadParam(cfg_);
  init_pos_ << init_pos[0], init_pos[1], init_pos[2];
  init_ori_ << init_ori[0], init_ori[1], init_ori[2];
  end_pos_ << end_pos[0], end_pos[1], end_pos[2];
  end_ori_ << end_ori[0], end_ori[1], end_ori[2];
  current_gate_idx = 0;
  quad_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_->updateDynamics(dynamics);

  // define a bounding box {xmin, xmax, ymin, ymax, zmin, zmax}
  world_box_ << -25, 25, -25, 25, -0.0, 20;
  if (!quad_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;


  // add camera
  if (use_camera_) {
    rgb_camera_ = std::make_shared<RGBCamera>();
    if (!configCamera(cfg_, rgb_camera_)) {
      logger_.error(
        "Cannot config RGB Camera. Something wrong with the config file");
    };

    quad_ptr_->addRGBCamera(rgb_camera_);
    //
    img_width_ = rgb_camera_->getWidth();
    img_height_ = rgb_camera_->getHeight();
    rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                              CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
    depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  }

  // use single rotor control or bodyrate control
  if (rotor_ctrl_ == Command::SINGLEROTOR) {
    act_mean_ = Vector<quadenv::kNAct>::Ones() *
                quad_ptr_->getDynamics().getSingleThrustMax() / 2;
    act_std_ = Vector<quadenv::kNAct>::Ones() *
               quad_ptr_->getDynamics().getSingleThrustMax() / 2;
  } else if (rotor_ctrl_ == Command::THRUSTRATE) {
    Scalar max_force = quad_ptr_->getDynamics().getForceMax();
    Vector<3> max_omega = quad_ptr_->getDynamics().getOmegaMax();
    //
    act_mean_ << (max_force / quad_ptr_->getMass()) / 2, 0.0, 0.0, 0.0;
    act_std_ << (max_force / quad_ptr_->getMass()) / 2, max_omega.x(),
      max_omega.y(), max_omega.z();
  }
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs) {
  quad_state_.setZero();
  pi_act_.setZero();

  // randomly reset the quadrotor state
  // reset position
  quad_state_.x(QS::POSX) = init_pos_[0];
  quad_state_.x(QS::POSY) = init_pos_[1];
  quad_state_.x(QS::POSZ) = init_pos_[2];
  // reset linear velocity
  quad_state_.x(QS::VELX) = 0;
  quad_state_.x(QS::VELY) = 0;
  quad_state_.x(QS::VELZ) = 0;
  // reset orientation
  Quaternion quat;
  eulerToQuaternion(quat, init_ori_);
  quad_state_.x(QS::ATTW) = quat.w();
  quad_state_.x(QS::ATTX) = quat.x();
  quad_state_.x(QS::ATTY) = quat.y();
  quad_state_.x(QS::ATTZ) = quat.z();
  quad_state_.qx /= quad_state_.qx.norm();
  quad_ptr_->reset(quad_state_);

  current_gate_idx = 0;

  // reset control command
  cmd_.t = 0.0;
  cmd_.setCmdMode(Command::SINGLEROTOR);
  if (rotor_ctrl_ == Command::SINGLEROTOR) {
    cmd_.thrusts.setZero();
  } else if (rotor_ctrl_ == Command::THRUSTRATE) {
    cmd_.setCmdMode(Command::THRUSTRATE);
    cmd_.collective_thrust = 0;
    cmd_.omega.setZero();
  }

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, bool random) { return reset(obs); }

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  if (obs.size() != obs_dim_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(), obs_dim_);
    return false;
  }

  quad_ptr_->getState(&quad_state_);
  Vector<3> P_w_gw;
  Quaternion q_wg;

  if (current_gate_idx >= num_gates) {
    P_w_gw = end_pos_;
    eulerToQuaternion(q_wg, end_ori_);
  }
  else {
    P_w_gw = gates_[current_gate_idx].pos;
    eulerToQuaternion(q_wg, gates_[current_gate_idx].ori);
  }

  Matrix<3,3> R_wg = q_wg.toRotationMatrix();

  Matrix<3,3> R_wg_inv;
  if (fabs(R_wg.determinant()) < 1e-6) {
    logger_.warn("Near-singular gate rotation matrix, skipping observation.");
    return false;
  }
  R_wg_inv = R_wg.inverse();

  Vector<3> P_g_bg = R_wg_inv * (quad_state_.p - P_w_gw);
  Vector<3> V_g_bg = R_wg_inv * quad_state_.v;
  Matrix<3,3> R_gb_matrix = (R_wg_inv * quad_state_.R()).eval();
  Vector<9> R_gb = Map<Vector<>>(R_gb_matrix.data(), R_gb_matrix.size());

  Vector<3> P_g_g1g = Vector<3>::Zero();
  Vector<9> R_gg1; R_gg1 << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  if (current_gate_idx < num_gates - 1) {
    Vector<3> P_w_g1w = gates_[current_gate_idx + 1].pos;
    Quaternion q_wg1;
    eulerToQuaternion(q_wg1, gates_[current_gate_idx + 1].ori);
    Matrix<3,3> R_wg1 = q_wg1.toRotationMatrix();

    if (fabs(R_wg1.determinant()) < 1e-6) {
      logger_.warn("Near-singular next-gate rotation matrix, skipping observation.");
      return false;
    }

    P_g_g1g = R_wg_inv * (P_w_g1w - P_w_gw);
    Matrix<3,3> R_gg1_matrix = (R_wg_inv * R_wg1).eval();
    R_gg1 = Map<Vector<>>(R_gg1_matrix.data(), R_gg1_matrix.size());
  }

  // Compose final obs
  obs.segment<quadenv::kNObs>(quadenv::kObs) << P_g_bg, R_gb, V_g_bg, P_g_g1g, R_gg1;

  // Check for NaNs, infs
  if (!obs.allFinite()) {
    logger_.error("Non-finite observation detected! obs = %s", obs.transpose().format(Eigen::IOFormat()));
    return false;
  }

  // Optional: check individual norm safety (just logging for now)
  if (P_g_bg.norm() > 1e6 || V_g_bg.norm() > 1e6) {
    logger_.warn("Unusually large P_g_bg or V_g_bg norm. P: %f, V: %f", P_g_bg.norm(), V_g_bg.norm());
    std::cout << P_g_bg << std::endl;
    std::cout << R_wg_inv << std::endl;
    std::cout << quad_state_.p << std::endl;
    std::cout << gates_[current_gate_idx].pos << std::endl;
    std::cout << current_gate_idx << std::endl;
    for (int i = 0; i < num_gates; i++) {
      std::cout << "Gate number " << i << " is " << gates_[i].pos << std::endl;
    }
  }

  return true;
}


bool QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs,
                        Ref<Vector<>> reward) {
  if (!act.allFinite() || act.rows() != act_dim_ || rew_dim_ != reward.rows())
    return false;
  //
  pi_act_ = act.cwiseProduct(act_std_) + act_mean_;

  cmd_.t += sim_dt_;
  quad_state_.t += sim_dt_;

  if (rotor_ctrl_ == Command::SINGLEROTOR) {
    cmd_.thrusts = pi_act_;
  } else if (rotor_ctrl_ == Command::THRUSTRATE) {
    cmd_.collective_thrust = pi_act_(0);
    cmd_.omega = pi_act_.segment<3>(1);
  }
  prev_pos_ = quad_state_.p;
  // simulate quadrotor
  quad_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // ---------------------- reward function design
  // progress reward
  // Compute distances from the current gate position.
  Scalar progress_reward = 0;
  Scalar gate_pass_bonus = 0;
  Scalar tracking_error = 0;
  Scalar current_segment_dist = 0;
  if (current_gate_idx == 0) {
    current_segment_dist = (gates_[current_gate_idx].pos - init_pos_).norm();
  }
  else if (current_gate_idx < num_gates) {
    current_segment_dist = (gates_[current_gate_idx].pos - gates_[current_gate_idx-1].pos).norm();
  }
  else {
    current_segment_dist = (end_pos_ - gates_[current_gate_idx].pos).norm();
  }

  if (current_gate_idx < num_gates) {
    const Scalar prev_dist = (gates_[current_gate_idx].pos - prev_pos_).norm();
    const Scalar curr_dist = (gates_[current_gate_idx].pos - quad_state_.p).norm();
    // Reward progress if the quadrotor moves closer to the gate.
    // This difference will be positive if progress is made.
    progress_reward = progress_coeff_ * (prev_dist - curr_dist);
    // Penalize tracking error as the current distance to the gate.
    tracking_error = tracking_coeff_ * curr_dist/current_segment_dist; // note: tracking_coeff_ should be > 0
    if (curr_dist < gate_pass_threshold) {
      gate_pass_bonus = pass_gate_reward;
      current_gate_idx++;
    }
  }
  else {
    Vector<3> euler;
    quaternionToEuler(quad_state_.q(), euler);
    const Scalar pos_error = (end_pos_ - quad_state_.p).norm();
    const Scalar ori_error = (end_ori_ - euler).norm();
    tracking_error = tracking_coeff_ * pos_error/current_segment_dist + tracking_coeff_ * ori_error;
  }

  const Scalar total_reward =
    progress_reward + tracking_error + time_penalty + gate_pass_bonus;

  reward << progress_reward, tracking_error, gate_pass_bonus, time_penalty, total_reward;
  return true;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = collision_penalty;
    return true;
  }

  const Scalar curr_dist = (end_pos_ - quad_state_.p).norm();

  if (current_gate_idx == num_gates && curr_dist < gate_pass_threshold) {
    reward = terminal_reward;
    return true;
  }

  if (cmd_.t >= max_t_ - sim_dt_) {
    return true;
  }
  return false;
}


bool QuadrotorEnv::getQuadAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && pi_act_.allFinite() && (act.size() == pi_act_.size())) {
    act = pi_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getQuadState(Ref<Vector<>> obs) const {
  if (quad_state_.t >= 0.0 && (obs.rows() == quadenv::kNQuadState)) {
    obs << quad_state_.t, quad_state_.p, quad_state_.qx, quad_state_.v,
      quad_state_.w, quad_state_.a, quad_ptr_->getMotorOmega(),
      quad_ptr_->getMotorThrusts();
    return true;
  }
  logger_.error("Get Quadrotor state failed.");
  return false;
}

bool QuadrotorEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
  if (!rgb_camera_ || !rgb_camera_->getEnabledLayers()[0]) {
    logger_.error(
      "No RGB Camera or depth map is not enabled. Cannot retrieve depth "
      "images.");
    return false;
  }
  rgb_camera_->getDepthMap(depth_img_);

  depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data,
                                    depth_img_.rows * depth_img_.cols);
  return true;
}


bool QuadrotorEnv::getImage(Ref<ImgVector<>> img, const bool rgb) {
  if (!rgb_camera_) {
    logger_.error("No Camera! Cannot retrieve Images.");
    return false;
  }

  rgb_camera_->getRGBImage(rgb_img_);

  if (rgb_img_.rows != img_height_ || rgb_img_.cols != img_width_) {
    logger_.error(
      "Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols "
      "%d != %d",
      rgb_img_.rows, img_height_, rgb_img_.cols, img_width_);
    return false;
  }

  if (!rgb) {
    // converting rgb image to gray image
    cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
    // map cv::Mat data to Eiegn::Vector
    img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
  } else {
    img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols *
                                            rgb_camera_->getChannels());
  }
  return true;
}


bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["simulation"]) {
    sim_dt_ = cfg["simulation"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["simulation"]["max_t"].as<Scalar>();
    rotor_ctrl_ = cfg["simulation"]["rotor_ctrl"].as<int>();
  } else {
    logger_.error("Cannot load [quadrotor_env] parameters");
    return false;
  }

  if (cfg["rewards"]) {
    // load reinforcement learning related parameters
    progress_coeff_ = cfg["rewards"]["progress_coeff"].as<Scalar>();
    tracking_coeff_ = cfg["rewards"]["tracking_coeff"].as<Scalar>();
    collision_penalty = cfg["rewards"]["collision"].as<Scalar>();
    pass_gate_reward = cfg["rewards"]["pass_gate"].as<Scalar>();
    gate_pass_threshold = cfg["rewards"]["gate_pass_threshold"].as<Scalar>();
    time_penalty = cfg["rewards"]["time_penalty"].as<Scalar>();
    terminal_reward = cfg["rewards"]["terminal"].as<Scalar>();
    // load reward settings
    reward_names_ = cfg["rewards"]["names"].as<std::vector<std::string>>();

    rew_dim_ = cfg["rewards"]["names"].as<std::vector<std::string>>().size();
  } else {
    logger_.error("Cannot load [rewards] parameters");
    return false;
  }

  if (cfg["initState"]) {
    init_pos = cfg["initState"]["pos"].as<std::vector<double>>();
    init_ori = cfg["initState"]["rot"].as<std::vector<double>>();
  }
  if (cfg["endState"]) {
    end_pos = cfg["endState"]["pos"].as<std::vector<double>>();
    end_ori = cfg["endState"]["rot"].as<std::vector<double>>();
  }

  if (cfg["gates"]) {
    gates_.clear();
    num_gates = 0;
    std::vector<std::string> gateKeys = cfg["gates"]["gate_order"].as<std::vector<std::string>>();
    for (const std::string &key : gateKeys) {
      Gate gt;
      // Build the full path to the current gate's parameters.
      std::vector<double> pos = cfg["gates"][key]["position"].as<std::vector<double>>();
      std::vector<double> ori = cfg["gates"][key]["rotation"].as<std::vector<double>>();
      // Assign the position and orientation.
      gt.pos << pos[0], pos[1], pos[2];
      gt.ori << ori[0], ori[1], ori[2];
      num_gates++;
      gt.id = num_gates;
      gates_.push_back(gt);
    }
  } else {
    logger_.error("Cannot load [gate] parameters");
    return false;
  }
  return true;
}

bool QuadrotorEnv::configCamera(const YAML::Node &cfg,
                                const std::shared_ptr<RGBCamera> camera) {
  if (!cfg["rgb_camera"]) {
    logger_.error("Cannot config RGB Camera");
    return false;
  }

  if (!cfg["rgb_camera"]["on"].as<bool>()) {
    logger_.warn("Camera is off. Please turn it on.");
    return false;
  }

  if (quad_ptr_->getNumCamera() >= 1) {
    logger_.warn("Camera has been added. Skipping the camera configuration.");
    return false;
  }

  // create camera
  rgb_camera_ = std::make_shared<RGBCamera>();

  // load camera settings
  std::vector<Scalar> t_BC_vec =
    cfg["rgb_camera"]["t_BC"].as<std::vector<Scalar>>();
  std::vector<Scalar> r_BC_vec =
    cfg["rgb_camera"]["r_BC"].as<std::vector<Scalar>>();

  //
  Vector<3> t_BC(t_BC_vec.data());
  Matrix<3, 3> r_BC =
    (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  std::vector<bool> post_processing = {false, false, false};
  post_processing[0] = cfg["rgb_camera"]["enable_depth"].as<bool>();
  post_processing[1] = cfg["rgb_camera"]["enable_segmentation"].as<bool>();
  post_processing[2] = cfg["rgb_camera"]["enable_opticalflow"].as<bool>();

  //
  rgb_camera_->setFOV(cfg["rgb_camera"]["fov"].as<Scalar>());
  rgb_camera_->setWidth(cfg["rgb_camera"]["width"].as<int>());
  rgb_camera_->setChannels(cfg["rgb_camera"]["channels"].as<int>());
  rgb_camera_->setHeight(cfg["rgb_camera"]["height"].as<int>());
  rgb_camera_->setRelPose(t_BC, r_BC);
  rgb_camera_->setPostProcessing(post_processing);


  // add camera to the quadrotor
  quad_ptr_->addRGBCamera(rgb_camera_);

  // adapt parameters
  img_width_ = rgb_camera_->getWidth();
  img_height_ = rgb_camera_->getHeight();
  rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                            CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
  depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  return true;
}

bool QuadrotorEnv::addQuadrotorToUnity(
  const std::shared_ptr<UnityBridge> bridge) {
  if (!quad_ptr_) return false;
  bridge->addQuadrotor(quad_ptr_);
  return true;
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << "]"
     << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib