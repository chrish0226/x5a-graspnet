#include "link_policy.hpp"
#include "arx_x5_src/interfaces/InterfacesThread.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <arx5_arm_msg/msg/robot_cmd.hpp>
#include <arx5_arm_msg/msg/robot_status.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

void reject_duplicate_driver() {
  for (const auto& entry : std::filesystem::directory_iterator("/proc")) {
    const auto name = entry.path().filename().string();
    if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos
        || name == std::to_string(getpid())) continue;
    std::ifstream stream(entry.path()/"comm"); std::string comm; std::getline(stream, comm);
    // Linux comm truncates executable names to 15 characters.
    if (comm == "X5Controller" || comm == "wrist_controlle")
      throw std::runtime_error("Another arm driver is running; refuse duplicate CAN control");
  }
}

class WristController : public rclcpp::Node {
 public:
  WristController() : Node("arm") {
    wrist::validate_sdk(); // BEFORE any SDK object that can initialize motors.
    reject_duplicate_driver();
    wrist::configure(declare_parameter("motor_feedback_timeout_ms", 300.0),
                     declare_parameter("other_motor_feedback_timeout_ms", 800.0));
    const auto can = declare_parameter("arm_can_id", "can1");
    auto urdf = declare_parameter("urdf_path", std::string(""));
    if (urdf.empty()) urdf = ament_index_cpp::get_package_share_directory("arx_x5_controller") + "/x5.urdf";
    sdk_ = std::make_unique<arx::x5::InterfacesThread>(urdf, can, 0);
    sdk_->arx_x(500, 2000, 10);
    sdk_->setHomePositions(std::vector<double>(6, 0));
    status_ = create_publisher<arx5_arm_msg::msg::RobotStatus>("/arm_status", 1);
    health_ = create_publisher<std_msgs::msg::String>("/arm/health", rclcpp::QoS(1).transient_local());
    command_ = create_subscription<arx5_arm_msg::msg::RobotCmd>("/arm_cmd", 1,
        [this](arx5_arm_msg::msg::RobotCmd::ConstSharedPtr msg) { command(*msg); });
    recover_ = create_service<std_srvs::srv::Trigger>("/arm/recover",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>, std::shared_ptr<std_srvs::srv::Trigger::Response> reply) {
          observe();
          if (!gate_.recover(wrist::monotonic_ns(), last_command_, 500000000, 500000000)) {
            reply->success = false;
            reply->message = "反馈须稳定500毫秒且旧指令须停止500毫秒；请先取消旧任务";
          } else {
            // Only reopen the command gate. SDK stays in PROTECT until a NEW
            // user-requested task sends a position target; no motor command here.
            reply->success = true;
            reply->message = "反馈已恢复，保护锁已复位；请发起新的准备或抓取任务";
          }
          publish_health();
        });
    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this] { publish(); });
    RCLCPP_INFO(get_logger(), "Wrist driver: feedback timeout 300ms by default; explicit recovery, no trajectory auto-resume");
  }
  ~WristController() override {
    // Single-thread executor has stopped before destruction. Cancel ROS timer
    // before dropping the interface; avoid the vendor on_shutdown reset race.
    if (timer_) timer_->cancel();
    sdk_.reset();
  }
 private:
  bool observe() {
    last_link_ = wrist::snapshot();
    errors_ = sdk_->getErrorCode();
    bool hardware_fault = false;
    for (int e : errors_) {
      // 0x?f means feedback offline; other values are original SDK faults.
      if (e != 0 && (e & 15) != 15) hardware_fault = true;
    }
    gate_.observe(last_link_.all_fresh && !hardware_fault, last_link_.protection_epoch, wrist::monotonic_ns());
    if (hardware_fault) gate_.protected_state = true;
    if (gate_.protected_state) sdk_->setArmStatus(arx::x5::InterfacesThread::PROTECT);
    return gate_.stable(wrist::monotonic_ns(), 200000000) && !gate_.protected_state;
  }
  void command(const arx5_arm_msg::msg::RobotCmd& msg) {
    last_command_ = wrist::monotonic_ns();
    if (!observe()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Command blocked: feedback/protection not recovered; cancel task then call /arm/recover");
      return;
    }
    if (msg.mode != arx::x5::InterfacesThread::POSITION_CONTROL || !std::isfinite(msg.gripper)) return;
    std::vector<double> q(msg.joint_pos.begin(), msg.joint_pos.end());
    for (double v : q) if (!std::isfinite(v)) return;
    double end[6];
    for (int i=0; i<6; ++i) { if (!std::isfinite(msg.end_pos[i])) return; end[i]=msg.end_pos[i]; }
    sdk_->setEndPose(arx::solve::Xyzrpy2Isometry(end));
    sdk_->setJointPositions(q);
    sdk_->setCatch(msg.gripper);
    sdk_->setArmStatus(arx::x5::InterfacesThread::POSITION_CONTROL);
  }
  void publish_health() {
    std::ostringstream out;
    out << "{\"online\":" << (last_link_.all_fresh ? "true" : "false")
        << ",\"protected\":" << (gate_.protected_state ? "true" : "false")
        << ",\"ready\":" << (gate_.stable(wrist::monotonic_ns(), 200000000) && !gate_.protected_state ? "true" : "false")
        << ",\"protection_epoch\":" << last_link_.protection_epoch << ",\"feedback_age_ms\":[";
    for(int i=0;i<7;++i) { if(i) out << ','; out << last_link_.ages_ms[i]; }
    out << "],\"sdk_error_codes\":[";
    for(size_t i=0;i<errors_.size();++i) { if(i) out << ','; out << errors_[i]; }
    out << "]}";
    std_msgs::msg::String msg; msg.data=out.str(); health_->publish(msg);
  }
  void publish() {
    observe();
    arx5_arm_msg::msg::RobotStatus msg;
    msg.header.stamp = now();
    auto xyz = arx::solve::Isometry2Xyzrpy(sdk_->getEndPose());
    auto q = sdk_->getJointPositons(), v = sdk_->getJointVelocities(), c = sdk_->getJointCurrent();
    for(int i=0;i<6;++i) msg.end_pos[i]=xyz.at(i);
    for(int i=0;i<7;++i) { msg.joint_pos[i]=q.at(i); msg.joint_vel[i]=v.at(i); msg.joint_cur[i]=c.at(i); }
    status_->publish(msg); publish_health();
  }
  std::unique_ptr<arx::x5::InterfacesThread> sdk_;
  wrist::RecoveryGate gate_; wrist::Snapshot last_link_{}; int64_t last_command_=0;
  std::vector<int> errors_;
  rclcpp::Publisher<arx5_arm_msg::msg::RobotStatus>::SharedPtr status_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr health_;
  rclcpp::Subscription<arx5_arm_msg::msg::RobotCmd>::SharedPtr command_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr recover_;
  rclcpp::TimerBase::SharedPtr timer_;
};
int main(int argc, char** argv) {
  if (argc==2 && std::string(argv[1])=="--check-sdk") {
    try { wrist::validate_sdk(); std::puts("SDK hash and runtime symbol binding verified; no hardware initialized"); return 0; }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
  }
  rclcpp::init(argc, argv);
  try { auto node=std::make_shared<WristController>(); rclcpp::spin(node); }
  catch (const std::exception& e) { std::fprintf(stderr, "Wrist driver refused: %s\n", e.what()); rclcpp::shutdown(); return 2; }
  rclcpp::shutdown();
}
