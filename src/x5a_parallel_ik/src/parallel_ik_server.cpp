// A batch is serialized at the ROS boundary. Inside it each thread has a
// separate RobotModelLoader, KDL solver and PlanningScene. No controller client,
// trajectory action or shared MoveIt scene mutation exists in this process.
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <x5a_task_interfaces/srv/parallel_ik.hpp>
#include <openssl/sha.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>

using Batch = x5a_task_interfaces::srv::ParallelIK;
using Code = moveit_msgs::msg::MoveItErrorCodes;
using Clock = std::chrono::steady_clock;

static std::string digest(const std::string& text)
{
  unsigned char result[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(text.data()), text.size(), result);
  std::ostringstream out;
  for (auto byte : result) out << std::hex << std::setw(2) << std::setfill('0') << int(byte);
  return out.str();
}

class ParallelIK : public rclcpp::Node
{
public:
  ParallelIK() : Node("x5a_parallel_ik")
  {
    workers_ = std::clamp<int64_t>(declare_parameter<int>("workers", 4), 1, 4);
    source_ = declare_parameter<std::string>("source_node", "/move_group");
    endpoint_ = declare_parameter<std::string>("service_name", "/x5a_parallel_ik/solve");
  }

  void initialize()
  {
    // Copy the RUNNING model, SRDF and solver/limit parameters, not a second
    // xacro that could silently disagree with the active arm calibration.
    auto remote = std::make_shared<rclcpp::SyncParametersClient>(shared_from_this(), source_);
    if (!remote->wait_for_service(std::chrono::seconds(10)))
      throw std::runtime_error("MoveIt parameter service unavailable");
    auto names = remote->list_parameters({"robot_description", "robot_description_semantic",
      "robot_description_kinematics", "robot_description_planning"}, 20).names;
    auto parameters = remote->get_parameters(names);
    std::string urdf, srdf;
    for (const auto& p : parameters)
    {
      if (p.get_name() == "robot_description") urdf = p.as_string();
      if (p.get_name() == "robot_description_semantic") srdf = p.as_string();
    }
    if (urdf.empty() || srdf.empty()) throw std::runtime_error("Missing active URDF/SRDF");
    digest_ = digest(urdf + std::string(1, '\0') + srdf);
    std::set<const void*> solvers;
    for (int i = 0; i < workers_; ++i)
    {
      rclcpp::NodeOptions options;
      // Keep launch's __node remapping/parameter file on the service node;
      // private model nodes must retain distinct names and copied parameters.
      options.use_global_arguments(false);
      options.start_parameter_services(false);
      options.start_parameter_event_publisher(false);
      options.enable_rosout(false);
      options.parameter_overrides(parameters);
      options.automatically_declare_parameters_from_overrides(true);
      auto node = std::make_shared<rclcpp::Node>("x5a_ik_model_" + std::to_string(i), options);
      auto loader = std::make_shared<robot_model_loader::RobotModelLoader>(node, "robot_description", true);
      const auto model = loader->getModel();
      if (!model) throw std::runtime_error("IK worker robot model failed to load");
      auto group = model->getJointModelGroup("arm");
      if (!group || !group->getSolverInstance()) throw std::runtime_error("IK worker has no arm solver");
      if (!solvers.insert(group->getSolverInstance().get()).second)
        throw std::runtime_error("IK workers unexpectedly share a solver instance");
      nodes_.push_back(node);
      loaders_.push_back(loader); // Keep plugin loaders alive for the process.
      // Build each worker's collision robot once. Rebuilding its mesh BVHs on
      // every batch costs hundreds of milliseconds and hides parallel gains.
      scenes_.push_back(std::make_unique<planning_scene::PlanningScene>(model));
    }
    service_ = create_service<Batch>(endpoint_,
      [this](const Batch::Request::SharedPtr request, Batch::Response::SharedPtr response) {
        solve(*request, *response);
      });
    health_ = create_service<std_srvs::srv::Trigger>("/x5a_parallel_ik/health",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        response->success = true;
        response->message = "independent_workers=" + std::to_string(workers_) + " model_digest=" + digest_;
      });
    RCLCPP_INFO(get_logger(), "PLAN-ONLY IK ready: %d independent solvers, model_digest=%s",
      workers_, digest_.c_str());
  }

private:
  bool validStateMessage(const moveit_msgs::msg::RobotState& state,
                         const moveit::core::RobotModel& model, bool complete) const
  {
    const auto& j = state.joint_state;
    if (j.name.size() != j.position.size()) return false;
    std::set<std::string> seen;
    const auto& variables = model.getVariableNames();
    for (size_t i = 0; i < j.name.size(); ++i)
      if (!std::isfinite(j.position[i]) ||
          std::find(variables.begin(), variables.end(), j.name[i]) == variables.end() ||
          !seen.insert(j.name[i]).second)
        return false;
    if (complete)
      for (const auto& name : model.getVariableNames())
        if (!seen.count(name)) return false;
    return true;
  }

  void solve(const Batch::Request& req, Batch::Response& res)
  {
    auto start = Clock::now();
    res.model_digest = digest_;
    const auto& model = *loaders_.front()->getModel();
    if (req.model_digest != digest_ || req.scene.is_diff || req.scene.robot_state.is_diff ||
        req.scene.robot_model_name != model.getName() ||
        !validStateMessage(req.scene.robot_state, model, true) ||
        req.requests.empty() || req.requests.size() > 32 || req.workers > 4 ||
        !std::isfinite(req.total_budget_s) || req.total_budget_s <= 0 || req.total_budget_s > 10)
    {
      res.message = "Invalid full scene, model digest, seed count or budget";
      return;
    }
    res.error_codes.resize(req.requests.size());
    res.solutions.resize(req.requests.size());
    for (auto& code : res.error_codes) code.val = Code::TIMED_OUT;
    const auto deadline = start + std::chrono::duration<double>(req.total_budget_s);
    std::atomic<size_t> next{0};
    std::atomic<bool> bad_scene{false};
    int count = std::min<int>(workers_, req.workers ? req.workers : workers_);
    std::vector<std::thread> threads;
    for (int w = 0; w < count; ++w)
      threads.emplace_back([&, w]() {
        try
        {
          // Separate collision world, ACM, padding, attached bodies and state
          // for every worker. All copies originate from this identical message.
          auto& scene = *scenes_[w];
          if (!scene.setPlanningSceneMsg(req.scene)) { bad_scene = true; return; }
          for (;;)
          {
            const auto i = next.fetch_add(1);
            if (i >= req.requests.size() || Clock::now() >= deadline) break;
            try { solveOne(scene, req.requests[i], deadline, res.error_codes[i], res.solutions[i]); }
            catch (const std::exception& exc)
            {
              res.error_codes[i].val = Code::FAILURE;
              RCLCPP_ERROR(get_logger(), "IK worker exception: %s", exc.what());
            }
          }
        }
        catch (const std::exception& exc)
        {
          bad_scene = true;
          RCLCPP_ERROR(get_logger(), "IK scene exception: %s", exc.what());
        }
      });
    for (auto& thread : threads) thread.join();
    res.success = !bad_scene;
    res.message = bad_scene ? "Worker failed to import complete planning scene" : "Batch complete; results in request order";
    res.elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
  }

  void solveOne(planning_scene::PlanningScene& scene, const moveit_msgs::msg::PositionIKRequest& req,
                const std::chrono::time_point<Clock, std::chrono::duration<double>>& deadline,
                Code& code, moveit_msgs::msg::RobotState& solution)
  {
    const auto& model = *scene.getRobotModel();
    const auto* group = model.getJointModelGroup(req.group_name);
    if (!group || req.group_name != "arm") { code.val = Code::INVALID_GROUP_NAME; return; }
    if (!model.hasLinkModel(req.ik_link_name)) { code.val = Code::INVALID_LINK_NAME; return; }
    if (!req.avoid_collisions || !req.pose_stamped_vector.empty() ||
        !validStateMessage(req.robot_state, model, false)) { code.val = Code::INVALID_ROBOT_STATE; return; }
    const auto& p = req.pose_stamped.pose;
    const double norm = p.orientation.x*p.orientation.x + p.orientation.y*p.orientation.y +
      p.orientation.z*p.orientation.z + p.orientation.w*p.orientation.w;
    if (!std::isfinite(p.position.x) || !std::isfinite(p.position.y) || !std::isfinite(p.position.z) ||
        !std::isfinite(norm) || std::abs(norm-1.) > .001 || req.timeout.sec < 0 || req.timeout.nanosec >= 1000000000)
    { code.val = Code::INVALID_GOAL_CONSTRAINTS; return; }
    double timeout = req.timeout.sec + req.timeout.nanosec*1e-9;
    timeout = std::min(timeout, std::chrono::duration<double>(deadline-Clock::now()).count());
    if (timeout <= 0) { code.val = Code::TIMED_OUT; return; }
    timeout = std::min(timeout, .5);
    auto state = scene.getCurrentState();
    if (!moveit::core::robotStateMsgToRobotState(req.robot_state, state))
    { code.val = Code::INVALID_ROBOT_STATE; return; }
    state.update();
    Eigen::Isometry3d target;
    tf2::fromMsg(p, target);
    const auto& frame = req.pose_stamped.header.frame_id;
    if (frame != model.getModelFrame())
    {
      if (!scene.knowsFrameTransform(frame)) { code.val = Code::FRAME_TRANSFORM_FAILURE; return; }
      target = scene.getFrameTransform(frame) * target;
    }
    kinematic_constraints::KinematicConstraintSet constraints(scene.getRobotModel());
    if (!constraints.add(req.constraints, scene.getTransforms()))
    { code.val = Code::INVALID_GOAL_CONSTRAINTS; return; }
    const auto validity = [&scene, &constraints, &deadline](moveit::core::RobotState* candidate,
              const moveit::core::JointModelGroup* jmg, const double* positions) {
      if (Clock::now() >= deadline) return false;
      candidate->setJointGroupPositions(jmg, positions);
      candidate->update();
      return candidate->satisfiesBounds(jmg) && !scene.isStateColliding(*candidate, jmg->getName()) &&
        constraints.decide(*candidate).satisfied;
    };
    if (!state.setFromIK(group, target, req.ik_link_name, timeout, validity))
    { code.val = Code::NO_IK_SOLUTION; return; }
    state.update();
    for (size_t i = 0; i < model.getVariableCount(); ++i)
      if (!std::isfinite(state.getVariablePositions()[i]))
      { code.val = Code::NO_IK_SOLUTION; return; }
    const auto actual = state.getGlobalLinkTransform(req.ik_link_name);
    const double distance = (actual.translation()-target.translation()).norm();
    const double angle = Eigen::AngleAxisd(actual.linear().transpose()*target.linear()).angle();
    if (!state.satisfiesBounds(group) || scene.isStateColliding(state, group->getName()) ||
        !constraints.decide(state).satisfied || distance > .003 || angle > .05)
    { code.val = Code::NO_IK_SOLUTION; return; }
    moveit::core::robotStateToRobotStateMsg(state, solution, true);
    code.val = Code::SUCCESS;
  }

  int workers_;
  std::string source_, endpoint_, digest_;
  std::vector<rclcpp::Node::SharedPtr> nodes_;
  std::vector<std::shared_ptr<robot_model_loader::RobotModelLoader>> loaders_;
  std::vector<std::unique_ptr<planning_scene::PlanningScene>> scenes_;
  rclcpp::Service<Batch>::SharedPtr service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr health_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<ParallelIK>();
    node->initialize();
    rclcpp::spin(node);
  }
  catch (const std::exception& exc)
  {
    RCLCPP_FATAL(rclcpp::get_logger("x5a_parallel_ik"), "%s", exc.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
