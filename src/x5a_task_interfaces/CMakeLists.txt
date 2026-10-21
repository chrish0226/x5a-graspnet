cmake_minimum_required(VERSION 3.8)
project(x5a_task_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PickPlace.action"
  "action/PlanRemoteGrasp.action"
  "action/X5aPickPlace.action"
  "action/ColorSequence.action"
  "srv/ParallelIK.srv"
  DEPENDENCIES geometry_msgs moveit_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
