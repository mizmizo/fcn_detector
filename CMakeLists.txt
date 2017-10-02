cmake_minimum_required(VERSION 2.8.3)
project(fcn_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  std_msgs
  roscpp
  message_filters
  message_generation
)

find_package(OpenCV REQUIRED)

catkin_python_setup()

add_message_files(
  FILES
  Box.msg
  ScoredBox.msg
  BoxArray.msg
  ScoredBoxArray.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)