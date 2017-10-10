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
  jsk_data
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

add_custom_target(install_model ALL COMMAND ${PROJECT_SOURCE_DIR}/scripts/install_model.py)
file(DOWNLOAD https://drive.google.com/file/d/0B09VRnpQxd6PdnphYVZScjkwa1U/view?usp=sharing models/chainer/whole_class_detector/model.npz)
file(DOWNLOAD https://drive.google.com/file/d/0B09VRnpQxd6PYmQwM1lPMGpVQlE/view?usp=sharing models/chainer/whole_class_detector/class_list.txt SHOW_PROGRESS)