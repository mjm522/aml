cmake_minimum_required(VERSION 2.8.3)
project(sawyer_gazebo)

find_package(catkin REQUIRED)

find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  gazebo_ros_control
  controller_manager
  intera_core_msgs
)

find_package(Boost REQUIRED COMPONENTS system)

find_package(gazebo REQUIRED)

catkin_package(
  CATKIN_DEPENDS 
  intera_core_msgs 
  gazebo_ros_control
  roscpp
  controller_manager_msgs
  DEPENDS
  gazebo
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
  ${catkin_LIBRARY_DIRS}
)

include_directories(include
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

add_library(sawyer_gazebo_ros_control
  src/sawyer_gazebo_ros_control_plugin.cpp
)

target_link_libraries(sawyer_gazebo_ros_control
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  gazebo_ros_control
)

add_dependencies(
  sawyer_gazebo_ros_control
  ${catkin_EXPORTED_TARGETS}
)

add_library(intera_gripper_gazebo_ros_control
  src/intera_gripper_ros_control_plugin.cpp
)

target_link_libraries(intera_gripper_gazebo_ros_control
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  gazebo_ros_control
)

add_dependencies(
  intera_gripper_gazebo_ros_control
  ${catkin_EXPORTED_TARGETS}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
