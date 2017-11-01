cmake_minimum_required(VERSION 2.8.3)
project(sawyer_sim_kinematics)

find_package(catkin
  REQUIRED COMPONENTS
  roscpp
  intera_core_msgs
  sensor_msgs
  tf
  tf_conversions
  kdl_parser
  gazebo_msgs
)

find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  std_msgs
  intera_core_msgs
  gazebo_msgs
  sensor_msgs
  kdl_parser
  LIBRARIES ${PROJECT_NAME}
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)
 
add_library(${PROJECT_NAME} src/arm_kinematics.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(sawyer_kinematics src/position_kinematics.cpp)
target_link_libraries(sawyer_kinematics ${catkin_LIBRARIES} ${PROJECT_NAME})

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
 add_dependencies(sawyer_kinematics
  ${catkin_EXPORTED_TARGETS}
  intera_core_msgs_gencpp
 )
