cmake_minimum_required(VERSION 2.8.3)
project(sawyer_sim_hardware)

find_package(catkin REQUIRED)

find_package(catkin 
  REQUIRED COMPONENTS
  std_msgs
  roscpp
  intera_core_msgs
  cv_bridge
  image_transport
  sawyer_sim_kinematics
  tf
)

find_package(Boost REQUIRED COMPONENTS system)
find_package(cmake_modules REQUIRED)

find_package(Eigen REQUIRED)
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    intera_core_msgs
    cv_bridge
    image_transport
    sawyer_sim_kinematics
  LIBRARIES ${PROJECT_NAME}
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
)

add_executable(sawyer_emulator src/sawyer_emulator.cpp)
target_link_libraries(sawyer_emulator ${catkin_LIBRARIES})
add_dependencies(sawyer_emulator ${catkin_EXPORTED_TARGETS})

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
