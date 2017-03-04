cmake_minimum_required(VERSION 2.8.3)
project(aml_pisa_hand)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    control_toolbox
    controller_interface
    controller_manager
    hardware_interface
    realtime_tools
    cmake_modules
)

## qbAPI is the default electronics the hand is sold with
SET(QBAPI ./qbAPI)

catkin_package()

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${QBAPI}/src
)

## the qbAPI library
add_library(qbmove SHARED
    ${QBAPI}/src/qbmove_communications.cpp
)

# the ROS-driver
add_executable(pisa_soft_hand
    src/pisa_soft_hand.cpp
)
add_dependencies(pisa_soft_hand
    qbmove
)
target_link_libraries(pisa_soft_hand
    ${catkin_LIBRARIES}
   qbmove
)
