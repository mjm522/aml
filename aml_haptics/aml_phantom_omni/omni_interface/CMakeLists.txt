cmake_minimum_required(VERSION 2.8.3)
project(omni_interface)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


catkin_python_setup()



# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)


#############
## Install ##
#############


#install(
#  DIRECTORY scripts/
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#  USE_SOURCE_PERMISSIONS
#)




#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_manipulation.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()



## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
