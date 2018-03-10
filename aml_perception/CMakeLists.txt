cmake_minimum_required(VERSION 2.8.3)
project(aml_perception)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") #-std=c++0x 

# PCL ########################## PCL (apparently) links the required BOOST libraries and Eigen3 also
find_package(PCL 1.7.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
################################

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


catkin_python_setup()


# add_service_files(
#   FILES
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs  # and/or other packages containing depended messages
# )


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs
)


SET(PROJECT_INCLUDE ${PROJECT_SOURCE_DIR}/include)
SET(PROJECT_CODE_SOURCE ${PROJECT_SOURCE_DIR}/src/aml_perception)
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_INCLUDE}
  ${PROJECT_SOURCE_DIR}
)

SET(LINK_LIBS ${PCL_LIBRARIES})

SET(PROJ_LIBS ${PROJECT_CODE_SOURCE}/pcl_processing.cpp)
              # ${PROJECT_CODE_SOURCE}/pcl_utils_server.cpp)

add_library(proj_libs STATIC ${PROJ_LIBS})


add_executable(test_clnt ${PROJECT_SOURCE_DIR}/tests/test_pcl.cpp)
target_link_libraries(test_clnt ${LINK_LIBS} ${catkin_LIBRARIES} proj_libs)
add_dependencies(test_clnt ${PROJECT_NAME}_generate_messages_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(subs ${PROJECT_CODE_SOURCE}/subs.cpp)
# target_link_libraries(subs ${LINK_LIBS} ${catkin_LIBRARIES} proj_libs)

add_executable(pcl_server ${PROJECT_CODE_SOURCE}/pcl_utils_server.cpp)
target_link_libraries(pcl_server ${LINK_LIBS} ${catkin_LIBRARIES} proj_libs )
add_dependencies(pcl_server ${PROJECT_NAME}_generate_messages_cpp)


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
