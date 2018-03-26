cmake_minimum_required(VERSION 2.8.3)
project(aml_sawyer_sim_examples)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rospack
  intera_core_msgs
  intera_interface
  gazebo_ros
  gazebo_msgs
  aml_sawyer_gazebo
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  rospack
  intera_core_msgs
  intera_interface
  gazebo_ros
  gazebo_msgs
  aml_sawyer_gazebo
)

catkin_python_setup()

foreach(dir models)
   install(DIRECTORY ${dir}/
      DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

