#!/bin/sh

##################################################################################
######### What this script does ##################################################
##################################################################################
# 0) This scripts assumes: install_{ROS_DIST}_deps.sh was successfully run
### 0.1) It also assumes "aml" has been cloned inside a catkin workspace, e.g. 
######## you have cloned "aml" located at aml_ws/src/aml
# 1) If conditons (0) are met, then it proceeds to setup the required packages for baxter and sawyer robots

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $ROOT_DIR
cd ../..
rm .rosinstall*
rm -rf baxter*
rm -rf intera*
rm -rf sawyer*
rm -rf aruco_ros
rm -rf pisa-iit-soft-hand
wstool init .

if [ "$ROS_DISTRO" == "indigo" ]; then
  wstool merge aml/3rdparty/baxter/rethink_packages111.rosinstall
  BLACK_LISTED_PKGS_ARG='-DCATKIN_BLACKLIST_PACKAGES="gazebo_ros_soft_hand;soft_hand_ros_control"'
elif [ "$ROS_DISTRO" == "kinetic" ]; then
  wstool merge aml/3rdparty/baxter/rethink_packages.rosinstall
  BLACK_LISTED_PKGS_ARG='-DCATKIN_BLACKLIST_PACKAGES="aml_sawyer_sim;aml_sawyer_simulator;aml_sawyer_sim_controllers;aml_sawyer_gazebo;aml_sawyer_sim_hardware;aml_sawyer_sim_kinematics"'
  #;baxter_simulator;baxter_sim_hardware;baxter_sim_controllers;baxter_gazebo;baxter_sim_io
else
    echo "ROS_DISTRO is not indigo or kinetic, may not be fully compatible"
fi

wstool update
wstool merge sawyer_robot/sawyer_robot.rosinstall
wstool merge sawyer_simulator/sawyer_simulator.rosinstall
wstool update
rosdep install --from-path . --ignore-src --rosdistro ${ROS_DISTRO} -y -r
cd ..

# 
catkin_make ${BLACK_LISTED_PKGS_ARG}
cp ./src/aml/3rdparty/baxter.sh .
cp ./src/aml/3rdparty/intera.sh .

cd $ROOT_DIR