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
wstool init .
wstool merge aml/3rdparty/baxter/rethink_packages.rosinstall
wstool update
wstool merge sawyer_robot/sawyer_robot.rosinstall
wstool update
rosdep install --from-path . --ignore-src --rosdistro ${ROS_DISTRO} -y -r
cd ..
catkin_make
cp ./src/aml/3rdparty/baxter.sh .
cp ./src/aml/3rdparty/intera.sh .

cd $ROOT_DIR