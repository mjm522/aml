#!/bin/bash

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
rosdep install --from-path . --ignore-src --rosdistro kinetic -y -r
cd ..
catkin_make
cp ./src/aml/3rdparty/baxter/baxter_sim.sh .
