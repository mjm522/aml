#!/bin/bash

# Don't forget: xhost +
# Modify mounting location as you like

ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"

shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh

xdocker run -h moveit -it --rm \
       --net rosnet --name moveit\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=moveit \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ${extra_params} \
       -v /home/ahayashi/work:/root/work \
       -w /root/work/catkin_ws \
       ros:indigo \
       bash -c "export ROS_PACKAGE_PATH=/root/work/catkin_ws/src:/opt/ros/indigo/share:/opt/ros/indigo/stacks && \
                source /root/work/catkin_ws/devel/setup.bash && \
                roslaunch pr2_moveit_config pr2_moveit_planning_execution.launch"
