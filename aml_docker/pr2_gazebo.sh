#!/bin/bash

# Don't forget: xhost +
# bash roscore.sh

ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"

shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh

docker run -h gazebo -it --rm \
       --net rosnet --name gazebo\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=gazebo \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ${extra_params} \
       ros:indigo \
       bash -c "export KINECT1=true && \
                roslaunch pr2_gazebo pr2_empty_world.launch"

