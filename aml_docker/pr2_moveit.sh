#!/bin/bash

# Don't forget: xhost +
# Modify mounting location as you like
# bash roscore.sh

docker run -h moveit -it --rm \
       --net rosnet --name moveit\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=moveit \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       -w /root/work/catkin_ws \
       -v /home/ahayashi/work:/root/work \
       ros:indigo \
       bash -c "export ROS_PACKAGE_PATH=/root/work/catkin_ws/src:/opt/ros/indigo/share:/opt/ros/indigo/stacks && \
                source /root/work/catkin_ws/devel/setup.bash && \
                roslaunch pr2_moveit_config demo.launch"
