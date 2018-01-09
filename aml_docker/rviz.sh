#!/bin/sh

# xhost +

docker run -h rviz -it --rm \
       --net host --name rviz \
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=rviz \
       --env="DISPLAY=192.168.0.9:0" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       aml:latest \
       rosrun rviz rviz
