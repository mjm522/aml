#!/bin/sh

# xhost +

docker run -h rqt_graph -it --rm \
       --net rosnet --name  rqt_graph\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=rqt_graph \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       ros:indigo \
       rosrun rqt_graph rqt_graph
