#!/bin/bash

# Don't forget: xhost +
# Modify mounting location as you like

docker run -it \
       --env="DISPLAY=192.168.0.9:0" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --privileged \
       -v /Users/ermanoarruda/Projects/ros_ws/:/root/Projects/ros_ws/ \
       -w /root/Projects/ros_ws/ \
       aml_test:latest \
       bash -c "export KINECT1=true && \
                /bin/bash"
