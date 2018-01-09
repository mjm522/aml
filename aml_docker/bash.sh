#!/bin/bash

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/Projects/"

#192.168.0.9:0
#192.168.0.9:0



# Running container and giving access to X11 in a safer way
nvidia-docker run -it \
       --user=$(id -u) \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --workdir="/home/$USER" \
       --volume="/home/$USER:/home/$USER" \
       --volume="/etc/group:/etc/group:ro" \
       --volume="/etc/passwd:/etc/passwd:ro" \
       --volume="/etc/shadow:/etc/shadow:ro" \
       --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="${WORK_DIR}:/home/Projects" \
       $DOCKER_IMAGE
       bash

# Unsafe container execution with X11 access 
# xhost +
# nvidia-docker run -it \
#        --env="DISPLAY" \
#        --env="QT_X11_NO_MITSHM=1" \
#        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#        --volume="${WORK_DIR}:/home/Projects" \
#        $DOCKER_IMAGE
#        bash
# xhost - (don't ever forget this)

# Container execution with X11 access in OSX (requires bridge_display.bash to be run beforehand)
# This does not support 3D rendering, i.e. we cannot run rviz (not hardware-accelerated)
# Display host ip
# DISPLAY_IP=192.168.0.9
# docker run -it \
#        --env="LIBGL_ALWAYS_SOFTWARE=1" \
#        --env="LIBGL_ALWAYS_INDIRECT=1" \
#        --env="DISPLAY=${DISPLAY_IP}:0" \
#        --env="QT_X11_NO_MITSHM=1" \
#        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#        --volume="${WORK_DIR}:/home/Projects" \
#        $DOCKER_IMAGE
#        bash



# --volume="path-in-my-computer:path-in-docker"
