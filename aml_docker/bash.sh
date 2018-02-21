#!/bin/bash

# Description: starts container and opens an interactive bash shell using image tag passed as parameter
# Usage: ./bash.sh <docker-image-tag>
# Example: ./bash.sh dev:indigo-cuda

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/catkin_ws/"
ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"
echo "ROOT_DIR: ${ROOT_DIR}"

#192.168.0.9:0
#192.168.0.9:0

if [ -z "$DOCKER_IMAGE" ]
then
      echo "usage: ./bash.sh <docker-image-tag>"
      echo "example: ./bash.sh dev:indigo-cuda"
      echo "to list built docker images run: docker images"
      exit 1
fi

shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh

#sudo nvidia-modprobe -u -c=0
# Running container and giving access to X11 in a safer way
# --network=rosnet \
# --hostname="aml_container" \ (give the container a name)
#        --network="host" \
# --publish-all="true" \ (to publish all ports from the container to the outside)
# --device="/dev/snd" \
#       --network="host" \
#        --hostname="aml_container" \
#       --privileged \ (We need to use --privileged flag if we are in the same network as the host to render 3D graphical interfaces (gazebo,rviz, etc))
# When the operator executes docker run --privileged, Docker will enable access to all devices on the host as well as set some configuration in AppArmor or SELinux to allow the container nearly all the same access to the host as processes running outside containers on the host. 
#

xdocker run -it \
       --user=$(id -u) \
       --env="DISPLAY" \
       --network="host" \
       --hostname="aml_container" \
       --privileged \
       --env="QT_X11_NO_MITSHM=1" \
       --workdir="/home/$USER" \
       --volume="${ROOT_DIR}/avahi-configs:/etc/avahi" \
       --volume="/home/$USER:/home/$USER" \
       --volume="/etc/group:/etc/group:ro" \
       --volume="/etc/passwd:/etc/passwd:ro" \
       --volume="/etc/shadow:/etc/shadow:ro" \
       --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="${WORK_DIR}:/home/Projects" ${extra_params} \
       $DOCKER_IMAGE \
       bash 

# Unsafe container execution with X11 access 
# xhost +
# nvidia-docker run -it \
#        --env="DISPLAY" \
#        --env="QT_X11_NO_MITSHM=1" \
#        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#        --volume="${WORK_DIR}:/home/Projects" \
#        $DOCKER_IMAGE \
#        bash
# xhost - (don't ever forget this)

# Container execution with X11 access in OSX (requires osx_bridge_display.bash to be run beforehand)
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
#        $DOCKER_IMAGE \
#        bash



# --volume="path-in-my-computer:path-in-docker"