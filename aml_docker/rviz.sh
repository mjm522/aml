#!/bin/bash

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/Projects/"
ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"

if [ -z "$DOCKER_IMAGE" ]
then
      echo "usage: ./rviz.sh <docker-image-tag>"
      echo "example: ./rviz.sh dev:indigo-cuda"
      echo "to list built docker images run: docker images"
      exit 1
fi

shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh

# xhost +

xdocker run -h rviz -it --rm \
       --net host --name rviz \
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=rviz \
       --env="DISPLAY=192.168.0.9:0" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ${extra_params} \
       ${DOCKER_IMAGE} \
       rosrun rviz rviz
