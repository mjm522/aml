#!/bin/bash

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/Projects/"
ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"

if [ -z "$DOCKER_IMAGE" ]
then
      echo "usage: ./rqt_graph.sh <docker-image-tag>"
      echo "example: ./rqt_graph.sh dev:indigo-cuda"
      echo "to list built docker images run: docker images"
      exit 1
fi

shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh

# xhost +

xdocker run -h rqt_graph -it --rm \
       --net rosnet --name  rqt_graph\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=rqt_graph \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ${extra_params} \
       ${DOCKER_IMAGE} \
       rosrun rqt_graph rqt_graph
