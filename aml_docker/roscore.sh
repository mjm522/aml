#!/bin/bash

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/Projects/"

if [ -z "$DOCKER_IMAGE" ]
then
      echo "usage: ./roscore.sh <docker-image-tag>"
      echo "example: ./roscore.sh dev:indigo-cuda"
      echo "to list built docker images run: docker images"
      exit 1
fi

shopt -s expand_aliases
source $HOME/.bashrc
source ./aml_aliases.sh

# docker network create rosnet

xdocker run -h master -it --rm \
       --net rosnet --name master \
       --env ROS_HOSTNAME=master ${extra_params} \
       ${DOCKER_IMAGE} \
       roscore
