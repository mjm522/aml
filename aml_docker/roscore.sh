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

# docker network create rosnet

docker run -h master -it --rm \
       --net rosnet --name master \
       --env ROS_HOSTNAME=master \
       ${DOCKER_IMAGE} \
       roscore
