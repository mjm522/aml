#!/bin/bash

DOCKER_IMAGE=$1
WORK_DIR="${HOME}/Projects/"

if [ -z "$DOCKER_IMAGE" ]
then
      echo "usage: ./rqt_graph.sh <docker-image-tag>"
      echo "example: ./rqt_graph.sh dev:indigo-cuda"
      echo "to list built docker images run: docker images"
      exit 1
fi

# xhost +

docker run -h rqt_graph -it --rm \
       --net rosnet --name  rqt_graph\
       --env ROS_MASTER_URI=http://master:11311 \
       --env ROS_HOSTNAME=rqt_graph \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       ${DOCKER_IMAGE} \
       rosrun rqt_graph rqt_graph
