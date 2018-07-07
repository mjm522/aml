#!/bin/bash


DOCKER_FILE_PATH=$1

if [ -z "$DOCKER_FILE_PATH" ]
then
      echo "usage: ./docker_build.sh <path-to-docker-file>"
      echo "example: ./docker_build.sh indigo-cuda"
      exit 1
fi

# if [[ -z ${http_proxy} ]]; then
#   HTTP_PROXY="127.0.0.1"
# else
#   HTTP_PROXY=${http_proxy}
# fi

# if [[ -z ${https_proxy} ]]; then
#   HTTPS_PROXY="127.0.0.1"
# else
#   HTTPS_PROXY=${https_proxy}
# fi

#cp -r avahi-configs ${DOCKER_FILE_PATH}
docker build ${DOCKER_FILE_PATH} -t dev:${DOCKER_FILE_PATH##*/} 
#--build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTPS_PROXY