#!/bin/bash


DOCKER_FILE_PATH=$1

if [ -z "$DOCKER_FILE_PATH" ]
then
      echo "usage: ./docker_build.sh <path-to-docker-file>"
      echo "example: ./docker_build.sh indigo-cuda"
      exit 1
fi

#cp -r avahi-configs ${DOCKER_FILE_PATH}
docker build ${DOCKER_FILE_PATH} -t dev:${DOCKER_FILE_PATH##*/}