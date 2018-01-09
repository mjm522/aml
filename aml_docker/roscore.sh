#!/bin/sh

# docker network create rosnet

docker run -h master -it --rm \
       --net rosnet --name master \
       --env ROS_HOSTNAME=master \
       ros:indigo \
       roscore
