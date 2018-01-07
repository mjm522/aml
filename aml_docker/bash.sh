#!/bin/sh

# Don't forget: xhost +
#xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=1
#192.168.0.9:0
docker run -it \
       -h docker \
       --env="LIBGL_ALWAYS_SOFTWARE=1" \
       --env="LIBGL_ALWAYS_INDIRECT=1" \
       --env="DISPLAY=192.168.0.9:0" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="/Users/ermanoarruda/Projects/:/home/Projects" \
       dev:aml
       bash


# --volume="path-in-my-computer:path-in-docker"
