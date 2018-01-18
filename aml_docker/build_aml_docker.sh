#!/bin/sh

WORK_DIR="/home/ermanoarruda/Projects/"

docker run -it \
	   --rm \
       --env="LIBGL_ALWAYS_SOFTWARE=1" \
       --env="LIBGL_ALWAYS_INDIRECT=1" \
       --env="DISPLAY=192.168.0.9:0" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="${WORK_DIR}:/home/Projects" \
       dev:aml
       pwd && cd /home/Projects/aml_ws/src/aml/aml_scripts && source setup_rospkg_deps.sh