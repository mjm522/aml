#!/bin/bash

ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"

if [ $# -ne 1 ]; then
	echo "usage: ./exec_container.sh <container_id>"
    echo "example: ./docker_build.sh fa11998579c"
    echo "to get current container id, run ./get_containerId.sh script"
  exit 1
fi

# CONTAINER_ID=$1

# if [ -z "$CONTAINER_ID" ]
# then
#       echo "usage: ./exec_container.sh <container_id>"
#       echo "example: ./docker_build.sh fa11998579c"
#       echo "to get current container id, run ./get_containerId.sh script"
#       exit 1
# fi


shopt -s expand_aliases
source $HOME/.bashrc
source ${ROOT_DIR}/aml_aliases.sh


echo 'Entering container:' $1
xdocker exec -it $1 bash
# -c "source /opt/ros/kinetic/setup.bash && /bin/bash"