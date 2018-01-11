#!/bin/bash

if [ $# -ne 1 ]; then
  echo 'please provide arg : CONTAINER ID'
  exit 1
fi

shopt -s expand_aliases
source $HOME/.bashrc
source ./aml_aliases.sh


echo 'Entering container:' $1
xdocker exec -it $1 \
	   ${extra_params}
       bash -c "cd aml_ws && ./baxter.sh sim"
# -c "source /opt/ros/kinetic/setup.bash && /bin/bash"