#!/bin/sh

if [ $# -ne 1 ]; then
  echo 'please provide arg : CONTAINER ID'
  exit 1
fi

echo 'Entering container:' $1
nvidia-docker exec -it $1 \
       bash -c "cd aml_ws && ./baxter.sh sim"
# -c "source /opt/ros/kinetic/setup.bash && /bin/bash"