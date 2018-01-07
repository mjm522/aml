#!/bin/bash

if [ $# -ne 1 ]; then
  echo 'please provide arg : CONTAINER ID'
  exit 1
fi

echo 'Entering container:' $1
docker exec -it $1 \
       bash -c "source /opt/ros/kinetic/setup.bash && /bin/bash"
