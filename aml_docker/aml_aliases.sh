#!/bin/bash

shopt -s expand_aliases
source $HOME/.profile

command_exists () {
    type "$1" &> /dev/null ;
}

if command_exists nvidia-docker && [ -z "$AML_DOCKER_NOFORCE_INTEL" ]; then
      extra_params=''
      alias xdocker="nvidia-docker"
      echo "nvidia-docker exists"
else
      alias xdocker="docker"
      extra_params=--device=/dev/dri:/dev/dri
      echo "nvidia-docker does not exist (falling back to docker)"
fi