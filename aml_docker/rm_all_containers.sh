#!/bin/bash

shopt -s expand_aliases
source $HOME/.bashrc
source ./aml_aliases.sh

xdocker rm `docker ps -a -q`
