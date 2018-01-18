#!/bin/bash

shopt -s expand_aliases
source $HOME/.bashrc

ROOT_DIR="$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)"
cd ${ROOT_DIR}

AML_PATH=$(cd ../ && pwd)


if [ -z "${AML_DIR}" ]
then
	  echo "Setting environment variable AML_DIR=${AML_PATH}"
	  echo ' ' >> ${HOME}/.bashrc
      echo "export AML_DIR=${AML_PATH}" >> ${HOME}/.bashrc
else
	  echo "AML environment variable already exists and set to AML_DIR=${AML_PATH}"
fi

