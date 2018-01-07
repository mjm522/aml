#!/bin/bash


sudo apt-get update
sudo apt-get -y install ros-indigo-rviz
sudo apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
                       ros-indigo-convex-decomposition ros-indigo-ivcon \
                       git-core python-argparse python-wstool python-vcstools python-rosdep ros-indigo-control-msgs \
                       ros-indigo-joystick-drivers ros-indigo-xacro ros-indigo-tf2-ros ros-indigo-rviz ros-indigo-cv-bridge \
                       ros-indigo-actionlib ros-indigo-actionlib-msgs ros-indigo-dynamic-reconfigure \
                       ros-indigo-trajectory-msgs ros-indigo-moveit \
                       ros-indigo-octomap-rviz-plugins \
                       gazebo2 ros-indigo-qt-build ros-indigo-gazebo-ros-pkgs ros-indigo-control-toolbox \
                       ros-indigo-realtime-tools ros-indigo-ros-controllers \
                       ros-indigo-tf-conversions ros-indigo-kdl-parser \
                       ros-indigo-ros-control ros-indigo-ros-controllers ros-indigo-gazebo-ros-control \
                       build-essential python-dev swig python-pygame && \
sudo rm -rf /var/lib/apt/lists/*

sudo pip install --upgrade pip 
sudo pip install protobuf

sudo apt-get install mesa-utils

sudo pip install --upgrade pip 

sudo pip install virtualenvwrapper

if [[ -z "${WORKON_HOME}" ]]; then
  echo "export WORKON_HOME=~/.virtualenvs" >> .bashrc
  echo ". /usr/local/bin/virtualenvwrapper.sh" >> .bashrc

  export WORKON_HOME=~/.virtualenvs
  . /usr/local/bin/virtualenvwrapper.sh
fi

bash

mkvirtualenv robotics_test
workon robotics_test

pip install git+git://github.com/pybox2d/pybox2d

pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide
pip install --upgrade tensorflow