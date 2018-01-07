#!/bin/bash

sudo apt-get update
sudo apt-get -y install ros-kinetic-rviz
sudo apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
                       ros-kinetic-convex-decomposition ros-kinetic-ivcon \
                       git-core python-argparse python-wstool python-vcstools python-rosdep ros-kinetic-control-msgs \
                       ros-kinetic-joystick-drivers ros-kinetic-xacro ros-kinetic-tf2-ros ros-kinetic-rviz ros-kinetic-cv-bridge \
                       ros-kinetic-actionlib ros-kinetic-actionlib-msgs ros-kinetic-dynamic-reconfigure \
                       ros-kinetic-trajectory-msgs ros-kinetic-moveit \
                       ros-kinetic-octomap-rviz-plugins \
                       gazebo7 ros-kinetic-qt-build ros-kinetic-gazebo-ros-pkgs ros-kinetic-control-toolbox \
                       ros-kinetic-realtime-tools ros-kinetic-ros-controllers \
                       ros-kinetic-tf-conversions ros-kinetic-kdl-parser \
                       ros-kinetic-ros-control ros-kinetic-ros-controllers ros-kinetic-gazebo-ros-control \
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