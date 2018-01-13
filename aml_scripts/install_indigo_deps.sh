#!/bin/sh

##################################################################################
######### What this script does ##################################################
##################################################################################
# 0) This scripts assumes: fresh install of Ubuntu 14.04 
# 1) installs dependencies for baxter & sawyer gazebo simulators 
# 2) creates a python virtual env named 'robotics'
# 3) installs all python2 dependences in the 'robotics' virtual env
### 3.1) This includes tensorflow (the non-gpu version)

sudo apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
sudo sh -c "echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list"

# installing ros-indigo-desktop-full
sudo apt-get update 
sudo apt-get install ros-indigo-desktop-full
sudo rosdep init
rosdep update
sudo apt-get install python-rosinstall
echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc
source ~/.bashrc

# dependencies for baxter and sawyer gazebo simulators
sudo apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
                       ros-indigo-convex-decomposition ros-indigo-ivcon \
                       git-core python-argparse python-wstool python-vcstools python-rosdep ros-indigo-control-msgs \
                       ros-indigo-joystick-drivers ros-indigo-xacro ros-indigo-tf2-ros ros-indigo-rviz ros-indigo-cv-bridge \
                       ros-indigo-actionlib ros-indigo-actionlib-msgs ros-indigo-dynamic-reconfigure \
                       ros-indigo-trajectory-msgs ros-indigo-moveit \
                       ros-indigo-octomap-rviz-plugins \
                       gazebo2 ros-indigo-qt-build ros-indigo-driver-common ros-indigo-gazebo-ros-pkgs ros-indigo-control-toolbox \
                       ros-indigo-realtime-tools ros-indigo-ros-controllers \
                       ros-indigo-tf-conversions ros-indigo-kdl-parser \
                       ros-indigo-ros-control ros-indigo-ros-controllers ros-indigo-gazebo-ros-control \
                       build-essential python-dev swig python-pygame && \
sudo rm -rf /var/lib/apt/lists/*

sudo pip install --upgrade pip 
sudo pip install protobuf

sudo pip install --upgrade pip 

sudo pip install virtualenvwrapper

if [[ -z "${WORKON_HOME}" ]]; then
  echo "export WORKON_HOME=~/.virtualenvs" >> .bashrc
  echo ". /usr/local/bin/virtualenvwrapper.sh" >> .bashrc

  export WORKON_HOME=~/.virtualenvs
  . /usr/local/bin/virtualenvwrapper.sh
fi

bash

mkvirtualenv robotics
workon robotics

pip install git+git://github.com/pybox2d/pybox2d

pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide
pip install --upgrade tensorflow