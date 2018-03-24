#!/bin/sh

##################################################################################
######### What this script does ##################################################
##################################################################################
# 0) This scripts assumes: fresh install of Ubuntu 16.04
# 1) installs dependencies for baxter & sawyer gazebo simulators 
# 2) creates a python virtual env named 'robotics'
# 3) installs all python2 dependences in the 'robotics' virtual env
### 3.1) This includes tensorflow (the non-gpu version)


sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
sudo apt-get update
sudo apt-get install ros-kinetic-desktop-full
sudo rosdep init
rosdep update
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo apt-get install python-rosinstall python-rosinstall-generator python-wstool build-essential

# dependencies for baxter and sawyer gazebo simulators
sudo apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
                       ros-kinetic-convex-decomposition ros-kinetic-ivcon \
                       git-core python-argparse python-wstool python-vcstools python-rosdep ros-kinetic-control-msgs \
                       ros-kinetic-joystick-drivers ros-kinetic-xacro ros-kinetic-tf2-ros ros-kinetic-rviz ros-kinetic-cv-bridge \
                       ros-kinetic-actionlib ros-kinetic-actionlib-msgs ros-kinetic-dynamic-reconfigure \
                       ros-kinetic-trajectory-msgs ros-kinetic-moveit \
                       ros-kinetic-octomap-rviz-plugins \
                       gazebo7 ros-kinetic-qt-build ros-kinetic-gazebo-ros-pkgs ros-kinetic-control-toolbox \
                       ros-kinetic-realtime-tools ros-kinetic-ros-controllers \
                       ros-kinetic-tf-conversions ros-kinetic-kdl-parser ros-kinetic-sns-ik-lib \
                       ros-kinetic-ros-control ros-kinetic-ros-controllers ros-kinetic-gazebo-ros-control \
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

pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide pandas
pip install pybullet==1.8.3
pip install --upgrade tensorflow