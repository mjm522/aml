FROM osrf/ros:kinetic-desktop-full

RUN apt-get update && \ 
    apt-get -y install ros-kinetic-rviz && \
    apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
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
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install protobuf

RUN apt-get update && apt-get install mesa-utils

RUN bin/bash -c "cd /tmp && \
                 git clone https://github.com/pybox2d/pybox2d && \
                 cd pybox2d && \
                 python setup.py build && \
                 python setup.py install"
RUN rm -rf /tmp/pybox2d*

RUN pip install --upgrade pip && \
    pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide && \
    pip install --upgrade tensorflow



##########################################################################
### Recommend: execute gazebo one time and save container for speeding up
# xxx@host$ bash bash.sh
# xxx@docker$ gazebo
#
### Stop and exit from docker by ctrl + d
### Check latest CONTAINER ID
# xxx@host$ docker ps -a
# xxx@host$ docker commit CONTAINER ID ros:gazebo
#
### Check images
# xxx@host$ docker images
### modify scripts from ros:kinetic --> ros:gazebo
##########################################################################