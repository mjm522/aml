FROM osrf/ros:indigo-desktop-full



# Dependencies for baxter and sawyer simulator
RUN apt-get update && \ 
    apt-get -y install sudo python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
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
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip==9.0.3 && \
    pip install protobuf && \
    pip install pandas

RUN \
  apt-get update && \
  apt-get -y install libgl1-mesa-glx libgl1-mesa-dri mesa-utils && \
  rm -rf /var/lib/apt/lists/*

RUN bin/bash -c "cd /tmp && \
                 git clone https://github.com/pybox2d/pybox2d && \
                 cd pybox2d && \
                 python setup.py build && \
                 python setup.py install"
RUN rm -rf /tmp/pybox2d*

RUN pip install --upgrade pip==9.0.3 && \
    pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide && \
    pip install --upgrade \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp27-none-linux_x86_64.whl
# TENSOR FLOW *CPU*

# Install pybullet
RUN pip install pybullet==1.8.3

# RUN mkdir ~/Projects



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
### modify scripts from ros:indigo --> ros:gazebo
##########################################################################