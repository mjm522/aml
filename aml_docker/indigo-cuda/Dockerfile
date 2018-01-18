FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu14.04

# Requirement:
# install nvidia-docker,
# container must be started with nvidia-docker

# build setups follows indigo-ros-core -> indigo-ros-base -> indigo-desktop-full
# only difference is not based on ubuntu:trusty

ENV DEBIAN_FRONTEND noninteractive

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg2 \
&& rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list


# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && sudo rosdep update

# install ros core packages
ENV ROS_DISTRO indigo
RUN apt-get update && apt-get install -y \
    ros-indigo-ros-core=1.1.5-0* \
    && rm -rf /var/lib/apt/lists/*


# install ros base packages
 RUN apt-get update && apt-get install -y \
    ros-indigo-ros-base=1.1.5-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros desktop full packages
RUN apt-get update && apt-get install -y \
    ros-indigo-desktop-full=1.1.5-0* \
    && rm -rf /var/lib/apt/lists/*





# setup entrypoint, need entrypoint.sh in the same folder with Dockerfile
COPY ./ros_entrypoint.sh /

# Dependencies for baxter and sawyer simulator
RUN apt-get update && \ 
    apt-get -y install python-pip python-scipy libprotobuf-dev protobuf-compiler libboost-all-dev \
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

RUN pip install --upgrade pip && \
    pip install protobuf

#RUN bin/bash -c "cd /tmp && \
#                 git clone https://github.com/pybox2d/pybox2d && \
#                 cd pybox2d && \
#                 python setup.py build && \
#                 python setup.py install"
#RUN rm -rf /tmp/pybox2d*


#six PySide
RUN pip install --upgrade pip && \
    pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy && \
    pip install --upgrade \ 
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp27-none-linux_x86_64.whl
# TENSOR FLOW *GPU*

# Fix paths so that CUDNN can be found
# See https://github.com/tensorflow/tensorflow/issues/8264
#RUN mkdir /usr/lib/x86_64-linux-gnu/include/ && \
#  ln -s /usr/lib/x86_64-linux-gnu/include/cudnn.h /usr/lib/x86_64-linux-gnu/include/cudnn.h && \
#  ln -s /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h && \
#  ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
#  ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.6 /usr/local/cuda/lib64/libcudnn.so.6



ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]