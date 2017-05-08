# Active Manipulation Learning (AML) 


## Dependencies

* [ROS (Indigo)](http://wiki.ros.org/indigo/Installation/Ubuntu)
* [BaxterSDK](http://sdk.rethinkrobotics.com/wiki/Hello_Baxter)
* [Baxter Simulator](http://sdk.rethinkrobotics.com/wiki/Simulator_Installation)

### Python Libraries

* [numpy](http://www.numpy.org/)
* [numpy-quaternion](https://pypi.python.org/pypi/numpy-quaternion)
* [pygame](http://www.pygame.org/download.shtml)
* [PySide](https://pypi.python.org/pypi/PySide/1.2.4)
* [pybox2d](https://github.com/pybox2d/pybox2d)
* [Pillow](https://pypi.python.org/pypi/Pillow/4.1.1)
* [Scipy](https://pypi.python.org/pypi/scipy/0.19.0)
* [six](https://pypi.python.org/pypi/six/1.10.0)
* [decorator](https://pypi.python.org/pypi/decorator/4.0.11)
* [matplotlib](https://pypi.python.org/pypi/matplotlib/2.0.1)
* [ipython](https://pypi.python.org/pypi/ipython/6.0.0)
* [cv2](https://github.com/opencv/opencv)

#### This document lists various setup instructions after a fresh installation of Ubuntu 14.04 on your machine. The end part of the document also contains some of the possible errors during installation and their solutions.

1. Installing ROS - Indigo: Follow instructions on this [page](http://wiki.ros.org/indigo/Installation/Ubuntu).
**Important Note:** install  “desktop-full” version

2. Installing virtual environment

	* Install packages 
	```
	$sudo apt-get install python-setuptools
	$sudo easy_install pip
	$sudo apt-get install python-pip
	$sudo pip install virtualenv
	$sudo pip install virtualenvwrapper
	```

	* Add the following two lines in your ~/.bashrc script:
	```
	export WORKON_HOME=~/.venvs  
	source /usr/share/virtualenvwrapper/virtualenvwrapper.sh  
	export PIP_VIRTUALENV_BASE=~/.venvs
	```
	* Close the bashrc file and source them:
	```
	$source ~/.bashrc
	$mkvirtualenv robotics
	$workon robotics
	```
     
3. Install CUDA 8.0 and Cuda-NN 5.1
    
    You can get the compiled binary files from following websites:
    
    a. [CUDA](https://developer.nvidia.com/cuda-downloads)

    b. [CUDANN](https://developer.nvidia.com/cudnn)

    * You will have to create an account in Nvdia to download the libraries
    * Goto "Download cuDNN v5.1 (Jan 20, 2017), for CUDA 8.0", Dowload cuDNN v5.1 Library for Linux
    * From the terminal goto the download directory
    	
    	```
    	$tar -zxf cudnn-8.0-linux-x64-v5.1.tgz
    	$cd cuda/
    	$sudo cp include/* /usr/local/cuda/include/
    	$sudo cp lib64/* /usr/local/cuda/lib64/
    	```

    * Add the following bit to ~/.bashrc file

    	```
    	export CUDA_HOME=/usr/local/cuda-8.0 
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 
		PATH=${CUDA_HOME}/bin:${PATH} 
		export PATH
    	```


4. Install tensorflow in virtual environment 
	```
	$workon robotics
	$pip install --upgrade tensorflow-gpu
	```


5. Create a new workspace and clone aml repository in it
	```
	$mkdir ~/catkin_ws
	$cd catkin_ws
	$mkdir baxter_ws
	$cd baxter_ws
	$mkdir src
	$cd src
	$git clone https://github.com/RobotsLab/AML.git
	```

6. Baxter simulator setup

 	* Dependencies
	```
	$sudo apt-get install gazebo2 ros-indigo-qt-build ros-indigo-driver-common ros-indigo-gazebo-ros-control ros-indigo-gazebo-ros-pkgs ros-indigo-ros-control ros-indigo-control-toolbox ros-indigo-realtime-tools ros-indigo-ros-controllers ros-indigo-xacro python-wstool ros-indigo-tf-conversions ros-indigo-kdl-parser
	```

	* Simulator (specific versions)
	```
	$cd ~/catkin_workspaces/baxter_ws/src
	$wstool init .
	$wstool merge baxter_simulator_with_aml.rosinstall (take this file from aml/3rdparty)
	baxter_simulator.rosinstall
	$wstool update
	```

   * Check if any other unmet dependencies, run this line from ws folder 
	```
	$rosdep install --from-path . --ignore-src --rosdistro indigo -y -r
	```

	* Final step
	```
	$cd ../
	$catkin_make
	```
       
7. Few other dependencies
           
	```
	$pip install numpy numpy-quaternion pygame decorator ipython jupyter matplotlib Pillow scipy six PySide
	```

8. Installing pybox2d - follow instructions in this [page](https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md).

9. Installing opencv for python

	**Note:** this compilation could take a while! And install this only after removing opencv-python (this is unofficial version) if installed previously.
	
	* Go to a folder of your choice
	```
	$git clone https://github.com/Itseez/opencv.git
	$cd opencv
	$git checkout 3.2.0
	$mkdir build
	$cd build
	$cmake ..                             
	$sudo make                   
	$sudo make install         
	$sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'  
	$sudo ldconfig
	```

	* Set symlink to virtual environment (on assumtion that your venv name is "robotics")
	```
	$cd ~/.venvs/robotics/lib/python2.7/site-packages/
	$ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
	```

	* Check installation
	```
	$workon robotics
	$python
	>>import cv2
	```

	* To compile samples
	```
	$cd <opencv folder path>/opencv/samples
	$cmake .               
	$sudo make 
	```

        
#### Possible Errors:

1. Could not find any downloads that satisfy the requirement tensorflow
   
   **Solution:**  ```$pip install --upgrade pip```

2. No module named catkin_pkg.package

   **Solution:** ```$pip install catkin_pkg```

3. No module named rospkg

   **Solution:** ```$pip install -U rospkg```

4. ImportError: No module named 'em'

   **Solution:** ```$pip install empty```

5. Could not stop controller 'left_joint_velocity_controller' since it is not running
  
	  **Solution:** goto ```$~/catkin_ws/baxter_ws/src/ baxter_gazebo/src/baxter_gazebo_ros_control_plugin.cpp```

	**Edit lines:**```::SwitchController::Request::STRICT to ::SwitchController::Request::BEST_EFFORT ```
	 (This happens in two places)

	*Note:* You have to rebuild the catkin_make from $baxter_ws



