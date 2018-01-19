/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew update
brew install cmake

brew tap ros/deps
brew tap osrf/simulation
brew tap homebrew/versions
brew tap homebrew/science

sudo -H pip install -U -I awsebcli

sudo -H pip install git+https://github.com/vcstools/wstool

sudo -H pip install -U -I  rosdep

sudo -H pip install rosinstall rosinstall_generator rospkg catkin-pkg Distribute sphinx






