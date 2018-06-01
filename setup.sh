#Gazebo8#
sudo apt-get update
sudo apt-get upgrade
sudo apt-get remove .*gazebo.* -y
sudo apt-get update -y
sudo apt-get install libignition-math2 -y
sudo apt-get install libgazebo7 -y
sudo apt-get install gazebo7 -y
sudo apt-get install libgazebo7-dev -y
sudo apt-get install ros-kinetic-gazebo-*
sudo apt-get install libignition-math3 -y
sudo apt-get install libgazebo8 -y
sudo apt-get install gazebo8 -y
sudo apt-get install ros-kinetic-gazebo8-*
sudo apt-get install ros-kinetic-gazebo-ros


#MLCS_sim#
sudo apt-get install ros-kinetic-joint-state-controller -y
sudo apt-get install ros-kinetic-joint-state-publisher -y
sudo apt-get install ros-kinetic-robot-state-publisher -y
sudo apt-get install ros-kinetic-controller-interface -y
sudo apt-get install ros-kinetic-roslint
sudo apt-get install ros-kinetic-control-toolbox -y
sudo apt-get install ros-kinetic-twist-mux -y
sudo apt-get install ros-kinetic-pr2-description -y
sudo apt-get install ros-kinetic-cob-scan-unifier -y
sudo apt-get install ros-kinetic-gmapping -y
sudo apt-get install ros-kinetic-kdl-parser -y

#Gym gazebo
sudo apt-get install cmake gcc g++ qt4-qmake libqt4-dev libusb-dev libftdi-dev -y
sudo apt-get install ros-kinetic-octomap-msgs -y
sudo apt-get install ros-kinetic-joy -y
sudo apt-get install ros-kinetic-geodesy -y
sudo apt-get install ros-kinetic-octomap-ros -y
sudo apt-get install ros-kinetic-pluginlib -y
sudo apt-get install ros-kinetic-trajectory-msgs -y
sudo apt-get install ros-kinetic-control-msgs -y
sudo apt-get install ros-kinetic-std-srvs -y
sudo apt-get install ros-kinetic-nodelet -y
sudo apt-get install ros-kinetic-urdf -y
sudo apt-get install ros-kinetic-rviz -y
sudo apt-get install ros-kinetic-kdl-conversions -y
sudo apt-get install ros-kinetic-eigen-conversions -y
sudo apt-get install ros-kinetic-tf2-sensor-msgs -y
sudo apt-get install ros-kinetic-pcl-ros -y
sudo apt-get install ros-kinetic-navigation -y

pip install gym
pip install rospkg catkin_pkg


#Installation
pip install -e .
cd installation
if [ -z "$ROS_DISTRO" ]; then
  echo "ROS not installed. Check the installation steps: https://github.com/erlerobot/gym#installing-the-gazebo-environment"
fi

program="gazebo"
condition=$(which $program 2>/dev/null | grep -v "not found" | wc -l)
if [ $condition -eq 0 ] ; then
    echo "Gazebo is not installed. Check the installation steps: https://github.com/erlerobot/gym#installing-the-gazebo-environment"
fi

source /opt/ros/kinetic/setup.bash

ws="catkin_ws"
if [ -d $ws ]; then
  echo "Error: catkin_ws directory already exists" 1>&2
fi
src=$ws"/src"
mkdir -p $src
cd $src
catkin_init_workspace
git clone https://github.com/MLCS-Yonsei/mlcs_sim.git
rm -rf mlcs_sim/.git
cd ..
catkin_make
source devel/setup.bash
catkin_make -j 1
bash -c 'echo source `pwd`/devel/setup.bash >> ~/.bashrc'
echo "## ROS workspace compiled ##"



