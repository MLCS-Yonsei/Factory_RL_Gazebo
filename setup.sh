#ROS#
#sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
#sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
#sudo apt-get update  -y
#sudo apt-get install ros-kinetic-desktop-full  -y
#sudo rosdep init
#sudo rosdep fix-permissions
#rosdep update
#echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
#echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
#source ~/.bashrc
#sudo apt-get install python-rosinstall -y
#printenv | grep ROS
#echo "export ROS_MASTER_URI=http://IP_OF_REMOTE_PC:11311" >> ~/.bashrc
#echo "export ROS_HOSTNAME=IP_OF_REMOTE_PC" >> ~/.bashrc

#Gazebo8#
#sudo apt-get update -y
#sudo apt-get upgrade -y
#sudo apt-get remove '.*gazebo.*' '.*sdformat.*' '.*ignition-math.*' '.*ignition-msgs.*' '.*ignition-transport.*' -y
#sudo apt-get install ros-kinetic-desktop-full -y
#sudo apt-get install libignition-math3 -y
#sudo apt-get install libignition-math3-dev -y
#sudo apt-get install libsdformat5 -y
#sudo apt-get install libsdformat5-dev -y
#sudo apt-get install libgazebo8 -y
#sudo apt-get install libgazebo8-dev -y
#sudo apt-get install gazebo8 -y
#sudo apt-get install gazebo8-plugin-base -y
#sudo apt-get install ros-kinetic-gazebo8-ros -y
#sudo apt-get install ros-kinetic-gazebo8-ros-control -y
#sudo apt-get install ros-kinetic-gazebo8-plugins -y

#MLCS_sim#
#sudo apt-get install ros-kinetic-joint-state-controller -y
#sudo apt-get install ros-kinetic-joint-state-publisher -y
#sudo apt-get install ros-kinetic-robot-state-publisher -y
#sudo apt-get install ros-kinetic-controller-manager -y
#sudo apt-get install ros-kinetic-controller-interface -y
#sudo apt-get install ros-kinetic-roslint -y
#sudo apt-get install ros-kinetic-control-toolbox -y
#sudo apt-get install ros-kinetic-twist-mux -y
#sudo apt-get install ros-kinetic-pr2-description -y
#sudo apt-get install ros-kinetic-cob-scan-unifier -y
#sudo apt-get install ros-kinetic-gmapping -y
#sudo apt-get install ros-kinetic-kdl-parser -y

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


#Alias#
#echo "alias eb='gedit ~/.bashrc'" >> ~/.bashrc
#echo "alias sb='source ~/.bashrc'" >> ~/.bashrc
#echo "alias yb='roslaunch bringup robot.launch'" >> ~/.bashrc
#echo "alias key='roslaunch teleop teleop_keyboard.launch'" >> ~/.bashrc
#echo "alias kg='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz'" >> ~/.bashrc

#Installation
pip install -e .
cd installation
if [ -z "$ROS_DISTRO" ]; then
  echo "ROS not installed."
fi

program="gazebo"
condition=$(which $program 2>/dev/null | grep -v "not found" | wc -l)
if [ $condition -eq 0 ] ; then
    echo "Gazebo is not installed."
fi

source /opt/ros/kinetic/setup.bash

if [ -z "$GYM_GAZEBO_WORLD" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD="`pwd`/../world/test.world >> ~/.bashrc'
  exec bash
fi
if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH="`pwd`/../models >> ~/.bashrc'
  exec bash
fi
source ~/.bashrc
echo "## Installation complete!! ##"



