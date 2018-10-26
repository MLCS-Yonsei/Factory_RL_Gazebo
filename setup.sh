#ROS#
# sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
# sudo apt-get update  -y
# sudo apt-get install ros-kinetic-desktop-full  -y
# sudo rosdep init
# sudo rosdep fix-permissions
# rosdep update
# echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
# sudo apt-get install python-rosinstall -y
# printenv | grep ROS
# echo "export ROS_MASTER_URI=http://localhost" >> ~/.bashrc
# echo "export ROS_HOSTNAME=localhost" >> ~/.bashrc
# source ~/.bashrc

#Catkin_ws
# mkdir -p ~/catkin_ws/src
# cd ~/catkin_ws/src
# catkin_init_workspace
# echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
# source ~/.bashrc

#Gazebo8#
# sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
# cat /etc/apt/sources.list.d/gazebo-stable.list
# wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
# sudo apt-get update -y
# sudo apt-get upgrade -y
# sudo apt-get install libignition-math3 -y
# sudo apt-get install libgazebo8 -y
# sudo apt-get install gazebo8 -y
# sudo apt-get install ros-kinetic-gazebo8-*

#Alias#
# echo "alias eb='gedit ~/.bashrc'" >>~/.bashrc
# echo "alias sb='source ~/.bashrc && source ~/catkin_ws/devel/setup.bash'" >>~/.bashrc
# echo "alias cm='cd ~/catkin_ws && catkin_make'" >>~/.bashrc
# echo "alias usb='ls -l /dev |grep ttyUSB'" >>~/.bashrc
# echo "alias gpu='nvidia-smi -l 1'" >>~/.bashrc
# echo "alias kg='killall -9 roscore roslaunch rosmaster gzserver nodelet gzclient rviz'" >>~/.bashrc
# echo "alias fd='sudo find / -name'" >>~/.bashrc


#Gym gazebo_dependency pkg#
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
sudo apt install python-pip -y

pip install gym
pip install rospkg catkin_pkg

#Installation
pip install -e .

#Gazebo mopdel copy
cd models
sudo cp -r 'env_floor' 'env_machine' 'env_tool' 'env_wall' '/usr/share/gazebo-8/models'
cd Actor
sudo cp 'walk.dae' 'walk1.dae' 'walk2.dae' 'walk3.dae' '/usr/share/gazebo-8/media/models'
