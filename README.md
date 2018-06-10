# Factory_RL_Gazebo
Youbot Reinforcement Learning in Factory environment (SLAM &amp; Navi)

This work uses **gym-gazebo**.
Visit [erlerobotics/gym-gazebo](https://github.com/erlerobot/gym-gazebo) for more information and videos.

## Dependency package
```bash
cd catkin_ws/src
git clone https://github.com/MLCS-Yonsei/mlcs_sim
cd ..
catkin_make
```

## Installation
One-line install script available.
```bash
cd
git clone https://github.com/MLCS-Yonsei/Factory_RL_Gazebo
cd Factory_RL_Gazebo
sh setup.sh
```

## Usage

### Running an environment

```bash
cd example
python main.py
```

### Killing background processes

Sometimes, after ending or killing the simulation `gzserver` and `rosmaster` stay on the background, make sure you end them before starting new tests.

We recommend creating an alias to kill those processes.

```bash
echo "alias killgazebo='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient rviz'" >> ~/.bashrc
```

### Lidar plugin
If your environment doesn't support GPU calculation, you should chage lidar module in `mlcs_sim/description/urdf/hokuyo_urg04_laser.gazebo.xacro`.

```bash
`libgazebo_ros_gpu_laser.so`
to
`libgazebo_ros_laser.so`
```
