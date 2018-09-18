# Factory_RL_Gazebo
Youbot Reinforcement Learning in Factory environment

This work uses **gym-gazebo**.
Visit [erlerobotics/gym-gazebo](https://github.com/erlerobot/gym-gazebo) for more information and videos.

## Dependency ROS package installation
One-line installation
```bash
cd
cd catkin_ws/src
git clone https://github.com/MLCS-Yonsei/mlcs_sim
cd mlcs_sim
sh setup.sh
cd ../..
catkin_make
```

## Main Installation
One-line installation
```bash
cd 
git clone https://github.com/MLCS-Yonsei/Factory_RL_Gazebo
cd Factory_RL_Gazebo
sh setup.sh
```

## Usage

### Running an environment

```bash
cd Factory_RL_Gazebo/runfile
python run_*.py
```