# Factory_RL_Gazebo
Youbot Reinforcement Learning in Factory environment (SLAM &amp; Navi)

This work uses Gym-gazebo.
Visit [erlerobotics/gym](http://erlerobotics.com/docs/Simulation/Gym/) for more information and videos.

## Installation
One-line install script available.
```bash
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
echo "alias k='killall -9 gzserver gzclient roslaunch rosmaster rviz'" >> ~/.bashrc
```
