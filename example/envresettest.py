from env_reset import env_reset
import time

#check '$roscore & rosrun gazebo_ros gazebo'

#env_reset().gazebo_warmup()

for i in range(100):
    chosen_floor, chosen_wall, chosen_large_tools, chosen_medium_tools, chosen_small_tools = env_reset().rand_deploy()

    time.sleep(10)

    env_reset().rand_move(chosen_floor, chosen_wall, chosen_large_tools, chosen_medium_tools, chosen_small_tools)

    time.sleep(10)