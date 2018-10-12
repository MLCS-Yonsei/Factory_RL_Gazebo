import subprocess
import time
import numpy as np
import math
import copy
import random
import commands
from itertools import combinations as it6ercomb
from env_settings import env_config

class env_reset(object):
    def __init__(self):
        self.gazebo_model_path = env_config.gazebo_model_path

        self.floor_list = env_config.floor_list
        self.wall_list = env_config.wall_list
        self.tool_list_large = env_config.tool_list_large
        self.tool_list_medium = env_config.tool_list_medium
        self.tool_list_small = env_config.tool_list_small

        self.floor_texture_num = env_config.floor_texture_num
        self.wall_texture_num = env_config.wall_texture_num
        self.tool_large_num = env_config.tool_large_num
        self.tool_medium_num = env_config.tool_medium_num
        self.tool_small_num = env_config.tool_small_num
        
        self.tool_large_num_desired = env_config.tool_large_num_desired
        self.tool_medium_num_desired = env_config.tool_medium_num_desired
        self.tool_small_num_desired = env_config.tool_small_num_desired
        self.coord_list_large = env_config.coord_list_large
        self.coord_list_medium = env_config.coord_list_medium
        self.coord_list_small = env_config.coord_list_small

        self.x_length = env_config.x_length
        self.y_length = env_config.y_length

        self.floor_index = math.floor(np.random.random(1)*(env_config.floor_texture_num))
        self.wall_index = math.floor(np.random.random(1)*(env_config.wall_texture_num))
    
    def rand_deploy(self):
        np.random.seed(int(math.floor(time.time())))
        chosen_large_tools = random.sample(self.tool_list_large, self.tool_large_num_desired)
        chosen_medium_tools = random.sample(self.tool_list_medium, self.tool_medium_num_desired)
        chosen_small_tools = random.sample(self.tool_list_small, self.tool_small_num_desired)

        chosen_small_tools_coord = random.sample(self.coord_list_small,self.tool_small_num_desired)
        chosen_floor = self.floor_list[int(np.random.choice(self.floor_texture_num,1))]
        chosen_wall = self.wall_list[int(np.random.choice(self.wall_texture_num,1))]
        
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: '+ str(chosen_floor) +'_floor, pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-20, -15), shell=True)

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: '+ str(chosen_wall) +'_wall, pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-20, -15), shell=True)
        
        for i in range(self.tool_large_num_desired):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(chosen_large_tools[i]) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(self.coord_list_large[i][0], self.coord_list_large[i][1]), shell=True)

        for i in range(self.tool_medium_num_desired):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(chosen_medium_tools[i]) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(self.coord_list_medium[i][0], self.coord_list_medium[i][1]), shell=True)

        for i in range(self.tool_small_num_desired):
            index_ = chosen_small_tools_coord[i]
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(chosen_small_tools[i]) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(index_[0], index_[1]), shell=True)

        empty_coord = list(set(self.coord_list_small)-set(chosen_small_tools_coord))

        return chosen_floor, chosen_wall, chosen_large_tools, chosen_medium_tools, chosen_small_tools, empty_coord

    def rand_move(self, chosen_floor, chosen_wall, chosen_large_tools, chosen_medium_tools, chosen_small_tools):
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(chosen_floor) +'_floor, pose: { position: { x: -50, y: -50 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(chosen_wall) +'_wall, pose: { position: { x: -50, y: -50 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)

        for tool_ in chosen_large_tools:
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(tool_) +', pose: { position: { x: -40, y: -40 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' , shell=True)

        for tool_ in chosen_medium_tools:
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(tool_) +', pose: { position: { x: -40, y: -40 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' , shell=True)


        for tool_ in chosen_small_tools:
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: ' + str(tool_) +', pose: { position: { x: -40, y: -40 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' , shell=True)
