import subprocess
import time
import numpy as np
import math
import copy
from random import shuffle
import commands
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
        tools=[]
        for tool,tool_num in self.tool_list:
            tools+=[tool+str(i+1) for i in range(tool_num)]
        
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: floor'+ '%d' %(self.floor_index+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-15, -15), shell=True)

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: wall'+ '%d' %(self.wall_index+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-15, -15), shell=True)
        
        for i,tool in enumerate(tools):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: %s' %(tool) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(self.rand_coord_list[i][0], self.rand_coord_list[i][1]), shell=True)

        # print('-'*50 +'\n Randomized environment model set done.')

        return self.rand_coord_list[len(tools)], self.floor_index+1, self.wall_index+1

    def gazebo_warmup(self):
        for tool_ in self.tool_list_large:
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_machine/' + tool_ +'/model.sdf -sdf -model '+ tool_ +' -y -40 -x -40', shell=True)
        
        for tool_ in self.tool_list_medium:
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_machine/' + tool_ +'/model.sdf -sdf -model '+tool_+' -y -40 -x -40', shell=True)
        
        for tool_ in self.tool_list_small:
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_machine/' + tool_ +'/model.sdf -sdf -model '+ tool_ +' -y -40 -x -40', shell=True)

        for floor_ in self.floor_list:
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_floor/' + floor_ +'_floor/model.sdf -sdf -model '+ floor_ +'_floor -y -50 -x -50', shell=True)

        for wall_ in self.wall_list:
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_wall/' + wall_ +'_wall/model.sdf -sdf -model '+ wall_ +'_wall -y -50 -x -50', shell=True)

        time.sleep(5)

    def rand_move(self, floor_index, wall_index):
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: floor'+ '%d' %(floor_index) +', pose: { position: { x: -50, y: -50 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: wall'+ '%d' %(wall_index) +', pose: { position: { x: -50, y: -50 ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)