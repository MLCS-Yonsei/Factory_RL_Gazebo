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
        self.tool_list = env_config.tool_list

        self.floor_texture_num = env_config.floor_texture_num
        self.wall_texture_num = env_config.wall_texture_num
        self.tool_num = env_config.tool_num
        self.lathe_num = env_config.lathe_num
        self.systec_num = env_config.systec_num

        self.x_length = env_config.x_length
        self.y_length = env_config.y_length

        self.floor_index = math.floor(np.random.random(1)*(env_config.floor_texture_num))
        self.wall_index = math.floor(np.random.random(1)*(env_config.wall_texture_num))

        self.rand_coord_list = copy.copy(env_config.coord_list)
        shuffle(self.rand_coord_list)
    
    def rand_deploy(self):
        np.random.seed(int(math.floor(time.time())))
        tools=[]
        for tool,tool_num in self.tool_list:
            tools+=[tool+str(i+1) for i in range(tool_num)]
        subprocess.call('rosservice call gazebo/reset_simulation',shell=True)
        
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: floor'+ '%d' %(self.floor_index+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-12.5, -12.5), shell=True)

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: wall'+ '%d' %(self.wall_index+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-12.5, -12.5), shell=True)
        
        for i,tool in enumerate(tools):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: %s' %(tool) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(self.rand_coord_list[i][0], self.rand_coord_list[i][1]), shell=True)

        print('-'*50 +'\n Randomized environment model set done.')



        print('--------------------------------------------------Start--------------------------------------------------')
        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: youbot' + ', pose: { position: { x: 0, y: 0 ,z: 10.0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)
        print('--------------------------------------------------End--------------------------------------------------')

        return self.rand_coord_list[len(tools)]

    def gazebo_warmup(self):
        for i in range(0, self.lathe_num):
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_machine/' + self.tool_list[0][0] +'/model.sdf -sdf -model lathe{0} -y -40 -x -40'.format(i+1), shell=True)
        
        for i in range(0, self.systec_num):
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_machine/' + self.tool_list[1][0] +'/model.sdf -sdf -model systec{0} -y -40 -x -40'.format(i+1), shell=True)

        for i in range(0, self.floor_texture_num):
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_floor/' + self.floor_list[i] +'_floor/model.sdf -sdf -model floor{0} -y -50 -x -50'.format(i+1), shell=True)

        for i in range(0, self.wall_texture_num):
            subprocess.call('rosrun gazebo_ros spawn_model -file ' + self.gazebo_model_path + 'env_wall/' + self.wall_list[i] +'_wall/model.sdf -sdf -model wall{0} -y -50 -x -50'.format(i+1), shell=True)

        time.sleep(5)

    def remainder_clear(self):
        print('Clearing...')
        for i in range(0, self.floor_texture_num):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: floor'+ '%d' %(i+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-50, -50), shell=True)

        for i in range(0, self.wall_texture_num):
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: wall'+ '%d' %(i+1) +', pose: { position: { x: %d, y: %d ,z: 0 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'' %(-50, -50), shell=True)

        print('Done.')