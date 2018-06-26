import os

class env_settings(object):
    def __init__(self):
        
        self.gazebo_model_path = os.path.abspath(os.path.join(__file__, os.pardir))+'/../models/'
        self.tool_pop_desired = 5 #Desired total machine tool population
        self.lathe_num = 10
        self.systec_num = 10
        self.floor_list = ['blue', 'darkgrey', 'darkred', 'green', 'lightgrey', 'urethane']
        self.wall_list = ['brownbrick', 'concrete', 'grey', 'oldbrick', 'redbrick']
        self.tool_list = [('lathe',10), ('systec',10)]

        self.floor_texture_num = len(self.floor_list) #Number of the floor type
        self.wall_texture_num = len(self.wall_list) #Number of the wall type
        self.tool_num = len(self.tool_list) #Number of the machine tool type

        self.x_length = 25 #Factory x direction length
        self.y_length = 25 #Factory y direction length
        self.coord_list = []
        for x in range(7):
            for y in range(7):
                self.coord_list.append([4*(x-3), 4*(y-3)])
        self.coord_list.remove([0, 0])


env_config = env_settings()