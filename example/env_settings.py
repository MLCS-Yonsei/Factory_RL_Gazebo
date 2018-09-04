import os

class env_settings(object):
    def __init__(self):
        
        self.gazebo_model_path = os.path.abspath(os.path.join(__file__, os.pardir))+'/../models/'
        self.floor_list = ['blue', 'darkgrey', 'darkred', 'green', 'lightgrey', 'urethane']
        self.wall_list = ['brownbrick', 'concrete', 'grey', 'oldbrick', 'redbrick']
        self.tool_list_large = ['chemical_tank','extrusion','hopper'] #20*8
        self.tool_list_medium = ['inject','unravel','cnc','cnc_hard'] #10*7
        self.tool_list_small = ['Automatic_Rotary_Packing_Machine','Inspection_Metal_Detector','aircompressor','drilling','forklift','freezer','lathe','milling','mixer_and_grinder','offset_press','plastifier','polishing','tablesawcut','transformer','workbench'] #5*5

        self.floor_texture_num = len(self.floor_list) #Number of the floor type
        self.wall_texture_num = len(self.wall_list) #Number of the wall type
        self.tool_large_num = len(self.tool_list_large)
        self.tool_medium_num = len(self.tool_list_medium)
        self.tool_small_num = len(self.tool_list_small)
        
        self.tool_large_num_desired = 2
        self.tool_medium_num_desired = 4
        self.tool_small_num_desired = 12

        self.x_length = 40 #Factory x direction length
        self.y_length = 30 #Factory y direction length
        self.coord_list_large = [(-10,-11),(10,-11)]
        self.coord_list_medium = [(-15,11.5),(-5,11.5),(5,11.5),(15,11.5)]
        self.coord_list_small = []
        for x in range(0,8):
            for y in range(0,3):
                coord = (-17.5+5*x, 5.5-5*y)
                self.coord_list_small.append(coord)

env_config = env_settings()