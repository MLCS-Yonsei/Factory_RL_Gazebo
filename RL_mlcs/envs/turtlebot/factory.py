import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from RL_mlcs.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image 

from gym.utils import seeding

class factoryEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "test.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = 3
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def wrap_observation(self,lidar,new_ranges,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth):
        ranges = []
        sonars = []
        min_range = 0.45
        min_sonar_range = 0.3
        done = False
        mod = len(lidar.ranges)/new_ranges
        for i, item in enumerate(lidar.ranges):
            if (i%mod==0):
                if item == float ('Inf') or np.isinf(item):
                    ranges.append(6.0)
                elif np.isnan(item):
                    ranges.append(0.0)
                else:
                    ranges.append(item)
            if (min_range > item > 0):
                done = True
        for item in [sonar_front,sonar_rear,sonar_left,sonar_right]:
            sonars.append(item.range)
            if (min_sonar_range > item.range > 0):
                done = True
        rgb = np.reshape(np.fromstring(rgb.data, np.uint8),[480,640,3])
        depth = np.reshape(np.fromstring(depth.data, np.uint8),[480,640,4])
        return ranges,sonars,rgb,depth,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.linear.y = action[1]
        vel_cmd.angular.z = action[2]
        self.vel_pub.publish(vel_cmd)
        
        lidar = None
        while lidar is None:
            try:
                lidar = rospy.wait_for_message('/scan_unified', LaserScan, timeout=5)
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        ranges,sonars,rgb,depth,done = self.wrap_observation(lidar,5,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return ranges,sonars,rgb,depth,reward,done,{}

    def _reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read lidar data
        lidar = None
        while lidar is None:
            try:
                lidar = rospy.wait_for_message('/scan_unified', LaserScan, timeout=5)
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
    
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        ranges,sonars,rgb,depth,done = self.wrap_observation(lidar,5,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth)

        return ranges,sonars,rgb,depth
