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

class testEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "test.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = spaces.Discrete(6) #F,B,L,R,LF,RF
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def discretize_observation(self,lidar,new_ranges,sonar_front,sonar_rear,sonar_left,sonar_right):
        discretized_ranges = []
        min_range = 0.2
        min_sonar_range = 0.15
        done = False
        mod = len(lidar.ranges)/new_ranges
        for i, item in enumerate(lidar.ranges):
            if (i%mod==0):
                if lidar.ranges[i] == float ('Inf') or np.isinf(lidar.ranges[i]):
                    discretized_ranges.append(6)
                elif np.isnan(lidar.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(lidar.ranges[i]))
            if (min_range > lidar.ranges[i] > 0):
                done = True
            if (min_sonar_range > sonar_front.range > 0):
                done = True
            if (min_sonar_range > sonar_rear.range > 0):
                done = True
            if (min_sonar_range > sonar_left.range > 0):
                done = True
            if (min_sonar_range > sonar_right.range > 0):
                done = True            
        return discretized_ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #BACKWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)    
        elif action == 2: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 3: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = -0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 4: #LEFT_FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 5: #RIGHT_FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)

        sonar_front = None
        while sonar_front is None:
            try:
                lidar = rospy.wait_for_message('/scan_unified', LaserScan, timeout=5)
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                rgb_i = np.fromstring(rgb.data, np.uint8)
                depth_i = np.fromstring(depth.data, np.uint8)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.discretize_observation(lidar,5,sonar_front,sonar_rear,sonar_left,sonar_right)

        if not done:
            if action == 0:
                reward = 5
            else:
                reward = 1
        else:
            reward = -200

        return state, reward, done, {}

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
        sonar_front = None
        while sonar_front is None:
            try:
                lidar = rospy.wait_for_message('/scan_unified', LaserScan, timeout=5)
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                rgb_i = np.fromstring(rgb.data, np.uint8)
                depth_i = np.fromstring(depth.data, np.uint8)
            except:
                pass
    
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state = self.discretize_observation(lidar,5,sonar_front,sonar_rear,sonar_left,sonar_right)

        return state
