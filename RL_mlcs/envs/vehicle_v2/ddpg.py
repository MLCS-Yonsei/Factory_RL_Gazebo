import gym
import time
import rospy
import roslaunch
import time
import subprocess
import numpy as np
import math

from gym import utils, spaces
from RL_mlcs.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from env_reset import env_reset

from rosgraph_msgs.msg import Clock
import tf
from gym.utils import seeding
from gym.spaces import Box, Discrete

class ddpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "vehicle_v2.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.odom_pub = rospy.Publisher('/pose', Odometry, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.action_space = 3
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.min_scan_range = 0.2
        self.min_sonar_range = 0.1
        self.min_dist_range = 0.1
        self.odom_data_tmp = [0,0,0,0,0,0]
        self.action_space = spaces.Box(low=np.array([-0.2,-0.2,-0.5]),high=np.array([0.2,0.2,0.5]))
        self.target = [0.0, 0.0]
        self.rand_deploy_list=[[None]]

    def set_target(self,target):
        self.target=target
        
    
    
    def calculate_observation(self,scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data):
        scan_data=[]
        sonar_data = []
        done = False
        reward=0
        #Scan normalize by dividing by 10
        for i, item in enumerate(scan.ranges):
            if i % 10 == 0:
                scan_data.append(min(scan.ranges[i:i+9]))
            if (self.min_scan_range > item > 0):
                done = True
                reward-=10
        if done:
            print('LiDAR detected')

        #Sonar unifier
        for item in [sonar_front,sonar_rear,sonar_left,sonar_right]:
            sonar_data.append(item.range)
            if (self.min_sonar_range > item.range > 0):
                done = True
                reward-=10  
        if done:
            print('========================================================')
            print('Sonar detected')
            print('========================================================')
        #RGB reshape
        rgb = np.reshape(np.fromstring(rgb.data, np.uint8),[96,128,3])
        depth = np.reshape(np.fromstring(depth.data, np.uint8),[96,128,4])
        rgbd = np.concatenate((rgb,depth),axis=2)
        #Relative distance & angle
        dist_to_target = math.sqrt((self.target[0] - odom_data[0])**2 + (self.target[1] - odom_data[1])**2)
        print('========================================================')
        print('Target pose')
        print(self.target)
        print('Odom')
        print(odom_data[:2])
        print('Distance to Goal')
        print(dist_to_target)
        print('========================================================')
        angle_to_target = np.arctan2((self.target[1] - odom_data[1]),(self.target[0] - odom_data[0])) - odom_data[2]
        if angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        if angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        state={}
        state['vector'] = scan_data+sonar_data+[dist_to_target,angle_to_target]
        state['rgbd'] = rgbd
        if (self.min_dist_range > dist_to_target):
            done = True
            reward+=10
            print('========================================================')
            print('Goal arrived')
            print('========================================================')
        return state,reward,done


    def odom_to_data(self, odom):
        odom_data = []
        odom_data.append(odom.position.x)	# [0]
        odom_data.append(odom.position.y)	# [1]
        odom_data.append(odom.position.z)	# [2]
        [roll, pitch, yaw] = tf.transformations.euler_from_quaternion([odom.orientation.x, odom.orientation.y, odom.orientation.z, odom.orientation.w])
        odom_data.append(roll)			# [3]
        odom_data.append(pitch)			# [4]
        odom_data.append(yaw)			# [5]
        return odom_data
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_x = action[0]
        vel_y = action[1]
        vel_z = action[2]
        vel_cmd = Twist()
        vel_cmd.linear.x = vel_x
        vel_cmd.linear.y = vel_y
        vel_cmd.angular.z = vel_z
        self.vel_pub.publish(vel_cmd)
        
        time = None
        while time is None:
            try:
                time = rospy.wait_for_message('/clock', Clock, timeout=5).clock
            except:
                pass
        
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/pose', Odometry, timeout=5).pose.pose
            except:
                pass

        scan = None
        while scan is None:
            try:
                scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        sonar_front = None
        while sonar_front is None:
            try:
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
            except:
                pass

        rgb = None
        while rgb is None:
            try:
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass

        depth = None
        while depth is None:
            try:
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        odom_data = self.odom_to_data(odom)
        odom_data_tmp = odom_data
        timestamp = time.secs+time.nsecs/1e+9

        state,reward,done = self.calculate_observation(scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data)

        self.vel_x_prev = vel_x
        self.vel_y_prev = vel_y
        self.vel_z_prev = vel_z

        distance_decrease = (self.state_prev['vector'][-2] - state['vector'][-2]) * 5.0
        reward += distance_decrease
        if done:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
            self.odom_data_tmp = odom_data_tmp
        self.state_prev = state

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        
        return state, reward, done, {}

    def _reset(self):
        if self.rand_deploy_list[0][0] is not None:
            env_reset().rand_move(self.rand_deploy_list[1], self.rand_deploy_list[2])

        self.rand_deploy_list = env_reset().rand_deploy()
        self.target = self.rand_deploy_list[0]

        subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: vehicle_v2' + ', pose: { position: { x: 0, y: 0 ,z: 0.4 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/pose', Odometry, timeout=5).pose.pose
            except:
                pass

        scan = None
        while scan is None:
            try:
                scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        sonar_front = None
        while sonar_front is None:
            try:
                sonar_front = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                sonar_rear = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                sonar_left = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                sonar_right = rospy.wait_for_message('/sonar_right', Range, timeout=5)
            except:
                pass

        rgb = None
        while rgb is None:
            try:
                rgb =  rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass

        depth = None
        while depth is None:
            try:
                depth =  rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
        
        odom_data = self.odom_to_data(odom)
        self.odom_data_tmp = odom_data
        self.ang_vel_prev = 0
        self.lin_vel_prev = 0
        state,reward,done = self.calculate_observation(scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data)

        self.state_prev = state

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return state