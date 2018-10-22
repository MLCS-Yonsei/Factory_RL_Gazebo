import gym
import time
import rospy
import roslaunch
import time
import subprocess
import numpy as np
import math
from random import choice

from gym import utils, spaces
from RL_mlcs.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from bringup_dual.msg import commendMsg
from env_reset import env_reset

from rosgraph_msgs.msg import Clock
import tf
from gym.utils import seeding
from gym.spaces import Box, Discrete

class srlEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "vehicle_v2.launch")
        self.cmd_pub = rospy.Publisher('/ns1/cmd_msg', commendMsg, queue_size=5)
        self.odom_pub = rospy.Publisher('/pose', Odometry, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.action_space = 3
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.min_scan_range = 0.5
        self.min_sonar_range = 0.1
        self.min_dist_range = 0.1
        self.odom_data_tmp = [0,0,0,0,0,0]
        self.action_space = spaces.Box(low=np.array([-0.2,-0.2,-0.5]),high=np.array([0.2,0.2,0.5]))
        self.target_set=[
            [ 17.5,-4.5],
            [ 17.5, 0.5],
            [ 12.5, 0.5],
            [  7.5,-4.5],
            [  7.5, 0.5],
            [  2.5,-4.5],
            [ -2.5,-4.5],
            [ -2.5, 0.5],
            [ -2.5, 5.5],
            [ -7.5, 5.5],
            [-12.5, 0.5],
            [-17.5,-4.5],
            [-17.5, 0.5]
        ]
        self.target = [0.0, 0.0]
        self.vel_x_prev = 0.0
        self.vel_y_prev = 0.0
        self.vel_phi_prev = 0.0
        self.scan_buffer = {'-2':None, '-1':None}
        self.depth_buffer = {'-2':None, '-1':None}
        
    def calculate_observation(self,odom_data,action):
        scan_data = []
        sonar_data = []
        done = False
        reward = -action[2]**2
        #Scan normalize by dividing by 10
        for i, item in enumerate(self.scan.ranges):
            if i % 10 == 0:
                scan_data.append(min(scan.ranges[i:i+9]))
            if (self.min_scan_range > item > 0):
                done = True
                reward-=10
        scan_data = np.reshape(scan_data,[-1,1,1])
        if done:
            print('LiDAR detected')
        if self.scan_buffer['-2'] == None:
            self.scan_buffer['-2'] = scan_data
            self.scan_buffer['-1'] = scan_data

        #Sonar unifier
        for key in self.sonar.keys():
            sonar_data.append(self.sonar[key].range)
            if (self.min_sonar_range > self.sonar[key].range > 0):
                done = True
                reward-=10  
        if done:
            print('========================================================')
            print('Sonar detected')
            print('========================================================')
        #RGB reshape
        rgb = np.reshape(np.fromstring(self.rgb.data, np.uint8),[96,128,3])
        depth = self.depth_from_raw(np.reshape(np.fromstring(self.depth.data, np.uint8),[96,128,4]))
        if self.depth_buffer['-2'] == None:
            self.depth_buffer['-2'] = depth
            self.depth_buffer['-1'] = depth
        #Relative distance & angle
        dist_to_target = math.sqrt((self.target[0] - odom_data[0])**2 + (self.target[1] - odom_data[1])**2)
        reward += 5*(self.dist_to_target_prev - dist_to_target)
        self.dist_to_target_prev = dist_to_target
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
        state['lidar'] = np.concatenate([scan_data,self.scan_buffer['-1'],self.scan_buffer['-2']],axis=2)
        state['proximity'] = sonar_data
        state['control'] = [self.vel_x_prev, self.vel_y_prev, self.vel_phi_prev]
        state['goal'] = [dist_to_target,angle_to_target]
        state['rgb'] = rgb
        state['depth'] = np.concatenate([depth,self.depth_buffer['-1'],self.depth_buffer['-2']],axis=2)
        self.scan_buffer['-2'] = self.scan_buffer['-1']
        self.scan_buffer['-1'] = scan_data
        self.depth_buffer['-2'] = self.depth_buffer['-1']
        self.depth_buffer['-1'] = depth
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

    def depth_from_raw(self, raw):
        depth = np.zeros([96, 128, 1], dtype=np.float32)
        for idx in range(2):
            depth += raw[:,:,idx].astype(np.float32)*256**idx
        depth += np.fmin(raw[:,:,3].astype(np.float32)-63.0, 1.0)
        depth /= 2.0**17
        return depth.astype(np.uint8)
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        pose_cmd = commendMsg()
        phi = self.odom_data_tmp[5]
        c = np.cos(phi)
        s = np.sin(phi)
        pose_cmd.xd = self.odom_data_tmp[0]+c*action[0]-s*action[1]
        pose_cmd.yd = self.odom_data_tmp[1]+s*action[0]+c*action[1]
        pose_cmd.phid = phi+action[2]
        self.cmd_pub.publish(pose_cmd)
        
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

        self.scan = None
        while self.scan is None:
            try:
                self.scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        self.sonar = {'front':None}
        while self.sonar['front'] is None:
            try:
                self.sonar['front'] = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                self.sonar['rear'] = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                self.sonar['left'] = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                self.sonar['right'] = rospy.wait_for_message('/sonar_right', Range, timeout=5)
            except:
                pass

        self.rgb = None
        while self.rgb is None:
            try:
                self.rgb =  rospy.wait_for_message('/kinect_rgb_camera/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass

        self.depth = None
        while self.depth is None:
            try:
                self.depth =  rospy.wait_for_message('/kinect_depth_camera/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
        


        odom_data = self.odom_to_data(odom)
        self.odom_data_tmp = odom_data
        timestamp = time.secs+time.nsecs/1e+9

        state,reward,done = self.calculate_observation(odom_data, action)

        self.vel_x_prev = action[0]
        self.vel_y_prev = action[1]
        self.vel_phi_prev = action[2]

        distance_decrease = (self.state_prev['vector'][-2] - state['vector'][-2]) * 5.0
        reward += distance_decrease
        if done:
            pose_cmd = commendMsg()
            pose_cmd.xd = odom_data[0]
            pose_cmd.yd = odom_data[1]
            pose_cmd.phid = odom_data[5]
            self.cmd_pub.publish(pose_cmd)
            # self.odom_data_tmp = odom_data_tmp
        self.state_prev = state

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        
        return state, reward, done, {}

    def reset(self):
        
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: vehicle_v2' + ', pose: { position: { x: 0, y: 0 ,z: 0.3 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)
            print ("Robot position reset")
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        pose_cmd = commendMsg()
        pose_cmd.xd = 0.0
        pose_cmd.yd = 0.0
        pose_cmd.phid = 0.0
        self.cmd_pub.publish(pose_cmd)
        odom = None
        while odom is None:
            try:
                # odom = rospy.wait_for_message('/odom', Odometry, timeout=5).pose.pose
                odom = rospy.wait_for_message('/pose', Odometry, timeout=5).pose.pose
            except:
                pass
        self.scan = None
        while self.scan is None:
            try:
                self.scan = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        self.sonar = {'front':None}
        while self.sonar['front'] is None:
            try:
                self.sonar['front'] = rospy.wait_for_message('/sonar_front', Range, timeout=5)
                self.sonar['rear'] = rospy.wait_for_message('/sonar_rear', Range, timeout=5)
                self.sonar['left'] = rospy.wait_for_message('/sonar_left', Range, timeout=5)
                self.sonar['right'] = rospy.wait_for_message('/sonar_right', Range, timeout=5)
            except:
                pass
        self.rgb = None
        while self.rgb is None:
            try:
                self.rgb =  rospy.wait_for_message('/kinect_rgb_camera/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass
        self.depth = None
        while self.depth is None:
            try:
                self.depth =  rospy.wait_for_message('/kinect_depth_camera/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
        odom_data = self.odom_to_data(odom)
        self.odom_data_tmp = odom_data
        self.vel_x_prev = 0.0
        self.vel_y_prev = 0.0
        self.vel_phi_prev = 0.0
        self.scan_buffer = {'-2':None, '-1':None}
        self.depth_buffer = {'-2':None, '-1':None}

        state,reward,done = self.calculate_observation(odom_data, action)

        self.state_prev = state
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return state
