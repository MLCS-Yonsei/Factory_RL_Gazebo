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

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Range
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
#from bringup_dual.msg import commendMsg

from rosgraph_msgs.msg import Clock
import tf
from gym.utils import seeding
from gym.spaces import Box, Discrete

class ddpgEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "vehicle_v2.launch")
        # self.vel_pub = rospy.Publisher('/ns1/cmd_msg', Pose2D, queue_size=5) # publisher for pose control
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5) # publisher for velocity control
        self.reset_pub = rospy.Publisher('/gazebo_reset', String, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.action_space = 3
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.min_scan_range = []
        for idx in range(36):
            self.min_scan_range.append(
                min(
                    np.abs(0.5/np.cos(np.pi*(2.0*idx+1.0/36.0))),
                    np.abs(0.4/np.sin(np.pi*(2.0*idx+1.0/36.0)))
                )
            )
        self.min_sonar_range = 0.1
        self.min_dist_range = 0.1
        self.odom_data_tmp = [0,0,0,0,0,0]
        self.action_space = spaces.Box(low=np.array([-0.2,-0.2,-0.5]),high=np.array([0.2,0.2,0.5]))
        self.target_set=[[17.5,-4.5],[17.5,0.5],[12.5,0.5],[7.5,-4.5],[7.5,0.5],[2.5,-4.5],[-2.5,-4.5],[-2.5,0.5],[-2.5,5.5],[-7.5,5.5],[-12.5,0.5],[-17.5,-4.5],[-17.5,0.5]]
        
    def calculate_observation(self,scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data):
        scan_data=[]
        sonar_data = []
        done = False
        crash = False
        reward=0
        #Scan normalize by dividing by 10
        for i, item in enumerate(scan.ranges):
            if i % 10 == 0:
                scan_data.append(min(scan.ranges[i:i+9]))
            if (self.min_scan_range[i/10] > item > 0):
                crash = True
        for idx, item in enumerate(scan_data):
            if np.isinf(item):
                scan_data[idx]= 11.0
        #Sonar unifier
        for item in [sonar_front,sonar_rear,sonar_left,sonar_right]:
            sonar_data.append(item.range)
            if (self.min_sonar_range > item.range > 0):
                crash = True
        #Relative distance & angle
        dist_to_target = math.sqrt((self.target[0] - odom_data[0])**2 + (self.target[1] - odom_data[1])**2)
        print('Target pose : '),
        print(self.target)
        print('Odom :'),
        print(odom_data[:2])
        print('Distance to Goal :'),
        print(dist_to_target)
        print('=====================================================================================')
        angle_to_target = np.arctan2((self.target[1] - odom_data[1]),(self.target[0] - odom_data[0])) - odom_data[2]
        if angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        if angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        target_angle = self.target[2]-odom_data[5]
        if target_angle > np.pi:
            target_angle -= 2 * np.pi
        if target_angle < -np.pi:
            target_angle += 2 * np.pi
        state = scan_data+sonar_data+[dist_to_target,angle_to_target,target_angle]
        
        reward = -2*(target_angle)**2
        if type(self.dist_to_target_prev) != type(None):
            reward += 10*(self.dist_to_target_prev-dist_to_target)
        self.dist_to_target_prev = dist_to_target
        
        if crash:
            done = True
            reward -= 100
        elif (self.min_dist_range > dist_to_target) and np.abs(target_angle)<0.1:
            done = True
            reward += 100
            print('#####################################################################################')
            print('Goal arrived')
            print('#####################################################################################')
        return state, reward, done


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


    def step(self, action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        lin_vel = (action[0]**2+action[1]**2)**0.5
        vel_cmd = Twist()
        if lin_vel > 0.2:
            vel_cmd.linear.x = 0.2*action[0]/lin_vel
            vel_cmd.linear.y = 0.2*action[1]/lin_vel
            vel_cmd.angular.z = 0.2*action[2]/lin_vel
        else:
            vel_cmd.linear.x = action[0]
            vel_cmd.linear.y = action[1]
            vel_cmd.angular.z = action[2]
        self.vel_pub.publish(vel_cmd)
        
        rostime = None
        while rostime is None:
            try:
                rostime = rospy.wait_for_message('/clock', Clock, timeout=5).clock
            except:
                pass
        
        odom = None
        while odom is None:
            try:
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5).pose.pose
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
                rgb =  rospy.wait_for_message('/kinect_rgb_camera/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass

        depth = None
        while depth is None:
            try:
                depth =  rospy.wait_for_message('/kinect_depth_camera/camera/depth/image_raw', Image, timeout=5)
            except:
                pass

        odom_data = self.odom_to_data(odom)
        odom_data_tmp = odom_data
        timestamp = rostime.secs+rostime.nsecs/1e+9

        state,reward,done = self.calculate_observation(scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data)

        self.vel_x_prev = action[0]
        self.vel_y_prev = action[1]
        self.vel_z_prev = action[2]

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

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            subprocess.call('rosservice call /gazebo/set_model_state \'{model_state: { model_name: vehicle_v2' + ', pose: { position: { x: 0, y: 0 ,z: 0.3 }, orientation: {x: 0, y: 0, z: 0, w: 0 } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world } }\'', shell=True)
            self.reset_pub.publish("True")
            print ('##########################')
            print ("## Robot position reset ##")
            print ('##########################')
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        
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
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5).pose.pose
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
                rgb =  rospy.wait_for_message('/kinect_rgb_camera/camera/rgb/image_raw', Image, timeout=5)
            except:
                pass
        depth = None
        while depth is None:
            try:
                depth =  rospy.wait_for_message('/kinect_depth_camera/camera/depth/image_raw', Image, timeout=5)
            except:
                pass
        odom_data = self.odom_to_data(odom)
        self.odom_data_tmp = odom_data
        self.ang_vel_prev = 0
        self.lin_vel_prev = 0
        self.target = choice(self.target_set)+[np.random.uniform(-np.pi,np.pi,size=None)]
        print(self.target)
        print(odom_data)
        self.dist_to_target_prev = None
        state,reward,done = self.calculate_observation(scan,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,odom_data)

        self.state_prev = state
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        return state
