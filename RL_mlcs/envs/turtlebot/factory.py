import gym
import time
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
from nav_msgs.msg import Odometry

from rosgraph_msgs.msg import Clock
import tf
from gym.utils import seeding
from gym.spaces import Box, Discrete

class factoryEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "test.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self.action_space = 3
        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.min_scan_range = 0.2
        self.min_sonar_range = 0.2
        self.min_dist_range = 0.3
        self.odom_data_tmp = [0,0,0,0,0,0]
        self.action_space = spaces.Box(low=np.array([0.0,-1.0]),high=np.array([0.5,1.0]))
        self.target = [0.0, 0.0]

    def calculate_observation(self,scan_front,scan_rear,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,pos_data):
        scan_data=[]
        sonar_data = []
        done = False
        #Scan normalize by dividing by 10
        for i, item in enumerate(scan_front.ranges):
            if i % 10 == 0:
                scan_data.append(min(scan_front.ranges[i:i+9])/5.0)
            if (self.min_scan_range > item > 0):
                done = True
        for i, item in enumerate(scan_rear.ranges):
            if i % 10 == 0:
                scan_data.append(min(scan_rear.ranges[i:i+9])/5.0)
            if (self.min_scan_range > item > 0):
                done = True
        #Sonar unifier
        for item in [sonar_front,sonar_rear,sonar_left,sonar_right]:
            sonar_data.append(item.range)
            if (self.min_sonar_range > item.range > 0):
                done = True                
        #RGB reshape
        rgb = np.reshape(np.fromstring(rgb.data, np.uint8),[480,640,3])
        depth = np.reshape(np.fromstring(depth.data, np.uint8),[480,640,4])
        rgbd = np.concatenate((rgb,depth),axis=2)
        #Relative distance & angle
        dist_to_target = ((self.target[0] - pos_data[0])**2 + (self.target[1] - pos_data[1])**2)**0.5
        angle_to_target = np.arctan2((self.target[1] - pos_data[1]),(self.target[0] - pos_data[0])) - pos_data[2]
        if angle_to_target > np.pi:
            angle_to_target -= 2 * np.pi
        if angle_to_target < -np.pi:
            angle_to_target += 2 * np.pi
        #State = [laser_scan_0 ~ laser_scan_9, sonar_front ~ sonar_ right, v_x_t-1, v_y_t-1, v_z_t-1, angle_to_target, distance_to_target, rgb, depth]
        state={}
        state['vector'] = scan_data+sonar_data+[dist_to_target/5.0,angle_to_target/np.pi]
        state['rgbd'] = rgbd
        if (self.min_dist_range > dist_to_target):
            done = True
        return state,done


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
                odom = rospy.wait_for_message('/odom', Odometry, timeout=5).pose.pose
            except:
                pass

        scan_front = None
        while scan_front is None:
            try:
                scan_front = rospy.wait_for_message('/scan_front', LaserScan, timeout=5)
                scan_rear = rospy.wait_for_message('/scan_rear', LaserScan, timeout=5)
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
        position_x = np.cos(self.odom_data_tmp[5]) * (odom_data[0] - self.odom_data_tmp[0]) + np.sin(self.odom_data_tmp[5]) * (odom_data[1] - self.odom_data_tmp[1])
        position_y = -np.sin(self.odom_data_tmp[5]) * (odom_data[0] - self.odom_data_tmp[0]) + np.cos(self.odom_data_tmp[5]) * (odom_data[1] - self.odom_data_tmp[1])
        orientation_yaw = odom_data[5] - self.odom_data_tmp[5]
        if orientation_yaw > np.pi:
            orientation_yaw -= 2 * np.pi
        if orientation_yaw < -np.pi:
            orientation_yaw += 2 * np.pi
        pos_data = [position_x,position_y,orientation_yaw]
        state,done = self.calculate_observation(scan_front,scan_rear,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,pos_data)

        self.vel_x_prev = vel_x
        self.vel_y_prev = vel_y
        self.vel_z_prev = vel_z

        distance_decrease = (self.state_prev['vector'][-2] - state['vector'][-2]) * 5.0
        reward = distance_decrease
        if done:
            reward-=10
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
            self.odom_data_tmp = odom_data_tmp
        self.state_prev = state
        return state, reward, done, {}

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     #reset_proxy.call()reset_proxy.call()
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        # target_set = [[-5,-4],[-5,0],[-5,1],[-5,2],[-5,3],[-5,4],\
        #              [-4,-4],[-4,0],[-4,1],[-4,4],\
        #              [-3,-4],[-3,-3],[-3,-2],[-3,0],[-3,1],[-3,4],\
        #              [-2,1],[-2,2],[-2,3],[-2,4],\
        #              [-1,-2],[-1,-1],[-1,1],[-1,4],\
        #              [0,-4],[0,-3],[0,-2],[0,-1],[0,1],\
        #              [1,-4],[1,-2],[1,-1],[1,0],[1,1],[1,2],[1,3],[1,4],\
        #              [2,-4],[2,-1],[2,0],[2,1],[2,3],[2,4],\
        #              [3,-4],[3,-3],[3,-2],[3,-1],[3,0],[3,1],[3,3],[3,4],\
        #              [4,-4],[4,-3],[4,-2],[4,-1],[4,0],[4,1],[4,2],[4,3],[4,4]]
        #read scan data
        # self.target = [3.5,-3.5]
        # self.target_x = self.target[0]
        # self.target_y = self.target[1]
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

        scan_front = None
        while scan_front is None:
            try:
                scan_front = rospy.wait_for_message('/scan_front', LaserScan, timeout=5)
                scan_rear = rospy.wait_for_message('/scan_rear', LaserScan, timeout=5)
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
        position_x = np.cos(self.odom_data_tmp[5]) * (odom_data[0] - self.odom_data_tmp[0]) + np.sin(self.odom_data_tmp[5]) * (odom_data[1] - self.odom_data_tmp[1])
        position_y = -np.sin(self.odom_data_tmp[5]) * (odom_data[0] - self.odom_data_tmp[0]) + np.cos(self.odom_data_tmp[5]) * (odom_data[1] - self.odom_data_tmp[1])
        orientation_yaw = odom_data[5] - self.odom_data_tmp[5]
        if orientation_yaw > np.pi:
            orientation_yaw -= 2 * np.pi
        if orientation_yaw < -np.pi:
            orientation_yaw += 2 * np.pi
        pos_data = [position_x, position_y, orientation_yaw]
        self.ang_vel_prev = 0
        self.lin_vel_prev = 0
        state,done = self.calculate_observation(scan_front,scan_rear,sonar_front,sonar_rear,sonar_left,sonar_right,rgb,depth,pos_data)

        self.state_prev = state
        return state

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
