import random
import scipy.io as sio
import time
import threading
import rclpy
import numpy as np
import tensorflow as tf

from math import atan2, pi
from squaternion import Quaternion

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import EntityState
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty

from agent import ActorNetwork

import logging

MISSIONS = 20

last_odom = None
lidar_data = None
num_collisions = 0

class GazeboEnv(Node):
	def __init__(self):
		super().__init__('gazebo_env')

		self.state_num = 28
		self.action_num = 2
		self.observation_space = np.empty(self.state_num)
		self.action_space = np.empty(self.action_num)

		self.target_x = 0.0                 
		self.target_y = 0.0                

		self.set_self_state = EntityState()
		self.set_self_state.name = "turtlebot3_waffle"
		self.set_self_state.pose.position.x = 0.0
		self.set_self_state.pose.position.y = 0.0
		self.set_self_state.pose.position.z = 0.0
		self.set_self_state.pose.orientation.x = 0.0
		self.set_self_state.pose.orientation.y = 0.0
		self.set_self_state.pose.orientation.z = 0.0
		self.set_self_state.pose.orientation.w = 1.0

		# Set up the ROS publishers and subscribers
		self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)                          
		self.set_state = self.create_client(SetEntityState, "/plugin/set_entity_state")

		self.unpause = self.create_client(Empty, "/unpause_physics")                          
		self.pause = self.create_client(Empty, "/pause_physics")                             
		self.reset_proxy = self.create_client(Empty, "/reset_world")                       


	def turtlebot_is_crashed(self, laser_values, range_limit):
		global num_collisions
		laser_crashed_reward = 0.0
		done = 0

		if min(laser_values) < 2*range_limit:
			laser_crashed_reward = -80.0

		if min(laser_values) < range_limit:
			print("======================================================================================================")
			print("=                                                                                 COLLISION DETECED  = ")
			print("======================================================================================================")
			done = 1
			laser_crashed_reward = -200.0
			num_collisions += 1

		return laser_crashed_reward, done


	def step(self, time_step, angular_z, linear_x):
		global lidar_data
		global last_odom

		vel_cmd = Twist()
		vel_cmd.linear.x = float((linear_x+1)/2*0.26)
		vel_cmd.angular.z = float(angular_z)

		turtlebot_x_previous = last_odom.pose.pose.position.x
		turtlebot_y_previous = last_odom.pose.pose.position.y

		self.vel_pub.publish(vel_cmd)

		while not self.unpause.wait_for_service(timeout_sec=1.0):
			print('Unpause: service /unpause_physics not available, waiting again...')
		try:
			self.unpause.call_async(Empty.Request())
		except:
			print("/Unpause_physics service call failed")

		# propagate state for time_step seconds
		time.sleep(time_step)

		extra_time = 0.0
		if min(lidar_data.ranges) < 0.2:
			extra_start = time.time()
			while min(lidar_data.ranges) < 0.2:
				pass
			extra_end = time.time()
			extra_time = extra_end - extra_start
			print(f"WAIT EXTRA TIME FOR LIDAR DATA: {extra_time}")


		while not self.pause.wait_for_service(timeout_sec=1.0):
			print('Pause: service /pause_physics not available, waiting again...')
		try:
			self.pause.call_async(Empty.Request())
		except (rclpy.ServiceException) as e:
			print("Pause_physics service call failed")

		turtlebot_x = last_odom.pose.pose.position.x
		turtlebot_y = last_odom.pose.pose.position.y

		quaternion = Quaternion(
			last_odom.pose.pose.orientation.w,
			last_odom.pose.pose.orientation.x,
			last_odom.pose.pose.orientation.y,
			last_odom.pose.pose.orientation.z,
		)

		euler = quaternion.to_euler(degrees=False)
		turtlebot_yaw_angle = round(euler[2], 4)

		# make input, angle between the turtlebot and the target
		turtlebot_goal_angle = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

		angle_diff = turtlebot_goal_angle - turtlebot_yaw_angle
		if angle_diff < -pi:
			angle_diff += 2*pi
		if angle_diff > pi:
			angle_diff -= 2*pi

		# prepare the normalized laser value and check if it is crash
		laser_values = []
		# process data from lidar
		for i in range(len(lidar_data.ranges)):
			if lidar_data.ranges[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data.ranges[i]):
				laser_values.append(0.0)
			else:
				laser_values.append(lidar_data.ranges[i])

		normalized_laser_values= [(x)/3.5 for x in laser_values]

		# prepare state
		current_distance_turtlebot_goal = np.linalg.norm([turtlebot_x - self.target_x, turtlebot_y - self.target_y])

		state = np.append(normalized_laser_values, current_distance_turtlebot_goal)
		state = np.append(state, angle_diff)
		state = np.append(state, (linear_x+1)/2*0.26) 
		state = np.append(state, angular_z)

		# make distance reward
		previous_distance_turtlebot_goal = np.linalg.norm([turtlebot_x_previous - self.target_x, turtlebot_y_previous - self.target_y])

		distance_reward = previous_distance_turtlebot_goal - current_distance_turtlebot_goal

		# make collision reward
		laser_crashed_reward, done = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
		laser_reward = sum(normalized_laser_values) - 24.0
		collison_reward = laser_crashed_reward + laser_reward

		# make velocity punish reward
		# velocity_reward = (linear_x+1)/2*0.26*1.5 - abs(angular_z)

		angular_punish_reward = 0.0
		linear_punish_reward = 0.0

		if angular_z > 0.8 or angular_z < -0.8:
			self.angular_punish_reward = -1.0

		if linear_x < -0.6:
			self.linear_punish_reward = -2.0

		# make arrive reward
		arrive_reward = 0.0
		if current_distance_turtlebot_goal < 1:
			print("======================================================================================================")
			print("=                                                                                       REACH GOAL!  = ")
			print("======================================================================================================")
			arrive_reward = 100
			done = 1


		# total reward
		reward = distance_reward * (5/(time_step+extra_time))*7 + arrive_reward + collison_reward + angular_punish_reward + linear_punish_reward

		return  state, reward, done


	def reset(self, target_x=8.0, target_y=8.0, robot_x=0.0, robot_y=0.0, is_random=True):
		global lidar_data
		global last_odom

		if is_random:
			index_list_1 = [-1, 1]
			index_list_2 = [-1, -0.5, 0, 0.5, 1]
			a = np.random.randint(0, 2)
			if a == 0:
				index_x = random.choice(index_list_1)
				index_y = random.choice(index_list_2)
			else:
				index_y = random.choice(index_list_1)
				index_x = random.choice(index_list_2)

			# for maze
			self.target_x = (np.random.random()-0.5) + 9*index_x
			self.target_y = (np.random.random()-0.5) + 9*index_y
		else:
			self.target_x = target_x
			self.target_y = target_y

	
		# for corridor
		# self.target_x = (np.random.random()-0.5)*5 + 14*index_x
		# self.target_y = (np.random.random()-0.5)*3

		# Resets the state of the environment and returns an initial observation.
		while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
			print("Reset : service /reset_world not available, waiting again..")
		try:
			self.reset_proxy.call_async(Empty.Request())
			print("Reset Gazebo environment succeed")
		except rclpy.ServiceException as e:
			print("/gazebo/reset_simulation service call failed")


		angle = np.random.uniform(-np.pi, np.pi)
		quaternion = Quaternion.from_euler(0.0, 0.0, angle)
		
		# for maze
		if is_random:
			x = 0.0
			y = (np.random.random() - 0.5)*8 
		else:
			x = robot_x
			y = robot_y


		robot_state = self.set_self_state
		robot_state.pose.position.x = x
		robot_state.pose.position.y = y
		robot_state.pose.orientation.x = quaternion.x
		robot_state.pose.orientation.y = quaternion.y
		robot_state.pose.orientation.z = quaternion.z
		robot_state.pose.orientation.w = quaternion.w

		request = SetEntityState.Request()
		request._state = robot_state

		while not self.set_state.wait_for_service(timeout_sec=1.0):
			print("Set state: service /plugin/set_state not available, waiting again..")

		future = self.set_state.call_async(request)
		rclpy.spin_until_future_complete(self, future)

		if future.result().success:
			print(f"Set robot state succeed at posittion: {x}, {y}")
		else:
			print("Set robot state failed")


		goal_state = EntityState()
		goal_state.name = "target"
		goal_state.pose.position.x = self.target_x
		goal_state.pose.position.y = self.target_y
		goal_state.pose.position.z = 0.0
		goal_state.pose.orientation.x = 0.0
		goal_state.pose.orientation.y = 0.0
		goal_state.pose.orientation.z = 0.0
		goal_state.pose.orientation.w = 0.0

		request = SetEntityState.Request()
		request._state = goal_state
		while not self.set_state.wait_for_service(timeout_sec=1.0):
			print("Set state: service /plugin/set_state not available, waiting again..")

		future = self.set_state.call_async(request)
		rclpy.spin_until_future_complete(self, future)

		if future.result().success:
			print(f"New goal posittion: {self.target_x}, {self.target_y}")
		else:
			print("Set new goal failed")


		vel_cmd = Twist()
		vel_cmd.linear.x = 0.0
		vel_cmd.angular.z = 0.0
		self.vel_pub.publish(vel_cmd)


		while not self.unpause.wait_for_service(timeout_sec=1.0):
			print('Unpause: service /unpause_physics not available, waiting again...')

		try:
			self.unpause.call_async(Empty.Request())
		except:
			print("/gazebo/unpause_physics service call failed")

		time.sleep(0.2)

		while not self.pause.wait_for_service(timeout_sec=1.0):
			print('Pause: service /pause_physics not available, waiting again...')

		try:
			self.pause.call_async(Empty.Request())
		except:
			print("/gazebo/pause_physics service call failed")


		turtlebot_x = last_odom.pose.pose.position.x
		turtlebot_y = last_odom.pose.pose.position.y

		quaternion = Quaternion(
			last_odom.pose.pose.orientation.w,
			last_odom.pose.pose.orientation.x,
			last_odom.pose.pose.orientation.y,
			last_odom.pose.pose.orientation.z,
		)

		euler = quaternion.to_euler(degrees=False)
		turtlebot_yaw_angle = round(euler[2], 4)

		# make input, angle between the turtlebot and the target
		turtlebot_goal_angle = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

		angle_diff = turtlebot_goal_angle - turtlebot_yaw_angle
		if angle_diff < -pi:
			angle_diff += 2*pi
		if angle_diff > pi:
			angle_diff -= 2*pi

		# prepare the normalized laser value and check if it is crash
		laser_values = []
		# process data from lidar
		for i in range(len(lidar_data.ranges)):
			if lidar_data.ranges[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data.ranges[i]):
				laser_values.append(0.0)
			else:
				laser_values.append(lidar_data.ranges[i])

		normalized_laser_values= [(x)/3.5 for x in laser_values]

		# prepare state
		current_distance_turtlebot_goal = np.linalg.norm([turtlebot_x - self.target_x, turtlebot_y - self.target_y])

		state = np.append(normalized_laser_values, current_distance_turtlebot_goal)
		state = np.append(state, angle_diff)
		state = np.append(state, 0.0) 
		state = np.append(state, 0.0)

		return state


class OdomSubscriber(Node):
	def __init__(self):
		super().__init__('odom_subscriber')
		self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

	def odom_callback(self, od_data):
		global last_odom
		last_odom = od_data


class LidarSubscriber(Node):
	def __init__(self):
		super().__init__('lidar_subscriber')
		self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

	def lidar_callback(self, li_data):
		global lidar_data
		lidar_data = li_data
		

if __name__ == '__main__':
		
	rclpy.init(args=None)

	env = GazeboEnv()
	odom_subscriber = OdomSubscriber()
	lidar_subscriber = LidarSubscriber()

	e = 170
	s = 74621

	path = 'actor_model-' + str(e) + '-' + str(s) + '.weights.h5'
	actor = ActorNetwork()
	actor.load_weights(path)	

	executor = MultiThreadedExecutor()
	executor.add_node(env)
	executor.add_node(odom_subscriber)
	executor.add_node(lidar_subscriber)

	executor_thread = threading.Thread(target=executor.spin, daemon=True)
	executor_thread.start()

	rate = env.create_rate(100)

	step_reward = [0,0]
	step_Q = [0,0]

	step = 0

	start_train_time = time.time()

	print("======================================================================================================")
	print("=                          TEST MODEL " + path + "                                  =")
	print("======================================================================================================")

	try:
		while rclpy.ok():
			for ep in range(1, MISSIONS):
				print("======================================================================================================")
				print(f"=                                                                                      MISSION: {ep}    = ")
				print("======================================================================================================")

				# tx = float(input("Enter target x: "))
				# ty = float(input("Enter target y: "))
				# rx = float(input("Enter robot x: "))
				# ry = float(input("Enter robot y: "))

				# current_state = env.reset(tx, ty, rx, ry, False)
				current_state = env.reset()
				total_reward = 0
				done = 0
				while done == 0:
					action = actor(tf.expand_dims(current_state, 0))[0].numpy()
					new_state, reward, done = env.step(time_step=0.1, angular_z=action[0], linear_x=action[1])
					print(f"action is: linear: {action[1]:.2f}, angular: {action[0]:.2f}, reward: {reward:.2f}")
					
					total_reward += reward
					step += 1

					current_state = new_state

					if done == 1:
						m, s = divmod(int(time.time() - start_train_time), 60)
						h, m = divmod(m, 60)
						print(f"Episode: {ep} score: {total_reward:.2f}  total step: {step}, time: {h}:{m:02d}:{s:02d}")
						break
			
			print(f"Finish {MISSIONS} missions with {num_collisions} collision in {step} steps")
			break

	except KeyboardInterrupt:
		pass

	rclpy.shutdown()
