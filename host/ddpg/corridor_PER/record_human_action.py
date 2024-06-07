import random
import time
import scipy.io as sio

import threading
import rclpy
import numpy as np

from math import atan2, pi
from collections import deque

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from squaternion import Quaternion
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan


""" 
How to record human action:
						  1. Launch gazebo simulation: ros2 launch turtlebot3_description gazebo.launch.py
						  2. Run keyboard_teleop	 : ros2 run keyboard_teleop keyboard_teleop_incremental
						                      or     : ros2 run keyboard_teleop keyboard_teleop_hold
						  3. Run this script         : python3 record_human_action.py
"""



"""
Global variables:
				last_odom   : store odom data of robot state, see OdomSubscriber class
				lidar_data  : store data of lidar sensor, see LidarSubscriber class
				cmd_vel_data: store human action, see CmdVelSubscriber class
"""

last_odom = None
lidar_data = None
cmd_vel_data = None


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

		self.return_action = np.empty(self.action_num)   
		self.return_action = self.return_action.reshape((1, self.action_space.shape[0]))                  


	def turtlebot_is_crashed(self, laser_values, range_limit):
		laser_crashed_reward = 0.0
		done = 0

		for i in range(len(laser_values)):
			if min(laser_values) < 2*range_limit:
				laser_crashed_reward = -80.0
			if laser_values[i] < range_limit:
				print("COLLISION DETECTED")
				done = 1
				laser_crashed_reward = -200.0
				self.reset()
				time.sleep(1)
				break
		return laser_crashed_reward, done

 
	def reset(self):
		index_list_1 = [-1, 1]

		print("reseting. . .")

		index_x = random.choice(index_list_1)
		self.target_x = 12.0*index_x
		self.target_y = (np.random.random()-0.5)*2


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

		# for corrdior
		x = 0.0
		y = 0.0

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

		time.sleep(0.1)

		# making reset state		
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


	def read_action(self):
		while cmd_vel_data is None:
			print("wait for cmd_vel data")

		action_value_total = cmd_vel_data
		print (f"action_value linear x is: {action_value_total.linear.x}") 
		print (f"action_value angular z is: {action_value_total.angular.z}")
		self.return_action[0][0] = action_value_total.angular.z
		self.return_action[0][1] = action_value_total.linear.x
		return self.return_action


	def read_state(self):
		while last_odom is None:
			print("wait for odom data. . . . . . . . . .")

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

		while cmd_vel_data is None:
			print("wait for cmd_vel data. . . . . . . . . .")
		linear_x = cmd_vel_data.linear.x
		angular_z = cmd_vel_data.angular.z

		turtlebot_goal_angle = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

		angle_diff = turtlebot_goal_angle - turtlebot_yaw_angle
		if angle_diff < -pi:
			angle_diff += 2*pi
		if angle_diff > pi:
			angle_diff -= 2*pi

		# prepare the normalized laser value and check if it is crash
		laser_values = []
		# process data from lidar
		while lidar_data is None:
			print("wait for lidar data. . . . . . . . . .")

		for i in range(len(lidar_data.ranges)):
			if lidar_data.ranges[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data.ranges[i]):
				laser_values.append(0)
			else:
				laser_values.append(lidar_data.ranges[i])

		normalized_laser_values= [(x)/3.5 for x in laser_values]

		# prepare state
		current_distance_turtlebot_goal = np.linalg.norm([turtlebot_x - self.target_x, turtlebot_y - self.target_y])

		state = np.append(normalized_laser_values, current_distance_turtlebot_goal)
		state = np.append(state, angle_diff)
		state = np.append(state, linear_x)
		state = np.append(state, angular_z)
		state = state.reshape(1, self.state_num)

		return state


	def read_game_step(self, time_step, linear_x, angular_z):
		while last_odom is None:
			print("wait for odom data. . . . . . . . . .")

		turtlebot_x_previous = last_odom.pose.pose.position.x
		turtlebot_y_previous = last_odom.pose.pose.position.y

		time.sleep(time_step)

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

		turtlebot_target_angle = atan2(self.target_y - turtlebot_y, self.target_x - turtlebot_x)

		angle_diff = turtlebot_target_angle - turtlebot_yaw_angle

		if angle_diff < -pi:
			angle_diff += 2*pi
		if angle_diff > pi:
			angle_diff -= 2*pi

		# prepare the normalized laser value and check if it is crash
		laser_values = []

		# process data from lidar
		while lidar_data is None:
			print("wait for lidar data. . . . . . . . . .")

		for i in range(len(lidar_data.ranges)):
			if lidar_data.ranges[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data.ranges[i]):
				laser_values.append(0)
			else:
				laser_values.append(lidar_data.ranges[i])

		normalized_laser_values= [(x)/3.5 for x in laser_values]

		# prepare state
		current_distance_turtlebot_goal = np.linalg.norm([turtlebot_x - self.target_x, turtlebot_y - self.target_y])

		state = np.append(normalized_laser_values, current_distance_turtlebot_goal)
		state = np.append(state, angle_diff)
		state = np.append(state, linear_x)
		state = np.append(state, angular_z)
		state = state.reshape(1, self.state_num)
		
		# make distance reward
		previous_distance_turtlebot_goal = np.linalg.norm([turtlebot_x_previous - self.target_x, turtlebot_y_previous - self.target_y])

		distance_reward = previous_distance_turtlebot_goal - current_distance_turtlebot_goal

		# make collision reward
		done = 0
		laser_crashed_reward, done = self.turtlebot_is_crashed(laser_values, range_limit=0.25)
		laser_reward = sum(normalized_laser_values) - 24.0
		collison_reward = laser_crashed_reward + laser_reward

		# make velocity punish reward
		angular_punish_reward = 0.0
		linear_punish_reward = 0.0

		if angular_z > 0.8 or angular_z < -0.8:
			self.angular_punish_reward = -1.0

		if linear_x < 0.2:
			self.linear_punish_reward = -2.0


		# make arrive reward
		arrive_reward = 0.0
		if current_distance_turtlebot_goal < 1:
			print("REACH GOAL")
			done = 1
			arrive_reward = 100
			self.reset()
			time.sleep(1)


		# total reward
		reward = distance_reward * (5/time_step) *1.2*7 + arrive_reward + collison_reward + angular_punish_reward + linear_punish_reward

		print(f"TOTAL REWARD: {reward:.2f}, distance_reward: {(distance_reward*(5/time_step)*1.2*7):.2f},  arrive_reward: {arrive_reward}, collision_reward: {collison_reward:.2f}, velocity_reward: {(angular_punish_reward+linear_punish_reward):.2f}, done", done)

		return  state, reward, done


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
		self.subscription = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)

	def lidar_callback(self, li_data):
		global lidar_data
		lidar_data = li_data


class CmdVelSubscriber(Node):
	def __init__(self):
		super().__init__('cmd_vel_subscriber')
		self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

	def cmd_vel_callback(self, vel_data):
		global cmd_vel_data
		cmd_vel_data = vel_data


class ExperienceBuffer:
	def __init__(self):
		self.memory = deque(maxlen=40000)

	
	def remember(self, cur_state, action, reward, new_state, done):
		cur_state = cur_state.reshape(28)
		action = action.reshape(2)

		self.array_reward = np.array(reward)
		self.array_reward = self.array_reward.reshape(1)

		new_state = new_state.reshape(28)

		done = np.array(done)
		done = done.reshape(1)

		self.memory_pack = np.concatenate((cur_state, action))
		self.memory_pack = np.concatenate((self.memory_pack, self.array_reward))
		self.memory_pack = np.concatenate((self.memory_pack, new_state))
		self.memory_pack = np.concatenate((self.memory_pack, done))

		self.memory.append(self.memory_pack)

		print(f"self.memory length is: {len(self.memory)}")

		if len(self.memory)%10 == 0:
			sio.savemat(
				file_name="hanh_lang_data.mat",
				mdict={"data":self.memory},
				appendmat=True,
				format="5",
				long_field_names=False,
				do_compression=False,
				oned_as="row"
			)


if __name__ == '__main__':
	rclpy.init(args=None)

	buffer = ExperienceBuffer()

	num_trials = 1000
	trial_len = 500
	record_human = 1

	env = GazeboEnv()
	odom_subscriber = OdomSubscriber()
	lidar_subscriber = LidarSubscriber()
	cmd_vel_subscriber = CmdVelSubscriber()
	
	executor = MultiThreadedExecutor()
	executor.add_node(env)
	executor.add_node(odom_subscriber)
	executor.add_node(lidar_subscriber)
	executor.add_node(cmd_vel_subscriber)

	executor_thread = threading.Thread(target=executor.spin, daemon=True)
	executor_thread.start()

	rate = env.create_rate(100)

	current_state = env.reset()

	try:
		while rclpy.ok():
			if record_human == 1:
				for i in range(num_trials):
					print(f"============================== trial: {i} ==============================")
					for j in range(trial_len):
						print(f"step: {j+1} ")
						current_state = env.read_state()
						current_state = current_state.reshape((1, env.observation_space.shape[0]))
						action = env.read_action()
						action = action.reshape((1, env.action_space.shape[0]))
						new_state, reward, done = env.read_game_step(
							time_step=0.1, 
							linear_x=action[0][0], 
							angular_z=action[0][1]
						)

						buffer.remember(current_state, action, reward, new_state, done)

						time.sleep(1)

						if done == 1:
							break

						if j == (trial_len - 1):
							print("TIME OUT!")
							break

						if len(buffer.memory) >= 1000:
							break

					if len(buffer.memory) >= 1000:
							break
					
				print("CONGRATULATION IT DONE!")
				break      

	except KeyboardInterrupt:
		pass

	rclpy.shutdown()

