import time
import threading
import rclpy
import numpy as np
import tensorflow as tf
import keras
import scipy.io as sio

from math import atan2, pi
from squaternion import Quaternion

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from keras.layers import Dense
from keras.initializers import glorot_normal

from common_definition import NUM_MISSION, MODEL_PATH, NUM_MAX_STEP, SAVE_REWARD, TEST_MAZE, VEL_THRESH_HOLD, LINEAR_THRESH_HOLD, ANGUALR_THRESH_HOLD

KERNEL_INITIALIZER = glorot_normal()

last_odom = None
lidar_data = None
num_collisions = 0
num_time_out = 0

def ActorNetwork(num_states=28, num_actions=2, action_high=1):
		"""
		Get Actor Network with the given parameters.

		Args:
			num_states: number of states in the NN
			num_actions: number of actions in the NN
			action_high: the top value of the action

		Returns:
			the Keras Model
		"""
		# Initialize weights between -3e-3 and 3-e3
		last_init = tf.random_normal_initializer(stddev=0.0005)

		inputs = keras.Input(shape=(num_states,), dtype=tf.float32)
		h1 = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(inputs)
		h2 = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(h1)
		h3 = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(h2)
		outputs = Dense(units=num_actions, activation='tanh', kernel_initializer=last_init)(h3)*action_high

		model = keras.Model(inputs, outputs)
		model.summary()
		return model


class RealEnv(Node):
	def __init__(self):
		super().__init__('real_env')

		self.state_num = 28
		self.action_num = 2
		self.observation_space = np.empty(self.state_num)
		self.action_space = np.empty(self.action_num)

		self.target_x = 0.0                 
		self.target_y = 0.0 

		# cmd_vel publisher
		self.vel_pub = self.create_publisher(Twist, "/diff_drive_controller/cmd_vel_unstamped", 10) 
	

	def turtlebot_is_crashed(self, laser_values, range_limit):
		global num_collisions
		laser_crashed_reward = 0.0
		done = 0

		if 0.12 < min(laser_values) < 2*range_limit:
			laser_crashed_reward = -80.0

		if 0.12 < min(laser_values) < range_limit:
			print("======================================================================================================")
			print("                                                                                   COLLISION DETECED  ")
			print("======================================================================================================")
			done = 1
			laser_crashed_reward = -200.0
			num_collisions += 1

		return laser_crashed_reward, done
	

	def step(self, time_step, angular_z, linear_x):
		global lidar_data
		global last_odom 

		vel_cmd = Twist()
		if VEL_THRESH_HOLD:
			vel_cmd.linear.x = float((linear_x+1)/2*LINEAR_THRESH_HOLD)
			vel_cmd.angular.z = float(angular_z*ANGUALR_THRESH_HOLD)
		else:
			vel_cmd.linear.x = float((linear_x+1)/2*0.26)
			if float((linear_x+1)/2*0.26) > 0.25:
				vel_cmd.angular.z =0.0
			else:
				vel_cmd.angular.z = float(angular_z)

		turtlebot_x_previous = last_odom.pose.pose.position.x
		turtlebot_y_previous = last_odom.pose.pose.position.y

		self.vel_pub.publish(vel_cmd)

		# propagate state for time_step seconds
		time.sleep(time_step)

		extra_time = 0.0
		if min(lidar_data) < 0.12:
			extra_start = time.time()
			while min(lidar_data) < 0.12:
				print('Lidar Data is  wrong ...........................................................................')
			extra_end = time.time()
			extra_time = extra_end - extra_start
			print(f"WAIT EXTRA TIME FOR LIDAR DATA: {extra_time}")


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
		for i in range(len(lidar_data)):
			if lidar_data[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data[i]):
				laser_values.append(0.0)
			else:
				laser_values.append(lidar_data[i])

		normalized_laser_values= [(x)/3.5 for x in laser_values]

		# prepare state
		current_distance_turtlebot_goal = np.linalg.norm([turtlebot_x - self.target_x, turtlebot_y - self.target_y])

		state = np.append(normalized_laser_values, current_distance_turtlebot_goal)
		state = np.append(state, angle_diff)
		state = np.append(state, float(last_odom.twist.twist.linear.x)) 
		state = np.append(state, float(last_odom.twist.twist.angular.z))

		# state = np.append(state, float((linear_x+1)/2*0.26)) 
		# state = np.append(state, float(angular_z))


		# make distance reward
		previous_distance_turtlebot_goal = np.linalg.norm([turtlebot_x_previous - self.target_x, turtlebot_y_previous - self.target_y])

		distance_reward = previous_distance_turtlebot_goal - current_distance_turtlebot_goal

		# make collision reward
		laser_crashed_reward, done = self.turtlebot_is_crashed(laser_values, range_limit=0.15)
		laser_reward = sum(normalized_laser_values) - 24.0
		collison_reward = laser_crashed_reward + laser_reward

		# make velocity punish reward
		angular_punish_reward = 0.0
		linear_punish_reward = 0.0

		if angular_z > 0.8 or angular_z < -0.8:
			self.angular_punish_reward = -1.0

		if linear_x < -0.6:
			self.linear_punish_reward = -2.0

		# make arrive reward
		arrive_reward = 0.0
		if current_distance_turtlebot_goal < 1.0:
			print("======================================================================================================")
			print("=                                                                                       REACH GOAL!  = ")
			print("======================================================================================================")
			arrive_reward = 100
			done = 1

			# stop robot
			vel_cmd = Twist()
			vel_cmd.linear.x = 0.0
			vel_cmd.angular.z = 0.0
			self.vel_pub.publish(vel_cmd)


		# total reward
		reward = distance_reward * (5/(time_step+extra_time))*7 + arrive_reward + collison_reward + angular_punish_reward + linear_punish_reward

		return  state, reward, done
	

	def reset(self, x_targ, y_targ):
		self.target_x = x_targ
		self.target_y = y_targ

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
		while(lidar_data is None):
			print("wait for lidar . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .")
		# process data from lidar
		for i in range(len(lidar_data)):
			if lidar_data[i] == float('inf'):
				laser_values.append(3.5) 
			elif np.isnan(lidar_data[i]):
				laser_values.append(0.0)
			else:
				laser_values.append(lidar_data[i])

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
		self.subscription = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)

	def odom_callback(self, od_data):
		global last_odom
		last_odom = od_data


class LidarSubscriber(Node):
	def __init__(self):
		super().__init__('lidar_subscriber')
		self.subscription = self.create_subscription(LaserScan, '/scan_filtered', self.lidar_callback, 10)

	def lidar_callback(self, li_data):
		global lidar_data
		"""
		[268 276 284 292 300 308 316 324 332 340 348 356 4  12 20 28 36 44 52 60 68 76 84 92]
		[0   1   2   3   4   5   6   7   8   9   10  11  12 13 14 15 16 17 18 19 20 21 22 23]
		"""
		start_angle = 268
		current_angle = start_angle
		laser_ranges = []

		for i in range(24):
			if li_data.ranges[current_angle] == float('-inf'):
				laser_ranges.append(3.5)
				#=========================================================================================================================
				# tmp = current_angle

				# for i in range(1, 6):
				# 	if i == 5:
				# 		laser_ranges.append(3.5)
				# 		break
					
				# 	if (tmp + i) == 360:
				# 		if li_data.ranges[0] == float('-inf'):
				# 			if li_data.ranges[tmp-i] == float('-inf'):
				# 				continue
				# 			else:
				# 				laser_ranges.append(li_data.ranges[tmp-i])
				# 				break
				# 		else:
				# 			laser_ranges.append(li_data.ranges[0])
				# 			break
				# 	else:	
				# 		if li_data.ranges[tmp+i] == float('-inf'):
				# 			if li_data.ranges[tmp-i] == float('-inf'):
				# 				continue
				# 			else:
				# 				laser_ranges.append(li_data.ranges[tmp-i])
				# 				break
				# 		else:
				# 			laser_ranges.append(li_data.ranges[tmp+i])
				# 			break
			else:
				laser_ranges.append(li_data.ranges[current_angle])

			current_angle += 8

			if current_angle >= 360:
				current_angle -= 360

		lidar_data = laser_ranges


if __name__ == "__main__":
	rclpy.init(args=None)

	env = RealEnv()
	odom_subscriber = OdomSubscriber()
	lidar_subscriber = LidarSubscriber()

	actor = ActorNetwork()
	actor.load_weights(MODEL_PATH + '.weights.h5')	

	executor = MultiThreadedExecutor()
	executor.add_node(env)
	executor.add_node(odom_subscriber)
	executor.add_node(lidar_subscriber)

	executor_thread = threading.Thread(target=executor.spin, daemon=True)
	executor_thread.start()

	rate = env.create_rate(100)

	step_reward = [0,0]

	step = 0
	start_train_time = time.time()

	print("======================================================================================================")
	print("                                      TEST MODEL " + MODEL_PATH + "                                   ")
	print(f"                                      NUM_MISSION:  {NUM_MISSION}                                    ")
	print(f"                                      MAX STEP:  {NUM_MAX_STEP}                                      ")
	if SAVE_REWARD:
		print("                                      SAVE REWARD:  TRUE                                          ")
	else:
		print("                                      SAVE REWARD:  FALSE                                         ")
	if VEL_THRESH_HOLD:
		print("                                      VEL_THRESH_HOLD:  TRUE                                      ")
	else:
		print("                                      VEL_THRESH_HOLD:  FALSE                                     ")
	if TEST_MAZE:
		print("                                      TEST ON MAZE                                                ")
	else:  
		print("                                      TEST ON CORRIDOR                                            ")
	print("======================================================================================================")

	try:
		while rclpy.ok():
			# get input from keyboard
			try:
				# Get input from the user
				input1 = input("Enter the x target value: ")
				input2 = input("Enter the y target value: ")
				
				# Convert input to double
				number1 = float(input1)
				number2 = float(input2)
			except ValueError:
				print("Invalid input. Please enter valid numbers.")

			current_state = env.reset(x_targ=number1, y_targ=number2)
			# print("reset state: ", current_state)
			total_reward = 0
			done = 0
			t = 0

			while done == 0:
					action = actor(tf.expand_dims(current_state, 0))[0].numpy()
					new_state, reward, done = env.step(time_step=0.1, angular_z=action[0], linear_x=action[1])

					if VEL_THRESH_HOLD:
						print(f"step: {t+1}, action is: linear: {((action[1] + 1)/2*LINEAR_THRESH_HOLD):.2f}, angular: {(action[0]*ANGUALR_THRESH_HOLD):.2f}, reward: {reward:.2f}")
					else:
						print(f"step: {t+1}, action is: linear: {((action[1] + 1)/2*0.26):.2f}, angular: {action[0]:.2f}, reward: {reward:.2f}")
					
					total_reward += reward
					step += 1

					if SAVE_REWARD:
						step_reward = np.append(step_reward, [step, reward])
						sio.savemat(MODEL_PATH + 'step_reward.mat', {'data':step_reward},True,'5', False, False,'row')

					current_state = new_state

					if done == 1:
						m, s = divmod(int(time.time() - start_train_time), 60)
						h, m = divmod(m, 60)
						print(f"Finish, score: {total_reward:.2f}  total step: {step}, num_collision: {num_collisions}, time: {h}:{m:02d}:{s:02d}")
						break

					t += 1
					if t == NUM_MAX_STEP:
						print("======================================================================================================")
						print("                                                                                             TIME OUT ")
						print("======================================================================================================")
						num_time_out += 1
						break

	except KeyboardInterrupt:
		pass

	rclpy.shutdown()

