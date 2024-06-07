#!/usr/bin/env python3
import os
import json
import random
import time
import math
import threading
import rclpy
import numpy as np

from math import pi

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import  SpawnEntity, SetEntityState
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from squaternion import Quaternion
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from agent import DQNAgent

TIME_DELTA = 0.1
GOAL_REACHED_DIST = 0.2
COLLISION_DIST = 0.2
EPISODES = 3000

last_odom = None
lidar_data = None

class GazeboEnv(Node):
    def __init__(self, action_size):
        super().__init__('gazebo_env')
        self.action_size = action_size

        self.goal_distance = 0.0          # absolute distance
        self.get_goal_box = False         # is robot get the goal ?
        self.init_goal = True             # is the first time set goal position?

        self.odom_x = 0.0                 # position x of robot
        self.odom_y = 0.0                 # position y of robot

        self.goal_x = 0.0                 # position x of the goal
        self.goal_y = 0.0                 # position y of the goal

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
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)                          # control robot
        
        self.pub_get_action = self.create_publisher(Float32MultiArray, 'get_action', 5)     # publish robot's action
        self.pub_result = self.create_publisher(Float32MultiArray, 'result', 5) 
                    # publish robot's reward
        self.set_state = self.create_client(SetEntityState, "/plugin/set_entity_state")

        self.unpause = self.create_client(Empty, "/unpause_physics")                        # unpause gazebo
        self.pause = self.create_client(Empty, "/pause_physics")                            # pause gazebo
        self.reset_proxy = self.create_client(Empty, "/reset_world")                        # reset gazebo

        # setup attributes for respawn new goal
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('scripts', 
                                                'gazebo_models/goal_box/model.sdf') 
        self.f = open(self.modelPath, 'r')
        self.entity = self.f.read()                          # goal_box model 
        self.goal_position = Pose()                         # goal position
        self.init_goal_x = 1.3                            
        self.init_goal_y = 1.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.entity_name = 'goal'
        self.last_index = 0
        self.index = 0
        self.spawn_entity_proxy = self.create_client(SpawnEntity, '/spawn_entity') 

    
    def spawn_init_goal(self):
        while not self.spawn_entity_proxy.wait_for_service(timeout_sec=1.0):
            print('Spawn_goal: service /spawn_entity not available, waiting again...')
        request = SpawnEntity.Request()
        request.name = self.entity_name
        request.xml = self.entity
        request.robot_namespace = 'robotos_name_space'
        request.initial_pose = self.goal_position
        request.reference_frame = 'world'
        future = self.spawn_entity_proxy.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result().success:
            print(f"Init Goal position : {self.goal_position.position.x}, {self.goal_position.position.y}")
        else:
            print("Spawn Goal failed")


    def set_new_goal(self, init_goal=False):
        if init_goal:
            self.spawn_init_goal()
        else:
            print("Setting new goal posittion. . .")
            # goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1.0, -1.9, 0.5, 2.0, 0.5, 0.0, -0.1, -2.0]
            # goal_y_list = [0.0, -0.5, -1.9, 1.5, -0.9, 1.0, 1.1, -1.5, 1.5, 1.8, -1.0, 1.6, -0.8]

            # MASE
            goal_x_list = [-4.0, -4.0, -4.0, -2.0, -2.0, 0.0,  0.0,  1.0, 2.0,  2.0, 4.0, 4.0,  4.0]
            goal_y_list = [ 4.0,  0.0, -4.0,  3.0, -3.0, 2.0, -2.0, -4.0, 4.0, -2.0, 3.0, 0.0, -3]

            position_ok = False
            while not position_ok:
                self.index = random.randrange(0, 13)
                if self.last_index != self.index:
                    position_ok = True
                    self.last_index = self.index

            self.goal_position.position.x = goal_x_list[self.index]
            self.goal_position.position.y = goal_y_list[self.index]

            goal_state = EntityState()
            goal_state.name = self.entity_name
            goal_state.pose.position.x = self.goal_position.position.x
            goal_state.pose.position.y = self.goal_position.position.y
            goal_state.pose.position.z = 0.0
            goal_state.pose.orientation.x = 0.0
            goal_state.pose.orientation.y = 0.0
            goal_state.pose.orientation.z = 0.0
            goal_state.pose.orientation.w = 1.0
            request = SetEntityState.Request()
            request._state = goal_state
            while not self.set_state.wait_for_service(timeout_sec=1.0):
                print("Set state : service /plugin/set_state not available, waiting again..")
            future = self.set_state.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            if future.result().success:
                print(f"New Goal position : {self.goal_position.position.x}, {self.goal_position.position.y}")
            else:
                print("Set new goal failed!")
        return  self.goal_position.position.x,  self.goal_position.position.y
        

    def get_goal_distance(self):
        goal_distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        return goal_distance


    def get_state(self, scan):
        scan_range = []
        done = False

        # process data from lidar
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('inf'):
                scan_range.append(3.5) # max_range
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)

        # Detect a collision from laser data
        if COLLISION_DIST > min(scan_range) > 0:
            done = True

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        yaw = round(euler[2], 4)

        # Calculate the relative angle between the robots heading and heading toward the goal
        goal_angle = math.atan2(self.goal_y - self.odom_y, self.goal_x - self.odom_x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2*pi
        elif heading < -pi:
            heading += 2*pi
        heading = round(heading, 2) 


        # Calculate distance to the goal from the robot
        current_distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # Detect if the goal has been reached and give a large positive reward
        if current_distance < GOAL_REACHED_DIST:
            done = True
            self.get_goal_box = True

        state = scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle]
        return state, done


    def set_reward(self, state, done, action):
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        for i in range(5):
            angle = -pi/4 + heading + i * pi/8 + pi/2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2*pi) / pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)

        if obstacle_min_range < 0.5:
            ob_reward = -5
        else:
            ob_reward = 0

        reward = round(yaw_reward[action] * 5, 2) * distance_rate + ob_reward

        if done and not self.get_goal_box:
            print("Collision is detected!")
            reward = -500

        if self.get_goal_box:
            print("GOAL is reached!")
            reward = 1000
            self.get_goal_box = False

        return reward


    def step(self, action):
        max_angular_vel = 1.5
        angular_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
        
        # Publish robot action, action spaces = [0 1 2 3 4]
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = angular_vel
        self.vel_pub.publish(vel_cmd)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            print('Unpause: service /unpause_physics not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/Unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            print('Pause: service /pause_physics not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("Pause_physics service call failed")

        state, done = self.get_state(lidar_data)
        reward = self.set_reward(state, done, action)

        return np.asarray(state), reward, done


    def reset(self):
        print("Reseting Gazebo environment. . .")
        # init random goal in empty space in environment
        if self.init_goal:
            self.goal_x, self.goal_y = self.set_new_goal(self.init_goal)
            self.init_goal = False
            x = 0.0
            y = 0.0
        else:
            self.goal_x, self.goal_y = self.set_new_goal(self.init_goal)
            # x = round(np.random.uniform(-1.5, 1.5), 2)
            # y = round(np.random.uniform(-1.5, 1.5), 2)
            x = 0.0
            y = 0.0

        # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service("/gazebo/reset_world")
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            print("Reset : service /reset_world not available, waiting again..")
        try:
            self.reset_proxy.call_async(Empty.Request())
            print("Reset Gazebo environment succeed")
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # CHANGE THIS -> random inital state
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, 0.0)
        
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


        self.odom_x = robot_state.pose.position.x
        self.odom_y = robot_state.pose.position.y

        self.goal_distance = self.get_goal_distance()
        print(f"Reset absolute goal distance: {self.goal_distance}")

        # randomly scatter boxes in the environment
        # self.random_box()

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            print('Unpause: service /unpause_physics not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            print('Pause: service /pause_physics not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        # get reset state
        state, done = self.get_state(lidar_data)
        return np.asarray(state)


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


if __name__ == '__main__':
    rclpy.init(args=None)

    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 28
    action_size = 5

    agent = DQNAgent(state_size, action_size)

    env = GazeboEnv(action_size)
    odom_subscriber = OdomSubscriber()
    lidar_subscriber = LidarSubscriber()
    
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor.add_node(odom_subscriber)
    executor.add_node(lidar_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = env.create_rate(2)

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()

    try:
        while rclpy.ok():
            for ep in range(agent.load_episode + 1, EPISODES):
                print(f"================================== Episode: {ep} ==================================")
                done = False
                state = env.reset()
                score = 0.0
                for t in range(agent.episode_step):
                    
                    action = agent.get_action(state)
                    next_state, reward, done = env.step(action)
                    agent.append_memory(state, action, reward, next_state, done)

                    # print(f"Step: {t},  reward: {reward}")

                    score += reward
                    state = next_state
                    
                    get_action.data = [float(action), float(score), float(reward)]
                    env.pub_get_action.publish(get_action)

                    if t%5 == 0 and len(agent.replay_buffer) >= agent.train_start:
                        start_time = time.time()
                        if global_step <= agent.target_update:
                            agent.train_model()
                        else:
                            agent.train_model(True)
                        end_time = time.time()
                        print(f"step: {t}, train time is: {end_time - start_time}")


                    if ep % 10 == 0:
                        agent.model.save(agent.dirPath + str(ep) + '.keras', overwrite=True)
                        # save epsilon
                        with open(agent.dirPath + str(ep) + '.json', 'w') as outfile:
                            json.dump(param_dictionary, outfile)

                    if t >= 500:
                        print('Time out!')
                        done = True

                    if done:
                        result.data = [float(score), float(np.max(agent.q_value))]
                        env.pub_result.publish(result)
                        agent.update_target_model()
                        scores.append(score)
                        episodes.append(ep)
                        m, s = divmod(int(time.time() - start_time), 60)
                        h, m = divmod(m, 60)

                        print(f"Episode: {ep} score: {score:.2f} memory: {len(agent.replay_buffer)} epsilon: {agent.epsilon} time: {h}:{m:02d}:{s:02d}")

                        param_keys = ['epsilon']
                        param_values = [agent.epsilon]
                        param_dictionary = dict(zip(param_keys, param_values))
                        break
                        
                    global_step += 1
                    if global_step % agent.target_update == 0:
                        print("UPDATE TARGET NETWORK")
                
                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
