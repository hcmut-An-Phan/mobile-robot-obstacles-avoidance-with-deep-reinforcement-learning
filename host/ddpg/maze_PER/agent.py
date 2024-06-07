import logging
import numpy as np
import tensorflow as tf
import keras

import scipy.io as sio

from keras.layers import Dense
from rank_based import RankBasePER
from utils import OUActionNoise

from common_definitions import (
	KERNEL_INITIALIZER, GAMMA, RHO,
	STD_DEV, BUFFER_SIZE, BATCH_SIZE,
	CRITIC_LR, ACTOR_LR, PARTITION_NUM, TOTAL_STEP
)


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
	

def CriticNetwork(num_states=28, num_actions=2, action_high=1):
	"""
	Get Critic Network with the given parameters.

	Args:
		num_states: number of states in the NN
		num_actions: number of actions in the NN
		action_high: the top value of the action

	Returns:
		the Keras Model
	"""
	last_init = tf.random_normal_initializer(stddev=0.00005)

	# state as input
	state_input = keras.Input(shape=(num_states,), dtype=tf.float32)
	state_out = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(state_input)

	# action as input
	action_input = keras.Input(shape=(num_actions,), dtype=tf.float32)
	action_out = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(action_input/action_high)

	# Both are passed through seperate layer before concatenating
	combine = keras.layers.Concatenate(axis=1)([state_out, action_out])
	h2 = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(combine)
	h3 = Dense(units=500, activation='relu', kernel_initializer=KERNEL_INITIALIZER)(h2)
	outputs = Dense(units=1, kernel_initializer=last_init)(h3)

	model = keras.Model([state_input, action_input], outputs)
	model.summary()

	return model


class AgentDDPG:
	def __init__(self, 
		num_states=28, 
		num_actions=2, 
		action_high=1, 
		action_low=-1, 
		gamma=GAMMA, 
		rho=RHO,
		std_dev=STD_DEV):

		self.actor_network = ActorNetwork()
		self.critic_network = CriticNetwork()
		self.actor_target= ActorNetwork()
		self.critic_target = CriticNetwork()

		# Making the weights equal initially
		self.actor_target.set_weights(self.actor_network.get_weights())
		self.critic_target.set_weights(self.critic_network.get_weights())

		self.conf = {
			'size': BUFFER_SIZE,
			'partition_num': PARTITION_NUM,
			'batch_size': BATCH_SIZE,
			'steps': TOTAL_STEP
		}
		self.buffer = RankBasePER(self.conf)

		self.gamma = tf.constant(gamma)
		self.rho = rho
		self.action_high = action_high
		self.action_low = action_low
		self.num_states = num_states
		self.num_actions = num_actions
		self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

		# optimizer
		self.critic_optimizer = keras.optimizers.Adam(CRITIC_LR, amsgrad=True)
		self.actor_optimizer = keras.optimizers.Adam(ACTOR_LR, amsgrad=True)

		# temporary variable for side effects
		self.cur_action = None

	# define update weights with tf.function for improved performance
	@tf.function(
		input_signature=[
			tf.TensorSpec(shape=(None, 28), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 28), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
		])
	def update_weights(self, s, a, r, sn, d, w):
		with tf.GradientTape() as tape:
			# define target
			y = r + self.gamma * (1 - d) * self.critic_target([sn, self.actor_target(sn)])
			# define delta Q*w
			critic_loss = tf.math.reduce_mean(tf.multiply(tf.math.abs(y - self.critic_network([s, a])), w))

		critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
		self.critic_optimizer.apply_gradients(
			zip(critic_grad, self.critic_network.trainable_variables))
		
		with tf.GradientTape() as tape:
			# define the delta mu
			actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
		actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
		self.actor_optimizer.apply_gradients(
			zip(actor_grad, self.actor_network.trainable_variables))
		
		return critic_loss, actor_loss
	

	@staticmethod
	def _update_target(model_target, model_ref, rho=0):
		"""
		Update target's weights with the given model reference

		Args:
			model_target: the target model to be changed
			model_ref: the reference model
			rho: the ratio of the new and old weights
		"""
		model_target.set_weights(
			[
				rho * ref_weight + (1 - rho) * target_weight
				for (target_weight, ref_weight)
				in list(zip(model_target.get_weights(), model_ref.get_weights()))
			]
		)
		

	def load_human_data(self):
		mat_contents = sio.loadmat('human_data.mat')
		data = mat_contents['data']

		for i in range(1000):
			cur_state = data[i][0:28]
			action = data[i][28:30]
			reward = data[i][30]
			new_state = data[i][31:59]
			done = data[i][59]

			cur_state = cur_state.reshape(28)
			action = action.reshape(2)
			action[1] = action[1]/0.26*2 - 1
			new_state = new_state.reshape(28)
			
			self.buffer.store((cur_state, action, reward, new_state, done))


	def act(self, state, _notrandom=True, noise=True):
		"""
		Run action by the actor network

		Args:
			state: the current state
			_notrandom: whether greedy is used
			noise: whether noise is to be added to the result action (this improves exploration)

		Returns:
			the resulting action
		"""
		# print("new action functions")
		# self.cur_action = self.actor_network(state)[0].numpy() 
		# self.cur_action = self.cur_action + self.noise()

		if _notrandom:
			print("not random")
			self.cur_action = self.actor_network(state)[0].numpy() 
		else:
			print("RANDOM")
			self.cur_action = (
				np.random.uniform(self.action_low, self.action_high, self.num_actions)
				+ (self.noise() if noise else 0)
			)

		self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)
		return self.cur_action


	def get_Q_value(self, state, action):
		Q_value = self.critic_network([state, action])[0].numpy()
		return Q_value


	def remember(self, prev_state, action, reward, state, done):
		"""
		Store states, reward, done value to the buffers
		"""
		# record it in the buffer based on its reward
		self.buffer.store((prev_state, action, reward, state, done))


	def learn(self, global_step):
		"""
		Run update for all networks (for training)
		"""

		exps, weights, indices = self.buffer.sample(global_step) 

		s, a, r, sn, d = zip(*exps)
		s = np.array(s)
		a = np.array(a)
		r = np.expand_dims(np.array(r), -1)
		sn = np.array(sn)
		d = np.expand_dims(np.array(d), -1)

		s_batch = tf.convert_to_tensor(s, dtype=tf.float32)
		a_batch = tf.convert_to_tensor(a, dtype=tf.float32)
		sn_batch = tf.convert_to_tensor(sn, dtype=tf.float32)

		next_action = self.actor_target(sn_batch)
		future_reward = self.critic_target([sn_batch, next_action])[0].numpy()
		target = r + self.gamma * future_reward * (1 - d)
		critic_value = self.critic_network([s_batch, a_batch])[0].numpy()
		error = np.abs(target - critic_value)
		self.buffer.update_priority(indices, error)
		self.buffer.rebalance()
		

		c_l, a_l = self.update_weights(tf.convert_to_tensor(s, dtype=tf.float32),
									   tf.convert_to_tensor(a, dtype=tf.float32),
									   tf.convert_to_tensor(r, dtype=tf.float32),
									   tf.convert_to_tensor(sn, dtype=tf.float32),
									   tf.convert_to_tensor(d, dtype=tf.float32),
									   tf.convert_to_tensor(weights, dtype=tf.float32))
									   

		self._update_target(self.actor_target, self.actor_network, self.rho)
		self._update_target(self.critic_target, self.critic_network, self.rho)

		return c_l, a_l


	def save_model(self, num_trials, trial_len):
		self.actor_network.save_weights('actor_model' + '-' +  str(num_trials) + '-' + str(trial_len) + '.weights.h5', overwrite=True)
		self.critic_network.save_weights('critic_model' + '-' +  str(num_trials) + '-' + str(trial_len) + '.weights.h5', overwrite=True)

		self.actor_target.save_weights('actor_target' + '-' +  str(num_trials) + '-' + str(trial_len) + '.weights.h5', overwrite=True)
		self.critic_target.save_weights('critic_target' + '-' +  str(num_trials) + '-' + str(trial_len) + '.weights.h5', overwrite=True)


	def load_weights(self, path):
		"""
		Load weights from path
		"""
		try:
			self.actor_network.load_weights('actor_model-' + path + ".weights.h5")
			self.critic_network.load_weights('critic_model-' + path + ".weights.h5")
			self.critic_target.load_weights('critic_target-' + path + ".weights.h5")
			self.actor_target.load_weights('actor_target-' + path + ".weights.h5")
		except OSError as err:
			logging.warning("Weights files cannot be found, %s", err)


	

	

