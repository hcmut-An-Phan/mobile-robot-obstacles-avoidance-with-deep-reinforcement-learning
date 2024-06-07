import logging
import numpy as np
import tensorflow as tf
import keras

import scipy.io as sio

from keras.layers import Dense
from keras.initializers import glorot_normal

KERNEL_INITIALIZER = glorot_normal()

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





	

	

