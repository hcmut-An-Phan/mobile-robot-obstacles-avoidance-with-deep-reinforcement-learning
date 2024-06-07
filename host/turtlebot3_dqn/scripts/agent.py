import os
import random
import json
import numpy as np
import keras

from collections import deque

from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation

class DQNAgent():
    def __init__(self, state_size, action_size):

        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/scripts', 'turtlebot3_dqn/save_models/stage_1_')

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.replay_buffer = deque(maxlen=1000000)

        self.model = self.build_model("main_network")
        self.target_model = self.build_model("target_network")
    
        self.update_target_model()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')


    def build_model(self, model_name):
        model = Sequential(name=model_name)
        drop_out = 0.2

        model.add(keras.Input(shape=(self.state_size,)))
        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(drop_out))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))

        # model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        # use adam to improve training time, huber for stability
        loss_function = keras.losses.Huber()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        model.summary()

        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_Q_value(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)
        
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)), verbose=0)
            self.q_value = q_value
            return np.argmax(q_value[0])


    def append_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))


    def train_model(self, target=False):
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)), verbose=0)
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)), verbose=0)

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)), verbose=0)

            next_q_value = self.get_Q_value(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)