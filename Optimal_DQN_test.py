# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import gym
import Environments.FL
import numpy as np
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
# TODO: tf.keras....
# import tensorflow as tf
from tensorflow.keras.losses import Huber
from keras.models import clone_model
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical

# from Agents.VQDQL import VQDQL
from Agents.DQN import DQN
from Framework.Agent import Agent
from Framework.Configuration import Configuration
from Framework.utils import get_num_states
import matplotlib.pyplot as plt


# Hallo Max


class DQN(Agent):
    def __init__(self, environment, configuration, name='', verbosity_level=10, debug=False):
        self.Y = np.array([
            [26, 27, 27, 26], [26, -10, 28, 27],
            [27, 29, 27, 28], [28, -10, 27, 27],
            [27, 28, -10, 26], [0, 0, 0, 0],
            [-10, 30, -10, 28], [0, 0, 0, 0],
            [28, -10, 29, 27], [28, 30, 30, -10],
            [29, 31, -10, 29], [0, 0, 0, 0],
            [0, 0, 0, 0], [-10, 30, 31, 29],
            [30, 31, 32, 30], [0, 0, 0, 0]
        ])
        self.D = []  # Memory
        self.num_states = get_num_states(environment)
        self.model = configuration.model
        self.lr = configuration.training_params.pop(0)
        self.lr_cooling = configuration.cooling_scheme.pop(0)
        # TODO: lr_cooling is not yet used.
        self.model.compile(loss=Huber(), optimizer=Adam())  # TODO : optimizer in Configuration
        self.target_model = clone_model(self.model)
        self.memory_size = configuration.memory_size

        self.train_counter = 0

        self.init_memory(environment)

        # init super lastly, because it starts the training
        super().__init__(environment=environment, verbosity_level=verbosity_level, debug=debug, name=name,
                         configuration=configuration)

    def init_memory(self, environment):
        while len(self.D) < self.memory_size:
            s = environment.reset()
            done = False
            while not done:
                a = environment.action_space.sample()
                s1, r, done = environment.step(a)
                self.D.append((to_categorical(s, num_classes=self.num_states), a, r,
                               to_categorical(s1, num_classes=self.num_states), done))
                s = s1
                if len(self.D) == self.memory_size:
                    break

    def loss_function(self, theta, t):
        return

    def train1(self, train_params, batch_size, verb=True):
        gamma, epsilon = train_params
        epsilon = .7

        state = self.environment.reset()
        total_reward = 0

        # sample from policy
        done = False
        while not done:
            a = self.sample_action(state, [epsilon])
            # if state in [2, 6, 10]:
            #   a = 1  # Down
            #   #else:  # state in [0, 1 ,14]
            # a = 2 # Right
            new_state, r, done, = self.environment.step(a)
            total_reward += r
            self.D.pop(0)
            self.D.append((to_categorical(state, num_classes=self.num_states), a, r,
                           to_categorical(new_state, num_classes=self.num_states), done))
            state = new_state

        # create minibatch for training
        for _ in range(1):
            mB_ind = np.random.choice(range(self.memory_size), size=batch_size, replace=True)
            mB = np.array(self.D)[mB_ind]
            y = np.zeros((batch_size, self.environment.action_space.n))
            x = np.zeros((batch_size, self.num_states))
            batch_losses = []
            for j in range(batch_size):
                x[j] = mB[j][0]
                y[j] = self.model.predict(x[j].reshape(1, self.num_states))[0]
                y[j][mB[j][1]] = mB[j][2] * mB[j][-1] or mB[j][2] + gamma * max(
                    self.target_model.predict(mB[j][3].reshape(1, self.num_states))[0])

                if verb and j < 5:
                    print('state: ', np.argmax(x[j]), 'action: ', mB[j][1], 'target: ',
                          self.Y[np.argmax(x[j])].max() - 12,
                          'actual: ', mB[j][2] * mB[j][-1] or mB[j][2] + gamma * max(
                            self.target_model.predict(mB[j][3].reshape(1, self.num_states))[0]),
                          'predict: ', self.model.predict(mB[j][3].reshape(1, self.num_states))[0])

            # train
            batch_loss = self.model.train_on_batch(x, y)
            batch_losses.append(batch_loss)

        if verb:
            print('train it:', (self.train_counter+1), ' Return: ', total_reward, ' Loss: ', np.mean(batch_losses))

        # replace target model
        if self.train_counter % self.configuration.target_replacement == 0:
            self.target_model.set_weights(self.model.get_weights())

        self.train_counter = self.train_counter + 1
        return total_reward

    def get_action(self, state):
        return np.argmax(
            self.model.predict(to_categorical(state, num_classes=self.num_states).reshape(1, self.num_states)))

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.action_space.sample()
        else:
            a = self.get_action(state)
        return a

    def evaluate(self, nb_iterations):
        total_rewards = np.zeros(nb_iterations)
        for i in range(nb_iterations):
            print('New Iteration')
            s = self.environment.reset()
            done = False
            total_reward = 0
            while not done:
                a = self.get_action(s)
                print('a', a)
                s1, r, done = self.environment.step(a)
                print('s1', s1)
                total_reward += r
                s = s1
            total_rewards[i] = total_reward
        return total_rewards



env_name = 'FL-v1'
env = gym.make(env_name)
num_states = get_num_states(env)
model = Sequential()
model.add(Dense(8, input_shape=(16,), activation='tanh'))
#model.add(BatchNormalization())  # TODO: Add in original DQN if this helps
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

iterations = 20000
# alpha, gamma, epsilon
training_params = [.1, .99, .9]
batch_size = 100
confDQN = Configuration(nb_iterations=iterations, training_params=training_params,
                        cooling_scheme=[lambda x, iter: x, lambda x, iter: x,
                                        lambda x, iter: 1 - (iter / iterations)], batch_size=batch_size,
                        plot_training=False, memory_size=300,
                        average=int(batch_size / 100))
# TODO: average should depend on iterations not batch_size?? -- discuss
confDQN.model = model
confDQN.target_replacement = 10

agent2 = DQN(env, debug=True, configuration=confDQN, verbosity_level=1000)
plt.plot(agent2.returns)
plt.show()

total_rewards = agent2.evaluate(1)
plt.plot(total_rewards)
plt.show()
print(np.mean(total_rewards))



