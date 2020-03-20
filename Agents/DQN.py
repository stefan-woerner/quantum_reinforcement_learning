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

from Framework.utils import get_num_states, get_state_hash
import numpy as np
from Framework.Agent import Agent
from keras.models import clone_model
from keras.utils import to_categorical
import keras.backend as K

# Hallo Max

class DQN(Agent):
    def __init__(self, environment, configuration, name='', debug=False):
        self.D = []  # Memory
        self.num_states = get_num_states(environment)
        self.model = configuration.model
        self.model.compile(loss='mean_squared_error', optimizer='sgd')
        self.target_model = clone_model(self.model)
        self.memory_size = configuration.memory_size

        self.train_counter = 0

        self.init_memory(environment)

        # init super lastly, because it starts the training
        super().__init__(environment=environment, debug=debug, name=name, configuration=configuration)

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

    def train1(self, train_params, batch_size):
        gamma, epsilon = train_params

        state = self.environment.reset()
        total_reward = 0

        # sample from policy
        done = False
        while not done:
            a = self.sample_action(state, [epsilon])
            new_state, r, done, = self.environment.step(a)
            total_reward += r
            self.D.pop(0)
            self.D.append((to_categorical(state, num_classes=self.num_states), a, r,
                           to_categorical(new_state, num_classes=self.num_states), done))
            state = new_state

        # create minibatch for training
        mB_ind = np.random.choice(range(self.memory_size), size=batch_size, replace=False)
        mB = np.array(self.D)[mB_ind]
        y = np.zeros((batch_size, self.environment.action_space.n))
        x = np.zeros((batch_size, self.num_states))
        for j in range(batch_size):
            x[j] = mB[j][0]
            y[j] = self.model.predict(x[j].reshape(1, self.num_states))[0]
            y[j][mB[j][1]] = mB[j][2] * mB[j][-1] or mB[j][2] + gamma * max(self.target_model.predict(mB[j][3].reshape(1, self.num_states))[0])


        # train
        print('train it:', self.train_counter)
        self.model.train_on_batch(x, y)

        # replace target model
        if self.train_counter % self.configuration.target_replacement == 0:
            self.target_model = clone_model(self.model)

        self.train_counter = self.train_counter + 1
        return total_reward

    def get_action(self, state):
        return np.argmax(self.model.predict(to_categorical(state, num_classes=self.num_states).reshape(1, self.num_states)))

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.action_space.sample()
        else:
            a = self.get_action(state)
        return a
