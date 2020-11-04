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
from Framework.utils import get_num_states
import seaborn as sns

import matplotlib.pyplot as plt


class DQN(Agent):
    def __init__(self, environment, configuration, name='', debug=False):
        # init super
        Agent.__init__(self, environment=environment, debug=debug, name=name, configuration=configuration)



        self.D = []  # Memory
        self.num_states = get_num_states(environment)
        self.model = configuration.model
        self.target_model = configuration.clone_model(self.model)
        self.memory_size = configuration.memory_size

        self.train_counter = 0
        self.episode = 0

        self.init_memory(environment)



    def init_memory(self, environment):
        while len(self.D) < self.memory_size:
            s = environment.reset()
            done = False
            while not done:
                a = environment.action_space.sample()
                s1, r, done, _ = environment.step(a)
                self.D.append((self.configuration.embedding(s), a, r,
                               self.configuration.embedding(s1), done))
                s = s1
                if len(self.D) == self.memory_size:
                    break

    def save(self, path):
        if path == '' :
            raise FileNotFoundError('Please provide a path')
        else:
            self.model.save_weights(path)


    def load(self, path):
        if path == '':
            raise FileNotFoundError('Please provide a path to the model.')
        elif not self.model:
            raise AssertionError('A model must be specified to load the weights')
        else:
            self.model.load_weights(path)


    def train1(self, train_params, batch_size, verb=True):
        gamma, epsilon = train_params

        state = self.environment.reset()
        total_reward = 0

        self.episode += 1

        # sample from policy
        done = False
        batch_losses = []
        while not done:
            a = self.sample_action(state, [epsilon])
            new_state, r, done,_ = self.environment.step(a)
            total_reward += r
            self.D.pop(0)
            self.D.append((self.configuration.embedding(state), a, r,
                           self.configuration.embedding(new_state), done))

            # create minibatch for training
            # for _ in range(1):
            mB_ind = np.random.choice(range(self.memory_size), size=batch_size, replace=True)
            mB = np.array(self.D)[mB_ind]
            x = np.concatenate(mB[:, 0]).reshape(batch_size, self.num_states)
            y = self.model.predict(x)
            y_target = gamma * np.max(self.target_model.predict(np.concatenate(mB[:, 3]).reshape(batch_size, self.num_states)), axis=1)

            for j in range(batch_size):
                y[j][mB[j][1]] = mB[j][2] * mB[j][-1] or mB[j][2] + y_target[j]

            # train
            batch_loss = self.model.train_on_batch(x, y)
            batch_losses.append(batch_loss)

            # replace target model
            if self.train_counter % self.configuration.target_replacement == 0:
                self.target_model.set_weights(self.model.get_weights())

            self.train_counter = self.train_counter + 1

            # update state
            state = new_state

        if verb:
            print('train it:', self.episode, ' Return: ', total_reward, ' Loss: ', np.mean(batch_losses))

        return total_reward

    def plot_q_values(self, states, shape = (4,3), iteration = 0, camera=None):
        maxi = 0
        matrix = np.zeros((np.prod(shape), np.prod(shape)))
        for i in range(states):
            j, k = int(i / shape[0]), i % shape[0]
            q_val = self.model.predict(self.configuration.embedding(i)[np.newaxis,:])[0]
            if np.max(q_val) > maxi:
                maxi = np.max(q_val)
            matrix[1 + (shape[1] * j), 0 + (shape[1] * k)] = q_val[0]  # left
            matrix[2 + (shape[1] * j), 1 + (shape[1] * k)] = q_val[1]  # down
            matrix[1 + (shape[1] * j), 2 + (shape[1] * k)] = q_val[2]  # right
            matrix[0 + (shape[1] * j), 1 + (shape[1] * k)] = q_val[3]  # up

        ax = sns.heatmap(matrix, vmin=0, vmax=maxi, cmap='YlGnBu')  # "RdBu_r")
        ax.hlines([3, 6, 9], *ax.get_xlim())
        ax.vlines([3, 6, 9], *ax.get_ylim())
        plt.show()

    def get_action(self, state):
        return np.argmax(
            self.model.predict(self.configuration.embedding(state)[np.newaxis,:]))

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.action_space.sample()
        else:
            a = self.get_action(state)
        return a
