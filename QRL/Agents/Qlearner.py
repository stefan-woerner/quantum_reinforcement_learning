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
import numpy as np

from QRL.Framework.Agent import RLAgent


class Qlearner(RLAgent):
    def __init__(self, environment, configuration, name='', debug=False):

        # init super lastly, because it starts the training
        super().__init__(environment=environment, debug=debug, name=name, configuration=configuration)

    def get_num_states(self):
        if type(self.environment.observation_space) != gym.spaces.tuple.Tuple:
            return self.environment.observation_space.n
        dim_list = []
        for sp in self.environment.observation_space:
            dim_list.append(sp.n)
        dim_list = np.array(dim_list)
        return dim_list.prod()

    def get_state_hash(self, state):
        if type(self.environment.observation_space) != gym.spaces.tuple.Tuple:
            return state
        dim_list = []
        for sp in self.environment.observation_space:
            dim_list.append(sp.n)
        dim_list = np.array(dim_list)
        h = 0
        for i in range(len(dim_list) - 1):
            h += state[i] * dim_list[i + 1:].prod()
        h += state[-1]
        return h

    def train1(self, train_params, batch_size):
        returnlist = []
        num_actions = self.environment.action_space.n
        num_states = self.get_num_states(self.environment)
        Q = np.zeros((num_states, num_actions))
        for it in range(self.configuration.iterations):
            state = self.environment.reset()
            R = 0

            done = False
            while not done:
                state_h = self.get_state_hash(self.environment, state)
                if np.random.random() < self.configuration.epsilon:
                    action = self.environment.action_space.sample()
                else:
                    candidates = np.where(Q[state_h] == np.max(Q[state_h]))[0]
                    action = np.random.choice(candidates)
                statep, reward, done, info = self.environment.step(action)
                if reward == 0:
                    if done:
                        reward = -0.2
                    else:
                        reward = -0.01
                else:
                    reward = 1.0
                R *= self.configuration.gamma
                R += reward
                statep_h = self.get_state_hash(self.environment, statep)
                Q[state_h, action] += self.configuration.alpha * (
                            reward + self.configuration.gamma * Q[statep_h].max() - Q[state_h, action])
                state = statep

            returnlist.append(R)
            if it % 10000 == 0:
                print('Iteration %d, Reward: %d' % (it, R))

        return returnlist, Q

    def get_action(self, state):
        return np.argmax(self.get_qvalues([state], self.theta)[0])

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.sample_actions_space()
        else:
            a = self.get_action(state)
        return a
