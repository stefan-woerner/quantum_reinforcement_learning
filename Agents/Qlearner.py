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


class Qlearner(Agent):
    def __init__(self, environment, configuration, name='', debug=False):
        num_actions = environment.action_space.n
        num_states = get_num_states(environment)
        self.Q = np.zeros((num_states, num_actions))

        # init super lastly, because it starts the training
        super().__init__(environment=environment, debug=debug, name=name, configuration=configuration)

    def loss_function(self, theta, t):
        pass

    def train1(self, train_params, batch_size):
        alpha, gamma, epsilon = train_params

        state = self.environment.reset()
        total_reward = 0

        done = False
        while not done:
            state_h = get_state_hash(self.environment, state)

            action = self.sample_action(state_h, [epsilon])

            state_new, reward, done = self.environment.step(action)

            total_reward *= gamma
            total_reward += reward
            state_new_h = get_state_hash(self.environment, state_new)
            self.Q[state_h, action] += alpha * (
                    reward + gamma * self.Q[state_new_h].max() - self.Q[state_h, action])
            state = state_new

        return total_reward


    def get_action(self, state):
        return np.argmax(self.Q[state])


    def sample_action(self, state, params):
        epsilon = params[0]
        print(epsilon)
        if np.random.random() < epsilon:
            a = self.environment.action_space.sample()
        else:
            a = self.get_action(state)
        return a
