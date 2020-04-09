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

from abc import abstractmethod
from Framework.utils import plot
import numpy as np

class Agent:
    def __init__(self, environment, configuration, name, verbosity_level = 10, debug=False):
        self.environment = environment
        self.debug = debug
        self.name = name
        self.verbosity_level = verbosity_level
        self.configuration = configuration
        if self.configuration:
            self.returns = self.train()


    @abstractmethod
    def loss_function(self, theta, t):
        pass

    @abstractmethod
    def sample_action(self, state, params):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def train1(self, train_params, batch_size,verb):
        pass

    def evaluate(self, nb_iterations):
        total_rewards = np.zeros(nb_iterations)
        for i in range(nb_iterations):
            s = self.environment.reset()
            done = False
            total_reward = 0
            while not done:
                a = self.get_action(s)
                s1, r, done = self.environment.step(a)
                total_reward += r
                s = s1
            total_rewards[i] = total_reward
        return total_rewards

    def train(self):
        total_rewards = np.zeros(self.configuration.nb_iterations)
        params = self.configuration.training_params

        for it in range(self.configuration.nb_iterations):
            for i in range(len(params)):
                params[i] = self.configuration.cooling_scheme[i](params[i], it)
            total_rewards[it] = self.train1(params, self.configuration.batch_size,((it+1) %self.verbosity_level == 0))

        if self.configuration.plot_training: plot(total_rewards=total_rewards, average=self.configuration.average)
        return total_rewards