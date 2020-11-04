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
from Framework.utils import get_num_states


class Agent:
    def __init__(self, environment, configuration, name, debug=False):
        self.environment = environment
        self.debug = debug
        self.name = name
        self.configuration = configuration

        self.num_actions = environment.action_space.n
        self.num_states = get_num_states(environment)


    @abstractmethod
    def sample_action(self, state, params):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def train1(self, train_params:list, batch_size:int, verb:bool):
        pass

    @abstractmethod
    def save(self, path:str):
        pass

    @abstractmethod
    def load(self, path:str):
        pass

    def evaluate(self, nb_iterations):
        step_counter = np.zeros(nb_iterations)
        total_rewards = np.zeros(nb_iterations)
        for i in range(nb_iterations):
            steps = 0
            s = self.environment.reset()
            done = False
            total_reward = 0
            while not done:
                a = self.get_action(s)
                s1, r, done,_ = self.environment.step(a)
                total_reward += r
                s = s1
                steps += 1
            total_rewards[i] = total_reward
            step_counter[i] = steps
        return total_rewards, step_counter

    def train(self):
        total_rewards = np.zeros(self.configuration.nb_iterations)
        params = self.configuration.training_params
        eval_rewards = np.zeros((int(self.configuration.nb_iterations / self.configuration.test_step), 2))

        for it in range(self.configuration.nb_iterations):
            for i in range(len(params)):
                params[i] = self.configuration.cooling_scheme[i](params[i], it)
            total_rewards[it] = self.train1(params, self.configuration.batch_size,
                                            ((it + 1) % self.configuration.verbosity_level == 0))

            if (it + 1) % self.configuration.test_step == 0 and (self.name == '' or self.name==0):
                _eval, steps = self.evaluate(1)
                eval_rewards[int(it / self.configuration.test_step), 0] = _eval.mean()
                eval_rewards[int(it / self.configuration.test_step), 1] = int(steps[0])
                if ((it + 1) % self.configuration.verbosity_level == 0):
                    print(f'Iteration: {it + 1} Evaluation: {eval_rewards[int(it / self.configuration.test_step), 0]} \
                    Steps: {eval_rewards[int(it / self.configuration.test_step), 1]}')
                if self.configuration.test_plot:
                        self.plot_q_values(self.num_states, (self.num_actions, 3), iteration=it)#, camera=camera)

        if self.configuration.plot_training and (self.name == '' or self.name==0):
            plot(total_rewards=total_rewards, average=self.configuration.average)
        return total_rewards, eval_rewards
