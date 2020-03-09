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

import numpy as np

class Evaluator:
    def __init__(self, agents_list, nb_iterations):
        self.agents_list = agents_list
        self.nb_iterations = nb_iterations

    def evaluate(self, plot_results = False, average=100):
        total_rewards_agents = np.zeros((len(self.agents_list), self.nb_iterations))
        for i in range(len(self.agents_list)):
            total_rewards_agents[i] = self.agents_list[i].evaluate(self.nb_iterations)

        if plot_results:
            plot(total_rewards_agents, average=average, instance='Agent')

        return total_rewards_agents