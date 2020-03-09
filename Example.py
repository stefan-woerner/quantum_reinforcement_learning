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

from Agents.VQDQL import VQDQL
from Framework.Configuration import Configuration
from Framework.Evaluator import Evaluator
import gym
import Environments.FL

# Setup the environment containing
env = gym.make('FL-v0')

# Set the training parameters
batch_size = 10
nb_iterations = 5
#alpha, gamma, epsilon
training_params =[.1, .99, .9]
#alpha, gamma, epsilon
cooling_schemes = [lambda x, iter: x, lambda x, iter: x, lambda x,iter: x*(0.99**iter)]

# Instantiate the trainer, conta
conf = Configuration(nb_iterations=nb_iterations, training_params=training_params, cooling_scheme=cooling_schemes,
                        batch_size=batch_size, plot_training=True, average=int(batch_size/5))

# Example to train two VQDQL agents

agent = VQDQL(env, memory_size= 100, nb_variational_circuits=1, configuration=conf)
#agent1 = VQDQL(env, memory_size= 50, nb_variational_circuits=1, debug=True, configuration=conf)

# Compare the performance
evaluator = Evaluator([agent, agent], 5)
rewards = evaluator.evaluate(plot_results=True, average=2)



