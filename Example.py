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
from Agents.Qlearner import Qlearner
from Agents.DQN import DQN
from Framework.Configuration import Configuration
from Framework.Evaluator import Evaluator
from Framework.utils import get_num_states
import gym
import Environments.FL
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

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
confVQD = Configuration(nb_iterations=nb_iterations, training_params=training_params, cooling_scheme=cooling_schemes,
                        batch_size=batch_size, plot_training=True, average=int(batch_size/5))

confQ = Configuration(nb_iterations=10000, training_params=training_params, cooling_scheme=cooling_schemes,
                        batch_size=1, plot_training=True, average=int(batch_size/100))

# NN model for DQN
num_states = get_num_states(env)
model = Sequential()
model.add(Dense(num_states, input_shape=(16, ), activation='relu'))
model.add(Dense(num_states, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

iterations = 100000
confDQN = Configuration(nb_iterations=iterations, training_params=training_params[1:], cooling_scheme=[lambda x, iter: x, lambda x,iter: 1-(iter/iterations)], batch_size=30, plot_training=True,memory_size=500,
                        average=int(batch_size/100))
confDQN.model = model
confDQN.target_replacement = 1e10

# Example to train two VQDQL agents

#agent = VQDQL(env, memory_size= 100, nb_variational_circuits=1, configuration=confVQD)
#agent1 = Qlearner(env, debug=True, configuration=confQ)
agent2 = DQN(env, debug=True, configuration=confDQN)

agent2.evaluate(100)

# Compare the performance
#evaluator = Evaluator([agent, agent1], 5)
#rewards = evaluator.evaluate(plot_results=True, average=2)
