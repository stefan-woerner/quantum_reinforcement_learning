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
from Framework.utils import plot
from Framework.Evaluator import Evaluator
from Framework.utils import get_num_states
import gym
import Environments.FL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Setup the environment containing
env_name = 'FL-v1'
env = gym.make(env_name)

# Set the training parameters
batch_size = 100
nb_iterations = 5
#alpha, gamma, epsilon
training_params =[.1, .99, .9]
#alpha, gamma, epsilon
cooling_schemes = [lambda x, iter: x, lambda x, iter: x, lambda x,iter: x*(0.9999**iter)]

# Instantiate the trainer
confVQD = Configuration(nb_iterations=nb_iterations, training_params=training_params, cooling_scheme=cooling_schemes,
                        batch_size=batch_size, plot_training=True, average=int(batch_size/5))

confQ = Configuration(nb_iterations=500, training_params=training_params, cooling_scheme=cooling_schemes,
                        batch_size=1, plot_training=True, average=int(batch_size/100))

# NN model for DQN
num_states = get_num_states(env)
model = Sequential()
model.add(Dense(8, input_shape=(16, ), activation='tanh'))

model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

iterations = 200000
confDQN = Configuration(nb_iterations=iterations, training_params=training_params,
                        cooling_scheme=[lambda x, iter: x, lambda x, iter: x,
                                        lambda x, iter: x], batch_size=batch_size,
                        plot_training=True, memory_size=300,
                        average=int(batch_size/100))
# TODO: average should depend on iterations not batch_size??-- discuss
confDQN.model = model
confDQN.target_replacement = 10

# Example to train two VQDQL agents

#agent = VQDQL(env, memory_size= 100, nb_variational_circuits=1, configuration=confVQD)
#agent1 = Qlearner(env, debug=True, configuration=confQ)
agent2 = DQN(env, debug=True, configuration=confDQN)

total_rewards = agent2.evaluate(1)
plt.plot(total_rewards)
plt.show()
print(np.mean(total_rewards))



# Compare the performance
#evaluator = Evaluator([agent, agent1], 5)
#rewards = evaluator.evaluate(plot_results=True, average=2)
