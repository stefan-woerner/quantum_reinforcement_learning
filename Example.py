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

from Agents.VQDQNL import VQDQNL
from Agents.Qlearner import Qlearner
from Agents.DQN import DQN
from Agents.DDQN import DDQN
from Agents.DQlearner import DQlearner
from Framework.Configuration import Configuration
import Framework.utils as utils
from Framework.Evaluator import Evaluator
from Framework.utils import get_num_states
import gym
import Environments.FL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Setup GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Setup the environment containing
env_name = 'FL-v1'
env = gym.make(env_name)

np.random.seed(2)
env.action_space.np_random.seed(2)

# Set the training parameters
batch_size = 10
nb_iterations = 750
#alpha, gamma, epsilon
cooling_schemes = [lambda x, iter: x, lambda x, iter: x, lambda x,iter: x*0.999]
# memory
memory_size = 80
# target update
target_replacement = 10

#alpha, gamma, epsilon
training_params_Q = [.6, .8, .9]

# Instantiate the trainer
confQ = Configuration(nb_iterations=nb_iterations, training_params=training_params_Q, cooling_scheme=cooling_schemes,
                        batch_size=1, average=int(batch_size/100), test_steps=1, verbosity_level=1e20)


#alpha, gamma, epsilon
training_params_DQN = [.6, .8, .9] # DQN opt .1, .8, .9 overfitting nach 590

# NN model for DQN
num_states = get_num_states(env)
model = Sequential()
model.add(Dense(8, input_shape=(16, ), activation='tanh'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss=Huber(), optimizer=SGD(training_params_DQN[0]))

confDQN = Configuration(nb_iterations=nb_iterations, training_params=training_params_DQN[1:],
                        cooling_scheme=cooling_schemes[1:], batch_size=batch_size,
                        memory_size=memory_size, average=int(nb_iterations/10), model=model,
                        target_replacement=target_replacement, test_steps=1, verbosity_level=10)

confDQN.clone_model = clone_model
confDQN.embedding = lambda state: to_categorical(state, num_classes=num_states).reshape(num_states)

training_params_VQDQN = [.22, .8, .9] # DQN opt .1, .8, .9 overfitting nach 590
confVQD = Configuration(nb_iterations=nb_iterations, training_params=training_params_VQDQN, cooling_scheme=cooling_schemes,
                        memory_size=memory_size, batch_size=batch_size, plot_training=False, average=int(batch_size/5), verbosity_level=10)
confVQD.nb_variational_circuits=1

# Instantiate the agents
#agent_Q = DQlearner(env, configuration=confQ)
agent_DQN = DDQN(env, configuration=confDQN)
#agent_VQDQN = VQDQNL(env, configuration=confVQD)

agent_DQN.init_memory(agent_DQN.environment)

# train them
#start = datetime.now()
#total_rewards_Q, eval_rewards_Q = agent_Q.train()
#print(datetime.now() - start)
#start = datetime.now()
total_rewards_DQN, eval_rewards_DQN = agent_DQN.train()
#print(datetime.now() - start)
#total_rewards_VQDQN, eval_rewards_VQDQN = agent_VQDQN.train()

#stats = utils.compute_stats([total_rewards_Q, total_rewards_DQN])
#utils.plot_stats(stats, ['Q', 'DQN'], path='avg_return.png')

#np.save(f'eval_rewards_Q', eval_rewards_Q)
#np.save(f'eval_rewards_DQN', eval_rewards_DQN)

# Plot results
#utils.plot(eval_rewards_Q[:,0], average=10)
utils.plot(eval_rewards_DQN[:,0],average=10)
#utils.plot(eval_rewards_VQDQN[:,0],average=10)



# Compare the performance
#evaluator = Evaluator([agent, agent1], 5)
#rewards = evaluator.evaluate(plot_results=True, average=2)
