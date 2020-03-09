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

from QRL.Framework.Configuration import RLAgent, Environment
import numpy as np
from qiskit.aqua.components.optimizers import ADAM
from qiskit import *
from datetime import datetime
import gym
from gym.envs.registration import register


class VQDQL(RLAgent):
    def __init__(self, environment, memory_size, nb_variational_circuits, trainer, name='', debug=False):
        self.D = []  # Memory
        self.memory_size = memory_size
        self.nb_variational_circuits = nb_variational_circuits
        self.debug = debug

        self.nb_qbits = int(np.max([environment.nb_actions, np.log2(environment.nb_states)]))
        self.theta = np.array(
            [2 * np.pi * np.random.random(3) for j in range(self.nb_qbits * self.nb_variational_circuits)]).flatten()

        self.init_memory(environment)

        # init super lastly, because it starts the training
        super().__init__(environment=environment, debug=debug, name=name, trainer=trainer)

    def init_memory(self, environment):
        while len(self.D) < self.memory_size:
            s = environment.reset()
            done = False
            while not done:
                a = environment.sample_actions_space()
                s1, r, done = environment.step(a)
                self.D.append((s, a, r, s1, done))
                s = s1
                if len(self.D) == self.memory_size:
                    break

    def get_qvalues(self, s_list, theta, shots=1024):
        qc_list = []
        for s in s_list:
            q = QuantumRegister(self.nb_qbits)
            c = ClassicalRegister(self.nb_qbits)
            qc = QuantumCircuit(q, c)

            d = np.binary_repr(int(s), self.nb_qbits)
            for j, i in enumerate(d):
                if i == '1':
                    qc.x(q[j])

            theta = theta.reshape((self.nb_qbits, 3))

            for rep in range(self.nb_variational_circuits):
                for i in range(1, self.nb_qbits):
                    qc.cx(q[i - 1], q[i])

                for i in range(self.nb_qbits):
                    qc.u3(theta[i][0], theta[i][1], theta[i][2], q[i])

            qc.measure(q, c)

            qc_list.append(qc)

        backend_sim = Aer.get_backend('statevector_simulator')
        qob = assemble(qc_list, backend_sim)

        #result_list = execute(qc_list, backend_sim, shots=shots).result()
        job = backend_sim.run(qob)
        result_list = job.result()
        expect_list = []
        for result in result_list.results:
            proba = abs(np.array(result.data.statevector)) ** 2

            expect = np.zeros(self.nb_qbits)

            for c in range(len(proba)):
                cbin = np.binary_repr(int(c), self.nb_qbits)

                for n in range(self.nb_qbits):
                    if cbin[n] == '1':
                        expect[n] += proba[c]

            expect_list.append(expect)

        return expect_list

    def loss(self, theta, t):
        su = 0
        Qs = self.get_qvalues(t[:, 1], theta)

        for i in range(len(t)):
            su += (t[i][0] - Qs[i][int(t[i][2])]) ** 2
        return su

    def train1(self, train_params, batch_size):
        alpha, gamma, epsilon = train_params
        s = self.environment.reset()
        done = False
        total_reward = 0
        while not done:
            a = self.sample_action(s, [epsilon])
            s1, r, done, = self.environment.step(a)
            total_reward += r
            self.D.pop(0)
            self.D.append((s, a, r, s1, done))
        mB_ind = np.random.choice(range(self.memory_size), size=batch_size, replace=False)
        mB = np.array(self.D)[mB_ind]
        t = []
        for j in range(batch_size):
            if mB[j][-1]:
                y_j = mB[j][2]
            else:
                y_j = mB[j][2] + gamma * max(self.get_qvalues([mB[j][3]], self.theta)[0])
            y_j /= 2
            y_j += 0.5
            t.append([y_j, mB[j][0], mB[j][1]])

        t = np.array(t)

        adam = ADAM(maxiter=10, lr=alpha)
        if self.debug:
            start = datetime.now()
        theta,_,_ = adam.optimize(3*self.nb_qbits,lambda x: self.loss(x,t), initial_point = self.theta)
        if self.debug:
            print(datetime.now() - start)

        return total_reward

    def get_action(self, state):
        return np.argmax(self.get_qvalues([state], self.theta)[0])

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.sample_actions_space()
        else:
            a = self.get_action(state)
        return a


class DeterministicFrozenLake(Environment):
    def __init__(self):
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name': '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.8196,
        )
        self.environment = gym.make("FrozenLakeNotSlippery-v0")
        super().__init__(nb_states = self.get_num_states(), nb_actions=self.environment.action_space.n)

    def get_num_states(self):
        if type(self.environment.observation_space) != gym.spaces.tuple.Tuple:
            return self.environment.observation_space.n
        dim_list = []
        for sp in self.environment.observation_space:
            dim_list.append(sp.n)
        dim_list = np.array(dim_list)
        return dim_list.prod()

    def reset(self):
        return self.environment.reset()

    def step(self, action):
        s1, r, done, _ = self.environment.step(action)
        if r == 0:
            if done:
                r = -0.2
            else:
                r = -0.01
        else:
            r = 1.0
        return s1, r, done

    def sample_actions_space(self):
        return self.environment.action_space.sample()