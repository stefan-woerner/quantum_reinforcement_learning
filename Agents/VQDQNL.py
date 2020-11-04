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

from Framework.Agent import Agent
from Framework.utils import get_num_states
import numpy as np
from qiskit.aqua.components.optimizers import ADAM
from qiskit import *
from datetime import datetime
import copy
import sympy as sy
from sympy.physics.quantum import TensorProduct


class VQDQNL(Agent):

    def __init__(self, environment, configuration, name='', verbosity_level = 10, debug=False):
        self.D = []  # Memory
        self.memory_size = configuration.memory_size
        self.nb_variational_circuits = configuration.nb_variational_circuits
        self.debug = debug

        self.train_counter = 0
        self.episode = 0

        num_actions = environment.action_space.n
        num_states = get_num_states(environment)
        self.nb_circuit_params = 1
        self.nb_qbits = int(np.max([num_actions, np.log2(num_states)]))
        self.theta = np.array(
            [np.random.random(self.nb_circuit_params) for j in range(self.nb_qbits * self.nb_variational_circuits)]).flatten()

        self.target_theta = copy.deepcopy(self.theta.copy())


        self.init_memory(environment)
        self.loss_mem = []

        # init super lastly, because it starts the training
        super().__init__(environment=environment, debug=debug, name=name, configuration=configuration)

    # def get_symbolic_diffs(self):
    #     return [sy.lambdify(self.symbolic_theta, sy.diff(self.symbolic_circuit, i), 'numpy') for i in self.symbolic_theta]
    #
    # def get_symbolic_circuit(self):
    #     def u3(theta, phi, lam):
    #         return sy.Matrix([[sy.cos(theta / 2), -sy.exp(sy.I * lam) * sy.sin(theta / 2)],
    #                           [sy.exp(sy.I * phi) * sy.sin(theta / 2), sy.exp(sy.I * (phi + lam)) * sy.cos(theta / 2)]])
    #
    #     def CNOT(nb_bits):
    #         circ = QuantumCircuit(nb_bits)
    #         for i in range(1, nb_bits):
    #             circ.cx(i - 1, i)
    #
    #         # Select the UnitarySimulator from the Aer provider
    #         simulator = Aer.get_backend('unitary_simulator')
    #         # Execute and get counts
    #         result = execute(circ, simulator).result()
    #         unitary = result.get_unitary(circ)
    #
    #         return sy.Matrix(unitary)
    #
    #     self.symbolic_theta  = np.array(
    #         [[sy.Symbol('t_{}'.format(j), real=True), sy.Symbol('p_{}'.format(j), real=True),
    #           sy.Symbol('l_{}'.format(j), real=True)] for j in range(self.nb_qbits * self.nb_variational_circuits)]).flatten()
    #     cnot = CNOT(self.nb_qbits)
    #
    #     gate_list = []
    #     count = 0
    #     for j in range(self.nb_variational_circuits):
    #         # Cnot
    #         gate_list.append(cnot.copy())
    #         u_list = []
    #         for i in range(self.nb_qbits):
    #             u_list.append(u3(self.symbolic_theta [count], self.symbolic_theta [count + 1], self.symbolic_theta [count + 2]))
    #             count += 3
    #         while len(u_list) > 1:
    #             u_list[1] = TensorProduct(u_list[1], u_list[0])
    #             u_list = u_list[1:]
    #         gate_list.append(u_list[0])
    #
    #     while len(gate_list) > 1:
    #         gate_list[-2] = gate_list[-1] * gate_list[-2]
    #         gate_list = gate_list[:-1]
    #     mat = gate_list[0]
    #     # mat.simplify()
    #     mat = mat * sy.Matrix(self.mask)
    #     return mat

    def init_memory(self, environment):
        while len(self.D) < self.memory_size:
            s = environment.reset()
            done = False
            while not done:
                a = environment.action_space.sample()
                s1, r, done = environment.step(a)
                self.D.append((s, a, r, s1, done))
                s = s1
                if len(self.D) == self.memory_size:
                    break

    def get_mask(self):
        mask = np.zeros((2**self.nb_qbits, self.nb_qbits))
        for i in range(self.nb_qbits):
            # probs[i*self.nb_qbits:i*self.nb_qbits].sum()
            mask[i * self.nb_qbits:(i + 1) * self.nb_qbits, i] = 1
        return mask

    def get_exp(self, result_list, index):
        res = np.zeros(self.nb_qbits)
        for key in result_list.get_counts(index):
            spl = -1 * np.array([int(i) or -1 for i in key])
            res = res + spl * (result_list.get_counts(index)[key] / result_list.results[index].shots)

    def get_qvalues(self, s_list, theta, shots=1024):
        qc_list = []
        for s in s_list:
            q = QuantumRegister(self.nb_qbits)
            c = ClassicalRegister(self.nb_qbits)
            qc = QuantumCircuit(q, c)

            # State encoding
            d = np.binary_repr(int(s), self.nb_qbits)
            for j, i in enumerate(d):
                qc.u3(int(i)*np.pi, -np.pi/2, np.pi/2, q[j]) # Rx = U3 with lambda = pi/2, phi=-pi/2
 #               qc.u1(int(i)*np.pi, q[j]) # Rz = U1

            # Variational circuits
            count = 0
            for rep in range(self.nb_variational_circuits):
                for i in range(1, self.nb_qbits):
                    qc.cx(q[i - 1], q[i])

                for i in range(self.nb_qbits):
                    qc.u3(theta[count], 0, 0, q[i])
                    count = count + self.nb_circuit_params

            qc.measure(q, c)

            qc_list.append(qc)

        #backend_sim = Aer.get_backend('statevector_simulator')
        backend_sim = Aer.get_backend('qasm_simulator')
        qob = assemble(qc_list, backend_sim)#, shots=10)

        # result_list = execute(qc_list, backend_sim, shots=shots).result()
        job = backend_sim.run(qob)
        result_list = job.result()
        expect_list = []

        for k in range(len(qc_list)):
            res = self.get_exp(result_list, k)
            expect_list.append(res)

        return expect_list

    def loss(self, theta, t):
        su = 0
        Qs = self.get_qvalues(t[:, 1], theta)

        for i in range(len(t)):
            diff = np.abs(t[i][0] - Qs[i][int(t[i][2])])
            if diff < 1:
                su += .5 * diff**2
            else:
                su += diff - .5
        return su

    def train1(self, train_params, batch_size, verb=True):
        alpha, gamma, epsilon = train_params

        self.episode += 1

        state = self.environment.reset()
        total_reward = 0
        loss_list = []

        # sample from policy
        done = False
        while not done:
            a = self.sample_action(state, [epsilon])
            new_state, r, done, = self.environment.step(a)
            total_reward += r
            self.D.pop(0)
            self.D.append((state, a, r, new_state, done))


            mB_ind = np.random.choice(range(self.memory_size), size=batch_size, replace=True)
            mB = np.array(self.D)[mB_ind]
            t = []
            for j in range(batch_size):
                y_j = mB[j][2] * mB[j][-1] or mB[j][2] + gamma * max(self.get_qvalues([mB[j][3]], self.target_theta)[0])
                t.append([y_j, mB[j][0], mB[j][1]])

            t = np.array(t)

            adam = ADAM(maxiter=10, amsgrad=True, lr=alpha)
            if self.debug:
                start = datetime.now()
            self.theta, batch_losses, _ = adam.optimize(len(self.theta), lambda x: self.loss(x, t), initial_point=self.theta)
                                                        #gradient_function=lambda x: self.gradient(x, t)
            loss_list.append(batch_losses)
            if self.debug:
                print(datetime.now() - start)

            if self.train_counter % self.configuration.target_replacement == 0:
                self.target_theta = copy.deepcopy(self.theta.copy())
            self.train_counter = self.train_counter + 1

            # update state
            state = new_state

        self.loss_mem.append(np.mean(np.array(loss_list).flatten()))
        if verb:
            print('train it:', self.episode, ' Return: ', total_reward, ' Loss: ', self.loss_mem[-1],
                  ' epsilon ', epsilon)

        return total_reward

    def get_action(self, state):
        return np.argmax(self.get_qvalues([state], self.theta)[0])

    def sample_action(self, state, params):
        epsilon = params[0]
        if np.random.random() < epsilon:
            a = self.environment.action_space.sample()
        else:
            a = self.get_action(state)
        return a
