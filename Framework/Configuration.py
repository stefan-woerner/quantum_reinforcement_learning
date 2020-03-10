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


class Configuration:
    def __init__(self, nb_iterations, training_params, cooling_scheme, memory_size=0, batch_size=1, plot_training=False, average=100):
        self.nb_iterations = nb_iterations
        self.training_params = training_params
        self.memory_size = memory_size
        self.cooling_scheme = cooling_scheme
        self.batch_size = batch_size
        self.plot_training = plot_training
        self.average = average