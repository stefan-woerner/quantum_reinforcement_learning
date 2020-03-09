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
import matplotlib.pyplot as plt


def plot(total_rewards, average=100, instance=None):
    curves = 1
    if len(total_rewards.shape) == 2:
        n = total_rewards.shape[1]
        curves = total_rewards.shape[0]
    else:
        n = len(total_rewards)
        total_rewards = [total_rewards]

    for i in range(curves):
        running_avg = np.empty(n)
        for t in range(n):
            running_avg[t] = total_rewards[i][max(0, t - average):(t + 1)].mean()
        if instance:
            plt.plot(running_avg, label='{} {}'.format(instance, i))
        else:
            plt.plot(running_avg)
    plt.title('Rewards Running Average over {} steps'.format(average))
    plt.show()