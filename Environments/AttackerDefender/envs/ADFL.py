import gym
from gym import error, spaces, utils
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.utils import seeding
from gym.envs.toy_text import discrete

import sys
from contextlib import closing
import math

import numpy as np
from six import StringIO, b

# class FooEnv(gym.Env):
#  metadata = {'render.modes': ['human']}#

#  def __init__(self):
#    ...
#  def step(self, action):
#    ...
#  def reset(self):
#    ...
#  def render(self, mode='human', close=False):
#    ...
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ATTACKER = 0
DEFENDER = 1

MAPS = {
    "4x4": [
        "SFFF",
        "FHFF",
        "FFGF",
        "FHFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FHFFFFFF",
        "FFFFHGFF",
        "FFFFFFFF"
    ],
}


class ADFL(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = desc.shape
        self.reward_range = (0, 1)
        self.max_episode_steps = 100
        self.num_steps = 0
        self.done = False
        self.successful_attack = False

        self.nA_a = 4
        self.nS_a = self.nrow * self.ncol

        self.holes = []
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                letter = desc[row, col]
                if letter in b'H':
                    self.holes.append(s)

        # Basically max_position**num_hole
        self.nS_d = np.sum([((self.ncol * self.nrow) - 1)**(j+1) for j in range(len(self.holes))])
        self.nA_d = len(self.holes) * 4

        self.action_space_a = spaces.Discrete(self.nA_a)
        self.observation_space_a = spaces.Discrete(self.nS_a)

        self.action_space_d = spaces.Discrete(self.nA_d)
        self.observation_space_d = spaces.Discrete(self.nS_d)

        self.seed()
        self.s_a = 0
        self.s_d = self.to_s_d(self.holes)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def to_rc(self, s):
        return int(s / self.ncol) , int(s % self.ncol)

    def to_s(self, row, col):
        return row * self.ncol + col

    def to_s_d(self, holes):
        return  np.sum([el * ((self.ncol * self.nrow) - 1)**j for j, el in enumerate(self.holes)])

    def inc(self, row, col, a):
        if a == LEFT:
            col = max(col - 1, 0)
        elif a == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif a == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif a == UP:
            row = max(row - 1, 0)
        return row, col

    def reset(self):
        self.s_a = 0
        self.s_d = self.to_s_d(self.holes)
        self.lastaction_a = None
        self.lastaction_d = None
        self.num_steps = 0
        self.done = False
        self.successful_attack = False
        return self.s_a, self.s_d

    def step_attacker(self, a):
        row, col = self.to_rc(self.s_a)
        n_row, n_col = self.inc(row, col, a)
        self.num_steps += 1
        if self.num_steps == self.max_episode_steps:
            self.done = True
        self.s_a = self.to_s(n_row, n_col)
        self.lastaction_a = a
        if self.desc[n_row][n_col] in b'H':
            self.done = True
            return self.s_a, -.2, self.done, None
        if self.desc[n_row][n_col] in b'G':
            self.done = True
            self.successful_attack = True
            return self.s_a, 1., self.done, None
        return self.s_a, -.01, self.done, None

    def step_defedenr(self, a):
        hole = self.holes[int(a/4)]
        h_row, h_col = self.to_rc(hole)
        direction = int(a%4)
        n_h_row, n_h_col = self.inc(h_row, h_col, direction)

        # Don't move if its the goal or a hole
        if self.desc[n_h_row][n_h_col] in b'GH':
            n_h_row, n_h_col =h_row, h_col
        self.holes[int(a / 4)] = self.to_s(n_h_row, n_h_col)
        self.desc[h_row, h_col], self.desc[n_h_row, n_h_col] = self.desc[n_h_row, n_h_col], self.desc[h_row, h_col]
        self.lastaction_b = a
        self.s_d = self.to_s_d(self.holes)
        if self.done and self.successful_attack:
            return self.s_d, -.2, self.done, None
        if self.done and not self.successful_attack:
            return self.s_d, 1., self.done, None
        return self.s_d, .01, self.done, None


    def step(self, a, player):
        if player == ATTACKER:
            return self.step_attacker(a)
        elif player == DEFENDER:
            return self.step_defedenr(a)
        else:
            raise Exception('Wrong player identifier')

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction_a is not None:
            outfile.write(f'Action Attacker {["Left", "Down", "Right", "Up"][self.lastaction_a]}\n')
        else:
            outfile.write("\n")
        if self.lastaction_d is not None:
            outfile.write(f'Action Defender  {["Left", "Down", "Right", "Up"][self.lastaction_d]}\n')
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


