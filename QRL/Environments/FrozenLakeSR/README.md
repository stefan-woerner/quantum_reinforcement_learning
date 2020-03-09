Classical FrozenLake environment with customizable Dimensions and Hole proportions. Default is 4x4 (aka the original Gym Version).
However: Reward for a normal step is always -1. Reward for hitting a hole is always -10. Reward for Completion is 2xSum of dimensions.

To install environment run in this folder:

pip install -e .

then 

import gym
import FL
env = gym.make('FL-v0')

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3