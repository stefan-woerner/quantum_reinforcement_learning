Classical FrozenLake environment with customizable Dimensions and Hole proportions. Default is 4x4 (aka the original Gym Version).
However: Reward for a normal step is always -1. Reward for hitting a hole is always -10. Reward for Completion is 2xSum of dimensions.

The maximum iteration length is currently set to 200 steps statically. It can be changed by accessing max_episode_steps 
directly.


To install this to be used stand alone write Setup.py with:
from setuptools import setup

setup(name = 'FL',
	  version = '0.1.9',
	  install_requires = ['gym', 'six', 'numpy']  #Can list more dependencies here!
	  )

Then run 

pip install -e .

then 

import gym
import FL
env = gym.make('FL-v1')

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3