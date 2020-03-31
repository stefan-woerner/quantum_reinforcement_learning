from unittest import TestCase

from gym.spaces import Discrete
from keras.models import Sequential
from mockito import mock
from mockito import verify
from mockito import when, expect

from Agents.DQN import DQN
from Environments.FL.envs import FL
from Framework.Configuration import Configuration


class TestDQN(TestCase):
    def test_init_memory(self):
        model = Sequential()
        when(model).compile(loss='mean_squared_error', optimizer='Adam').thenReturn()
        environment = mock({
            'observation_space': Discrete(8),
        'action_space': Discrete(3)}, FL)
        when(environment).reset().thenReturn(0)
        #when(environment).step(...).thenReturn((1, 10, True))
        expect(environment, times=2).step(...).thenReturn((1,10,True))

        configuration = mock({
            'model': model,
            'memory_size': 2,
            'nb_iterations': 0,
            'training_params': [],
            'plot_training': False
        }, Configuration)

        test_Agent = DQN(environment, configuration)

        verify(environment, times= 2).step(...)

        #self.assertCountEqual(test_Agent.D,[(1,...,1,True),(1,...,1,True)])
        #np.testing.assert_array_equal(test_Agent.D,[([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0], ..., 1,10, True), ([1.0,0,0,0,0,0,0,0], ...,10, True)])
        #self.fail()
