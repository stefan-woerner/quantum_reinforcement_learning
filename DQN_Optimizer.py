import gym
import optuna
import Environments.FL
from Framework.Configuration import Configuration
from Agents.DQN import DQN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


def objective(trial):
    env_name = 'FL-v1'
    env = gym.make(env_name)
    batch_size = int(trial.suggest_discrete_uniform('Batch Size', 30, 100, 10))
    lr = trial.suggest_categorical('Learning Rate', [1e-6, 1e-5, 1e-4, 1e-3])
    iterations = 1000  # TODO: Set at least to 10x to see effects better
    training_params = [lr, .99, .9]
    cooling_schemes = [lambda x, iter: x, lambda x, iter: x, lambda x, iter: 1 - (iter / 10000)]
    memory_size = 300


    confDQN = Configuration(nb_iterations=iterations, training_params=training_params,
                            cooling_scheme=cooling_schemes, batch_size=batch_size,
                            plot_training=False, memory_size=memory_size)

    # Assume that one hidden layer should be enough, since 1hot encoding essentially means,
    # that each state is entered as a trainable constant to the hidden layer.
    # Theoretically, we would just need a different weight from there to each action.
    # So try first with num_states hidden units.
    # Next step: vary this number in the same order.
    # TODO: Train network on optimal policy Q to check this theory
    model = Sequential()
    model.add(Dense(16, input_shape=(16,), activation='tanh'))
    model.add(Dense(env.action_space.n, activation='linear'))

    confDQN = Configuration(nb_iterations=iterations, training_params=training_params, cooling_scheme=cooling_schemes,
                            batch_size=batch_size, plot_training=False, memory_size=memory_size, average=int(batch_size/100))

    confDQN.model = model
    confDQN.target_replacement = 2e10
    # TODO: Make this also part of the Search Space
    # TODO: Recommondation (stackoverflow) is 500 with 10000 it.
    agent = DQN(env, debug=True, configuration=confDQN, verbosity_level=100)
    return agent.evaluate(100).mean()
    # TODO: Include Pruning
    # TODO: Set up DAtabase backend


if __name__ == '__main__':
    study_name = 'First attempt'
    study = optuna.create_study(study_name=study_name, direction='maximize', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


