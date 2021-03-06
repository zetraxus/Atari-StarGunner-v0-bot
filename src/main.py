import gym

from agent import Agent
from test import test
from train import train
from agent import Algorithm
from datetime import datetime

from utils import print_logs

MODEL_WEIGHTS_PATH = "model_weights/"
RESULTS_PATH = "results/"
SAVE_WEIGHTS, LOAD_WEIGHTS, RENDER, SAVE_RESULTS = False, False, True, False
ALGORITHMS = [Algorithm.Q_LEARNING, Algorithm.SARSA]
params = {'TEST_NUM': 2, 'EPISODES_NUM': 5, 'START_LEARNING_ITERATION': 100, 'PENALTY': -10000,
          'PREMIUM': 200, 'LEARNING_RATE': 0.0001, 'EPSILON_MIN': 0.1, 'EPSILON_DECAY': 0.000001,
          'DISCOUNT_FACTOR': 0.95, 'BATCH_SIZE': 32}

if __name__ == "__main__":
    environment = gym.make("StarGunner-v0", frameskip=4)
    log_filename = datetime.now().strftime(RESULTS_PATH + "result_%d-%m-%Y_%H:%M:%S")

    for algorithm in ALGORITHMS:
        print_logs(SAVE_RESULTS, log_filename, str(algorithm.name) + ' - ' + str(params))
        agent = Agent(algorithm, environment.action_space, params)
        if LOAD_WEIGHTS:
            agent.load_network(MODEL_WEIGHTS_PATH + algorithm.name)
        else:
            train(agent, environment, SAVE_WEIGHTS, MODEL_WEIGHTS_PATH + algorithm.name, RENDER, SAVE_RESULTS,
                  log_filename, params)

        result = test(agent, environment, RENDER, SAVE_RESULTS, log_filename, params)
        print_logs(SAVE_RESULTS, log_filename, "Avg test reward = " + str(result))
