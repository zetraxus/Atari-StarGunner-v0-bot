import gym

from src.agent import Agent
from src.test import test
from src.train import train

MODEL_WEIGHTS_PATH = "model_weights/network_weights"
SAVE_WEIGHTS = False
LOAD_WEIGHTS = False
RENDER = False

if __name__ == "__main__":
    environment = gym.make("StarGunner-v0", frameskip=4)
    agent = Agent(environment.action_space)

    if LOAD_WEIGHTS:
        agent.load_network(MODEL_WEIGHTS_PATH)
    else:
        train(agent, environment, SAVE_WEIGHTS, MODEL_WEIGHTS_PATH, RENDER)

    print("Avg test reward = " + str(test(agent, environment, RENDER)))

# todo
# history_length
# batch learning
# discount factor
# input shape
# gpu
