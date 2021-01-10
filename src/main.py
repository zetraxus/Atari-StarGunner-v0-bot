import gym

from src.agent import Agent
from src.test import test
from src.train import train
import tensorflow as tf

MODEL_WEIGHTS_PATH = "model_weights/network_weights"
SAVE_WEIGHTS = True
LOAD_WEIGHTS = True
RENDER = False

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
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
