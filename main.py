from datetime import datetime

import gym
from agent import Agent

EPISODES_NUM = 20
MAX_ITERATIONS = 10000
TEST_NUM = 10
RENDER = False


def train(agent, env):
    for episode in range(1, EPISODES_NUM + 1):
        obs = env.reset()
        total_reward, iteration, done = 0, 0, False
        while not done and iteration < MAX_ITERATIONS:
            env.render() if RENDER else None
            action = agent.get_action(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, next_obs, reward)
            obs, total_reward, iteration = next_obs, total_reward + reward, iteration + 1

        print("{0} Episode = {1} -> reward = {2} it = {3}".format(str(datetime.now().strftime("%H:%M:%S")),
                                                                  str(episode), str(total_reward), str(iteration)))


def test(agent, env):
    obs = env.reset()
    total_reward, done = 0, False
    while not done:
        env.render() if RENDER else None
        action = agent.get_action(obs, training=False)
        next_obs, reward, done, _ = env.step(action)
        obs, total_reward = next_obs, total_reward + reward

    print("Reward (test) = " + str(total_reward))
    return total_reward


if __name__ == "__main__":
    environment = gym.make("StarGunner-v0")
    agent = Agent(environment.action_space)
    train(agent, environment)

    test_reward = 0
    for _ in range(TEST_NUM):
        test_reward += test(agent, environment)

    print("Avg test reward = " + str(test_reward / TEST_NUM))

# todo
# environment.frameskip = 4 ?
# history_length
# batch learning
# epsilon ?
# discount factor
# save/load weights