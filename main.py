from datetime import datetime

import gym

from agent import Agent

EPISODES_NUM = 50
MAX_ITERATIONS = 10000
TEST_NUM = 10
RENDER = True
PENALTY = -500
PREMIUM = 200


def get_reward_for_learning(agent, game_reward, info):
    lives = info['ale.lives']
    if lives < agent.lives:
        agent.decrement_lives()
        return PENALTY

    if game_reward > 0:
        return PREMIUM
    else:
        return 0


def train(agent, env):
    for episode in range(1, EPISODES_NUM + 1):
        obs = env.reset()
        total_reward, iteration, done = 0, 0, False
        while not done and iteration < MAX_ITERATIONS:
            env.render() if RENDER else None
            action = agent.get_action(obs, training=True)
            next_obs, game_reward, done, info = env.step(action)
            reward_for_learning = get_reward_for_learning(agent, game_reward, info)
            agent.update(obs, action, next_obs, reward_for_learning)
            obs, total_reward, iteration = next_obs, total_reward + game_reward, iteration + 1

        print("{0} Episode = {1} -> reward = {2} it = {3}, epsilon = {4}".format(
            str(datetime.now().strftime("%H:%M:%S")), str(episode), str(total_reward),
            str(iteration), str(agent.epsilon)))


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
