import gym

EPISODES_NUM = 1
MAX_ITERATIONS = 5000
TEST_NUM = 10


class DQN:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs, training):
        return self.env.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        pass


def train(agent, env):
    for episode in range(1, EPISODES_NUM + 1):
        obs = env.reset()
        total_reward, iteration, done = False, 0, 0
        while not done and iteration < MAX_ITERATIONS:
            action = agent.get_action(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, next_obs, reward)
            obs, total_reward, iteration = next_obs, total_reward + reward, iteration + 1

        print("Episode = " + str(episode) + " -> reward = " + str(total_reward) + " it = " + str(iteration))


def test(agent, env):
    obs = env.reset()
    total_reward, done = 0, False
    while not done:
        action = agent.get_action(obs, training=False)
        next_obs, reward, done, _ = env.step(action)
        obs, total_reward = next_obs, total_reward + reward

    print("Reward (test) = " + str(total_reward))
    return total_reward


if __name__ == "__main__":
    environment = gym.make('StarGunner-v0')
    agent_DQN = DQN(environment)
    train(agent_DQN, environment)

    # test_reward = 0
    # for _ in range(TEST_NUM):
    #     test_reward += test(agent_DQN, environment)
    #
    # print("Avg test reward = " + str(test_reward / TEST_NUM))
