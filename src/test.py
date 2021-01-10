TEST_NUM = 10


def single_test(agent, env, render):
    obs = env.reset()
    total_reward, done = 0, False
    while not done:
        env.render() if render else None
        action = agent.get_action(obs, training=False)
        next_obs, reward, done, _ = env.step(action)
        obs, total_reward = next_obs, total_reward + reward

    print("Reward (test) = " + str(total_reward))
    return total_reward


def test(agent, environment, render):
    test_reward = 0
    for _ in range(TEST_NUM):
        test_reward += single_test(agent, environment, render)

    return test_reward / TEST_NUM
