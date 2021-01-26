from utils import print_logs


def single_test(agent, env, render, save_results, log_filename):
    obs = env.reset()
    total_reward, it, done = 0, 0, False
    while not done:
        env.render() if render else None
        action = agent.get_action(training=False, obs=obs)
        next_obs, reward, done, _ = env.step(action)
        obs, total_reward, it = next_obs, total_reward + reward, it + 1

    print_logs(save_results, log_filename, "Reward (test) = " + str(total_reward) + ', it = ' + str(it))
    return total_reward


def test(agent, environment, render, save_results, log_filename, params):
    test_reward = 0
    for _ in range(params['TEST_NUM']):
        test_reward += single_test(agent, environment, render, save_results, log_filename)

    return test_reward / params['TEST_NUM']
