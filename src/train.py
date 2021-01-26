from datetime import datetime
from utils import print_logs


def get_reward_for_learning(agent, game_reward, info, params):
    lives = info['ale.lives']
    if lives < agent.lives:
        agent.decrement_lives()
        return params['PENALTY']

    if game_reward > 0:
        return params['PREMIUM']
    else:
        return 0


def train(agent, env, save_weights, model_weights_path, render, save_results, log_filename, params):
    for episode in range(1, params['EPISODES_NUM'] + 1):
        obs = env.reset()
        total_reward, iteration, done = 0, 0, False
        while not done:
            env.render() if render else None
            action = agent.get_action(training=True, obs=obs)
            next_obs, game_reward, done, info = env.step(action)
            if iteration > params['START_LEARNING_ITERATION']:
                reward_for_learning = get_reward_for_learning(agent, game_reward, info, params)
                agent.update(obs, action, next_obs, reward_for_learning)
            obs, total_reward, iteration = next_obs, total_reward + game_reward, iteration + 1

        info = "{0} Episode = {1} -> reward = {2} it = {3}, epsilon = {4}".format(
            str(datetime.now().strftime("%H:%M:%S")), str(episode), str(total_reward),
            str(iteration), str(agent.epsilon))
        print_logs(save_results, log_filename, info)

    if save_weights:
        agent.save_network(model_weights_path)
