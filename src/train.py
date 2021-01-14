from datetime import datetime

EPISODES_NUM = 10
MAX_ITERATIONS = 5000
START_LEARNING_ITERATION = 100

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
        return 1


def train(agent, env, save_weights, model_weights_path, render):
    for episode in range(1, EPISODES_NUM + 1):
        obs = env.reset()
        total_reward, iteration, done = 0, 0, False
        while not done and iteration < MAX_ITERATIONS:
            env.render() if render else None
            action = agent.get_action(obs, training=True)
            next_obs, game_reward, done, info = env.step(action)
            if iteration > START_LEARNING_ITERATION:
                reward_for_learning = get_reward_for_learning(agent, game_reward, info)
                agent.update(obs, action, next_obs, reward_for_learning)
            obs, total_reward, iteration = next_obs, total_reward + game_reward, iteration + 1

        print("{0} Episode = {1} -> reward = {2} it = {3}, epsilon = {4}".format(
            str(datetime.now().strftime("%H:%M:%S")), str(episode), str(total_reward),
            str(iteration), str(agent.epsilon)))

    if save_weights:
        agent.save_network(model_weights_path)
