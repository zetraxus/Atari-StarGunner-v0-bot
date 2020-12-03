from image_transformations import process_frame
from q_network import build_q_network


class DQN:
    def __init__(self, env):
        self.env = env
        self.network = build_q_network(n_actions=env.action_space.n)

    def get_action(self, obs, training):
        frame = process_frame(obs)
        q_vals = self.network.predict(frame.reshape((-1, 84, 84, 1)))[0]
        return q_vals.argmax()
        # return self.env.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        pass
