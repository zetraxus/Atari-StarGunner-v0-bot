from image_transformations import preprocess_image


class DQN:
    def __init__(self, env, img_target_width, img_target_height):
        self.env = env
        self.target_width = img_target_width
        self.target_height = img_target_height

    def get_action(self, obs, training):
        img = preprocess_image(obs, self.target_width, self.target_height)
        return self.env.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        pass
