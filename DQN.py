import numpy as np
import tensorflow as tf

from image_transformations import process_frame
from q_network import build_q_network

EPSILON_MIN = 0.1
max_num_steps = 1000 * 10
EPSILON_DECAY = 0.01
# EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
DISCOUNT_FACTOR = 0.95


class DQN:
    def __init__(self, action_space):
        self.action_space = action_space
        self.network = build_q_network(n_actions=action_space.n)
        self.epsilon = 1.0

    def get_action(self, obs, training):
        frame = process_frame(obs)

        # todo: change
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if (not training) or (np.random.random() > self.epsilon):
            return self.network.predict(frame)[0].argmax()
        else:
            return self.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        frame = process_frame(obs)
        next_frame = process_frame(next_obs)

        q_max = self.network.predict(next_frame)[0].max()
        target_q = reward + (DISCOUNT_FACTOR * q_max)

        with tf.GradientTape() as tape:
            q_values = self.network(frame)
            Q = tf.reduce_sum(q_values, axis=1)
            loss = tf.keras.losses.Huber()(target_q, Q)

        model_gradients = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(model_gradients, self.network.trainable_variables))
