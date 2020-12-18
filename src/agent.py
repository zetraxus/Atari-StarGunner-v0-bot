import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.image_transformations import process_frame
from src.q_network import build_q_network

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.00001
DISCOUNT_FACTOR = 0.95
TARGET_SIZE = (70, 70)


class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.network = build_q_network(n_actions=action_space.n, input_shape=TARGET_SIZE)
        self.epsilon = 1.0
        self.lives = 5

    def decrement_lives(self):
        self.lives -= 1

    def get_lives(self):
        return self.lives

    def get_action(self, obs, training):
        frame = process_frame(obs, TARGET_SIZE)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if (not training) or (np.random.random() > self.epsilon):
            return self.network.predict(frame)[0].argmax()
        else:
            return self.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        frame = process_frame(obs, TARGET_SIZE)
        next_frame = process_frame(next_obs, TARGET_SIZE)

        q_max = self.network.predict(next_frame)[0].max()
        target_q = reward + (DISCOUNT_FACTOR * q_max)

        with tf.GradientTape() as tape:
            q_values = self.network(frame)
            action_one_hot = to_categorical(action, self.action_space.n)
            Q = tf.reduce_sum(tf.multiply(q_values, action_one_hot), axis=1)
            loss = tf.keras.losses.Huber()(target_q, Q)

        model_gradients = tape.gradient(loss, self.network.trainable_variables)
        self.network.optimizer.apply_gradients(zip(model_gradients, self.network.trainable_variables))

    def save_network(self, filepath):
        self.network.save_weights(filepath)

    def load_network(self, filepath):
        self.network.load_weights(filepath)
