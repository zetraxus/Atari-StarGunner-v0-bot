from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical

from src.buffer import Buffer
from src.image_transformations import process_frame
from src.q_network import build_q_network

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.00001
DISCOUNT_FACTOR = 0.95
TARGET_SIZE = (84, 84)
BATCH_SIZE = 32


class Agent:
    def __init__(self, algorithm, action_space):
        self.algorithm = algorithm
        self.action_space = action_space
        self.network = build_q_network(n_actions=action_space.n, input_shape=TARGET_SIZE)
        self.epsilon = 1.0
        self.lives = 5
        self.buffer = Buffer()

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

        calculated_q = self.__calc_sarsa_q(next_obs,
                                           next_frame) if self.algorithm == Algorithm.SARSA else self.__calc_q_learn_q(next_frame)
        target_q = reward + (DISCOUNT_FACTOR * calculated_q)

        self.buffer.add_experience(frame, next_frame, target_q, action)

        if self.buffer.size() == BATCH_SIZE:
            with tf.GradientTape() as tape:
                q_values = self.network(self.buffer.buffer_frames)
                action_one_hot = to_categorical(self.buffer.buffer_actions, self.action_space.n)
                Q = tf.reduce_sum(tf.multiply(q_values, action_one_hot), axis=1)
                loss = tf.keras.losses.Huber()(self.buffer.buffer_targets_q, Q)

            model_gradients = tape.gradient(loss, self.network.trainable_variables)
            self.network.optimizer.apply_gradients(zip(model_gradients, self.network.trainable_variables))
            self.buffer.clear()

    def __calc_q_learn_q(self, next_frame):
        return self.network.predict(next_frame)[0].max()

    def __calc_sarsa_q(self, next_obs, next_frame):
        new_action = self.get_action(next_obs, True)
        return self.network.predict(next_frame)[0][new_action]

    def save_network(self, filepath):
        self.network.save_weights(filepath)

    def load_network(self, filepath):
        self.network.load_weights(filepath)


class Algorithm(Enum):
    Q_LEARNING = 'Q_LEARNING'
    SARSA = 'SARSA'
