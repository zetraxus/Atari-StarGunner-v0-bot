from enum import Enum

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

EPSILON_MIN = 0.1
EPSILON_DECAY = 0.00001
DISCOUNT_FACTOR = 0.95
TARGET_SIZE = (84, 84)
BATCH_SIZE = 32


class Agent:
    def __init__(self, algorithm, action_space, learning_rate):
        self.algorithm = algorithm
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.network = self.build_q_network()
        self.epsilon = 1.0
        self.lives = 5
        self.buffer = Buffer()

        if self.algorithm == Algorithm.SARSA:
            self.__calc_q = self.__calc_sarsa_q
        elif self.algorithm == Algorithm.Q_LEARNING:
            self.__calc_q = self.__calc_q_learn_q

    def decrement_lives(self):
        self.lives -= 1

    def get_lives(self):
        return self.lives

    def get_action(self, training, obs=None, frame=None):
        if frame is None:
            frame = self.process_frame(obs)
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if (not training) or (np.random.random() > self.epsilon):
            return self.network.predict(frame)[0].argmax()
        else:
            return self.action_space.sample()

    def update(self, obs, action, next_obs, reward):
        frame = self.process_frame(obs)
        next_frame = self.process_frame(next_obs)

        calculated_q = self.__calc_q(next_frame)
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

    def __calc_sarsa_q(self, next_frame):
        new_action = self.get_action(training=True, frame=next_frame)
        return self.network.predict(next_frame)[0][new_action]

    def save_network(self, filepath):
        self.network.save_weights(filepath)

    def load_network(self, filepath):
        self.network.load_weights(filepath)

    def build_q_network(self):
        model_input = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1))
        x = Conv2D(32, (8, 8), strides=4, activation='relu')(model_input)
        x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(self.action_space.n)(x)

        model = Model(model_input, x)
        model.compile(Adam(self.learning_rate), loss=tf.keras.losses.Huber())

        return model

    @staticmethod
    def process_frame(frame):
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[36:36 + 150, 10:10 + 150]
        frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        frame = frame.reshape((*TARGET_SIZE, 1))
        frame = frame.reshape((-1, TARGET_SIZE[0], TARGET_SIZE[1], 1))
        return frame


class Buffer:
    def __init__(self):
        self.buffer_frames = []
        self.buffer_next_frames = []
        self.buffer_targets_q = []
        self.buffer_actions = []

    def add_experience(self, frame, next_frame, target_q, action):
        self.buffer_frames.append(frame)
        self.buffer_next_frames.append(next_frame)
        self.buffer_targets_q.append(target_q)
        self.buffer_actions.append(action)

    def clear(self):
        self.buffer_frames = []
        self.buffer_next_frames = []
        self.buffer_targets_q = []
        self.buffer_actions = []

    def size(self):
        return len(self.buffer_frames)


class Algorithm(Enum):
    Q_LEARNING = 'Q_LEARNING'
    SARSA = 'SARSA'
