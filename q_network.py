import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_q_network(n_actions, learning_rate=0.01, input_shape=(84, 84), history_length=1):
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)
    x = Conv2D(32, (8, 8), strides=4, activation='relu')(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu')(x)
    x = Conv2D(1024, (7, 7), strides=1, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(n_actions)(x)

    model = Model(model_input, x)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model
