from .utils import *
from .blocks import *
import tensorflow as tf


def Generator(filters=64, n_res_blocks=16):
    input_tensor = tf.keras.layers.Input(shape=(64, 64, 3))
    x = tf.keras.layers.Lambda(normalize)(input_tensor)
    x = tf.keras.layers.Conv2D(filters, 9, padding='same')(x)
    x = x_sec = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    for _ in range(n_res_blocks):
        x = Residual(x, filters)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x_sec, x])
    x = Upsample(x, filters * 4)
    x = Upsample(x, filters * 4)
    x = tf.keras.layers.Conv2D(3, 9, padding='same', activation='tanh')(x)
    output_tensor = tf.keras.layers.Lambda(denormalize)(x)
    return tf.keras.Model(input_tensor, output_tensor)