import tensorflow as tf
from .utils import *


def Upsample(tensor, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(tensor)
    x = tf.keras.layers.Lambda(pixel_shuffle(2))(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    return x


def Residual(tensor, filters, momentum=0.8):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(tensor)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Add()([tensor, x])
    return x