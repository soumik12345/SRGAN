from .utils import *
from .blocks import *
import tensorflow as tf


def vgg_model(output_layer):
    vgg19 = tf.keras.applications.vgg19.VGG19(
        input_shape=[None, None, 3],
        include_top=False
    )
    return tf.keras.Model(
        vgg19.input,
        vgg19.layers[output_layer].output
    )


def Generator(input_shape, filters=64, n_res_blocks=16):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(normalize_0_1)(input_tensor)
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
    output_tensor = tf.keras.layers.Lambda(denormalize_1_1)(x)
    return tf.keras.Model(input_tensor, output_tensor)



def Discriminator(input_shape, filters=64):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(normalize_1_1)(input_tensor)
    x = discriminator_block(x, filters, bn=False)
    x = discriminator_block(x, filters, 2)
    x = discriminator_block(x, filters * 2)
    x = discriminator_block(x, filters * 2, 2)
    x = discriminator_block(x, filters * 4)
    x = discriminator_block(x, filters * 4, 2)
    x = discriminator_block(x, filters * 8)
    x = discriminator_block(x, filters * 8, 2)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(input_tensor, output_tensor)