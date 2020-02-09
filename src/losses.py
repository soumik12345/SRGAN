from .models import *
import tensorflow as tf


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mse = tf.keras.losses.MeanSquaredError()
vgg = vgg_model(5)


def generator_loss(self, generator_output):
    return bce(tf.ones_like(generator_output), generator_output)


def discriminator_loss(self, hr_output, sr_output):
    hr_loss = bce(tf.ones_like(hr_output), hr_output)
    sr_loss = bce(tf.zeros_like(sr_output), sr_output)
    return hr_loss + sr_loss


def content_loss(hr, sr):
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    sr_features = vgg(sr) / 12.75
    hr_features = vgg(hr) / 12.75
    return mse(hr_features, sr_features)