import tensorflow as tf


bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mse = tf.keras.losses.MeanSquaredError()


def generator_loss(self, generator_output):
    return bce(tf.ones_like(generator_output), generator_output)


def discriminator_loss(self, hr_output, sr_output):
    hr_loss = bce(tf.ones_like(hr_output), hr_output)
    sr_loss = bce(tf.zeros_like(sr_output), sr_output)
    return hr_loss + sr_loss