import tensorflow as tf


def normalize_0_1(x):
    return x / 255.0


def normalize_1_1(x):
    return x / 127.5 - 1


def denormalize_0_1(x):
    return x * 255.0


def denormalize_1_1(x):
    return (x + 1) * 127.5


def pixel_shuffle(scale):
    '''Reference: https://arxiv.org/abs/1609.05158'''
    return lambda x: tf.nn.depth_to_space(x, scale)