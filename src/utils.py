import tensorflow as tf


def normalize(x, mode='01'):
    '''Normalize tensor
    Params:
        x    -> Tensor
        mode -> '01' meaning [0, 1] or '-11' meaning [-1, 1]
    '''
    return x / 255.0 if mode == '01' else x / 127.5 - 1


def denormalize(x, mode='01'):
    '''DeNormalize tensor
    Params:
        x    -> Tensor
        mode -> '01' meaning [0, 1] or '-11' meaning [-1, 1]
    '''
    return x * 255.0 if mode == '01' else (x + 1) * 127.5


def pixel_shuffle(scale):
    '''Reference: https://arxiv.org/abs/1609.05158'''
    return lambda x: tf.nn.depth_to_space(x, scale)