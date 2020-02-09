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


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (
            tf.image.flip_left_right(lr_img),
            tf.image.flip_left_right(hr_img)
        )
    )


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(
        shape=(), maxval=4,
        dtype=tf.int32
    )
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)