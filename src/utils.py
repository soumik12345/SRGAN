from configs import *
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


def random_crop(lr_img, hr_img, hr_crop_size=256, scale=2):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]
    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    hr_w = lr_w * scale
    hr_h = lr_h * scale
    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]
    return lr_img_cropped, hr_img_cropped