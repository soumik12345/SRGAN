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


def _random_crop(image, height, width):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,),
        maxval=image_dims[0] - height,
        dtype=tf.int32
    )[0]
    offset_w = tf.random.uniform(
        shape=(1,),
        maxval=image_dims[1] - width,
        dtype=tf.int32
    )[0]
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_h, offset_width=offset_w,
        target_height=H, target_width=W
    )
    return image


def random_crop(lr_image, hr_image):
    lr_crop = _random_crop(lr_image, LR_SHAPE[0], LR_SHAPE[1])
    hr_crop = _random_crop(hr_image, HR_SHAPE[0], HR_SHAPE[1])
    return lr_crop, hr_crop