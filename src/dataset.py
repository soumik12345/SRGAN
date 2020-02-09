import os
from .utils import *
from configs import *
import tensorflow as tf


AUTOTUNE = tf.python.data.experimental.AUTOTUNE


def load_data(hr_file, lr_file):
    hr = tf.io.read_file(hr_file)
    hr = tf.image.decode_png(hr, channels=3)
    hr = tf.cast(hr, dtype=tf.float32)
    hr = tf.image.resize(images=hr, size=HR_SHAPE)
    lr = tf.io.read_file(lr_file)
    lr = tf.image.decode_png(lr, channels=3)
    lr = tf.cast(lr, dtype=tf.float32)
    lr = tf.image.resize(images=lr, size=LR_SHAPE)
    return hr, lr


def get_dataset(hr_path, lr_path, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((hr_path, lr_path))
    dataset = dataset.map(
        map_func=load_data,
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.map(
        map_func=random_flip,
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.map(
        map_func=random_rotate,
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=AUTOTUNE
    )
    return dataset