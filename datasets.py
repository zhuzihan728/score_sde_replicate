import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import numpy as np


def get_data_scaler(centered):
    if centered:
        return lambda x: x * 2.0 - 1.0
    return lambda x: x


def get_data_inverse_scaler(centered):
    if centered:
        return lambda x: (x + 1.0) / 2.0
    return lambda x: x


def get_dataset(config):
    batch_size   = config.training.batch_size
    image_size   = config.data.image_size
    dataset_name = config.data.dataset

    def preprocess_cifar10(example):
        image = tf.cast(example['image'], tf.float32)
        if config.data.uniform_dequantization:
            image = (image + tf.random.uniform(tf.shape(image))) / 256.0
        else:
            image = image / 255.0
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_celeba(example):
        image = tf.cast(example['image'], tf.float32)
        crop_size = 140
        shape = tf.shape(image)
        top  = (shape[0] - crop_size) // 2
        left = (shape[1] - crop_size) // 2
        image = tf.image.crop_to_bounding_box(image, top, left, crop_size, crop_size)
        image = tf.image.resize(image, [image_size, image_size], method='bilinear', antialias=True)
        if config.data.uniform_dequantization:
            image = (image + tf.random.uniform(tf.shape(image))) / 256.0
        else:
            image = image / 255.0
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_lsun(example):
        image = tf.cast(example['image'], tf.float32)
        shape = tf.shape(image)
        min_side = tf.minimum(shape[0], shape[1])
        image = tf.image.crop_to_bounding_box(image, (shape[0] - min_side) // 2, (shape[1] - min_side) // 2, min_side, min_side)
        image = tf.image.resize(image, [image_size, image_size], method='bilinear', antialias=True)
        if config.data.uniform_dequantization:
            image = (image + tf.random.uniform(tf.shape(image))) / 256.0
        else:
            image = image / 255.0
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    if dataset_name.upper() == 'CIFAR10':
        ds_builder  = tfds.builder('cifar10')
        preprocess  = preprocess_cifar10
        train_split = 'train'
        eval_split  = 'test'
    elif dataset_name.lower() in ('celeb_a', 'celeba'):
        ds_builder  = tfds.builder('celeb_a')
        preprocess  = preprocess_celeba
        train_split = 'train'
        eval_split  = 'validation'
    elif dataset_name.lower().startswith('lsun'):
        ds_builder  = tfds.builder(dataset_name)
        preprocess  = preprocess_lsun
        train_split = 'train'
        eval_split  = 'validation'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_builder.download_and_prepare()

    train_ds = (ds_builder.as_dataset(split=train_split)
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(10000)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    eval_ds = (ds_builder.as_dataset(split=eval_split)
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=True)
               .repeat()
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, eval_ds
