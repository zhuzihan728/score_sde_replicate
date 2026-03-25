import tensorflow_datasets as tfds
import tensorflow as tf
import jax
import numpy as np


def get_data_scaler(centered):
    """Returns a function that scales image pixel values."""
    if centered:
        # Scale from [0, 1] → [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(centered):
    """Returns a function that reverses the scaling (for visualization)."""
    if centered:
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x

def get_dataset(config):
    """Load and preprocess a dataset.
    
    Supports: CIFAR10, celeb_a, lsun/bedroom, lsun/church_outdoor
    
    Returns:
        train_ds: Training dataset as a tf.data.Dataset
        eval_ds: Evaluation dataset as a tf.data.Dataset
    """
    batch_size = config.training.batch_size
    image_size = config.data.image_size
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
        # central_crop(img, 140)
        crop_size = 140
        shape = tf.shape(image)
        top = (shape[0] - crop_size) // 2
        left = (shape[1] - crop_size) // 2
        image = tf.image.crop_to_bounding_box(image, top, left, crop_size, crop_size)
        # Resize to target resolution
        image = tf.image.resize(image, [image_size, image_size],
                                method='bilinear', antialias=True)
        if config.data.uniform_dequantization:
            image = (image + tf.random.uniform(tf.shape(image))) / 256.0
        else:
            image = image / 255.0
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    def preprocess_lsun(example):
        image = tf.cast(example['image'], tf.float32)
        # Center crop to square, then resize
        shape = tf.shape(image)
        min_side = tf.minimum(shape[0], shape[1])
        image = tf.image.crop_to_bounding_box(
            image,
            (shape[0] - min_side) // 2,
            (shape[1] - min_side) // 2,
            min_side, min_side
        )
        image = tf.image.resize(image, [image_size, image_size],
                                method='bilinear', antialias=True)
        if config.data.uniform_dequantization:
            image = (image + tf.random.uniform(tf.shape(image))) / 256.0
        else:
            image = image / 255.0
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    # Select dataset and preprocessing
    if dataset_name.upper() == 'CIFAR10':
        ds_builder = tfds.builder('cifar10')
        preprocess = preprocess_cifar10
        train_split = 'train'
        eval_split = 'test'

    elif dataset_name.lower() == 'celeb_a':
        ds_builder = tfds.builder('celeb_a')
        preprocess = preprocess_celeba
        train_split = 'train'
        eval_split = 'validation'

    elif dataset_name.lower().startswith('lsun'):
        # e.g. 'lsun/bedroom' or 'lsun/church_outdoor'
        ds_builder = tfds.builder(dataset_name)
        preprocess = preprocess_lsun
        train_split = 'train'
        eval_split = 'validation'

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds_builder.download_and_prepare()

    train_ds = ds_builder.as_dataset(split=train_split)
    eval_ds = ds_builder.as_dataset(split=eval_split)

    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(10000)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    eval_ds = (eval_ds
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size, drop_remainder=True)
            #    .repeat()
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, eval_ds