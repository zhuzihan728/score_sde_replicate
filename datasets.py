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
    
    Returns:
        train_ds: Training dataset as a tf.data.Dataset
        eval_ds: Evaluation dataset as a tf.data.Dataset
    """
    batch_size = config.training.batch_size
    image_size = config.data.image_size

    def preprocess(example):
        """Convert uint8 image to float32 in [0, 1] and apply augmentation."""
        image = tf.cast(example['image'], tf.float32) / 255.0
        
        # Resize if needed (CIFAR-10 is already 32x32, but CelebA isn't)
        image = tf.image.resize(image, [image_size, image_size])
        
        # Random horizontal flip
        if config.data.random_flip:
            image = tf.image.random_flip_left_right(image)
        
        # Uniform dequantization: add U[0, 1/256) noise
        # Converts discrete pixel values to continuous distribution
        if config.data.uniform_dequantization:
            image = image + tf.random.uniform(image.shape) / 256.0
        
        return image

    # Load CIFAR-10 from tensorflow_datasets
    ds_builder = tfds.builder(config.data.dataset)
    ds_builder.download_and_prepare()

    train_ds = ds_builder.as_dataset(split='train')
    eval_ds = ds_builder.as_dataset(split='test')

    # Apply preprocessing, shuffle, batch, prefetch
    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(10000)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    eval_ds = (eval_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size, drop_remainder=True)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    return train_ds, eval_ds