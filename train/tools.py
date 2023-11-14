import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras.metrics import SparseCategoricalAccuracy
from tools import cancer_to_number
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras import layers


def check_gpu() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        print("GPU is available")
        print("Physical devices:", physical_devices)
    else:
        print("GPU is not available")


class BalancedSparseCategoricalAccuracy(SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


def read_image(image_path: str, image_size: tuple[int, int]) -> tf.Tensor:
    file = tf.io.read_file(image_path)
    image = tf.io.decode_png(file, 3)
    image = tf.image.resize(image, image_size)
    image = tf.image.per_image_standardization(image)
    return image