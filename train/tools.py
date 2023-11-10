import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras.metrics import SparseCategoricalAccuracy
from tools import cancer_to_number
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras import layers


def get_labels(image_dir: str, csv_file: str, 
               use_thumbnails: bool) -> list[str]:
    df = pd.read_csv(csv_file)
    image_id_col = "image_id"
    label_col = "label"
    df[image_id_col] = df[image_id_col].astype('str')
    df[image_id_col] = df[image_id_col] + ("_thumbnail" if use_thumbnails else "")
    image_files = [f.stem for f in Path(image_dir).glob('*.png')]
    df = df[df[image_id_col].isin(image_files)]
    return df[label_col].tolist()


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

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)



def get_class_weights(image_dir: str, csv_file: str, 
                      use_thumbnails: bool) -> dict[int, float]:
    labels = get_labels(image_dir, csv_file, use_thumbnails)
    int_labels = [cancer_to_number[label] for label in labels]

    weights = class_weight.compute_sample_weight('balanced', int_labels)

    class_weights = {label: weights[i] for i, label in enumerate(int_labels)}

    return class_weights


class StandardizationLayer(layers.Layer):
    def call(self, inputs):
        return tf.map_fn(tf.image.per_image_standardization, inputs)
