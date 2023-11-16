import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from train.tools import (check_gpu, \
    BalancedSparseCategoricalAccuracy, read_image_train)
from tools import cancer_to_number
import pandas as pd
import numpy as np


def train(model: tf.keras.Model, data_dir: str, csv_file: str,
          image_size: tuple[int, int],
          batch_size: int, validation_split: int,
          random_seed: int, epochs: int, lr: float,
          save_weights_file: str,
          use_thumbnails: bool, crop_size_increment: int) -> None:
    df = pd.read_csv(csv_file)

    if use_thumbnails:
        df = df[df["is_tma"] == False]
    
    image_id_col = "image_id"
    label_col = "label"
    image_pathes_col = "image_pathes"

    df[image_pathes_col] = df[image_id_col].astype('str')
    df[image_pathes_col] = df[image_pathes_col].apply(lambda x: f"{data_dir}/{x}{'_thumbnail' if use_thumbnails else ''}.png")
    
    x = (
        tf.data.Dataset.from_tensor_slices(df[image_pathes_col].values)
        .map(lambda image_path: read_image_train(image_path, image_size, crop_size_increment), num_parallel_calls=tf.data.AUTOTUNE)
    )

    integer_labels = [cancer_to_number[label] for label in df[label_col].values]
    y = tf.data.Dataset.from_tensor_slices(integer_labels)

    ds = tf.data.Dataset.zip((x, y))

    val_len = int(len(df) * validation_split)
    
    val_ds = (
        ds
        .take(val_len)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_ds = (
        ds
        .skip(val_len)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    check_gpu()

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    class_weights = len(integer_labels) - np.bincount(integer_labels)
    class_weights = class_weights / np.sum(class_weights)
    class_weights = {idx: weight for idx, weight in enumerate(class_weights)}

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds,
              class_weight=class_weights)

    model.save_weights(save_weights_file)

main = train
