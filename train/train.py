import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from train.tools import (get_image_paths, check_gpu, \
    BalancedSparseCategoricalAccuracy, \
    get_class_weights, StandardizationLayer)
from tools import cancer_to_number


def train(model: tf.keras.Model, data_dir: str, csv_file: str,
          image_size: tuple[int, int],
          batch_size: int, validation_split: int,
          random_seed: int, epochs: int, lr: float,
          save_model_path: str, rescale_multiplier: float,
          use_thumbnails: bool, use_tma: bool) -> None:
    image_pathes = get_image_paths(data_dir, csv_file, use_tma, use_thumbnails)

    num_samples = len(image_pathes)
    num_val_samples = int(num_samples * validation_split)
    num_train_samples = num_samples - num_val_samples
    x = (
        tf.data.Dataset.from_tensor_slices(image_pathes)
        .map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    )
    y = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((x, y))

    val_ds = (
        ds
        .take(num_val_samples)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_ds = (
        ds
        .skip(num_val_samples)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    data_aug_train = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                       layers.Rescaling(rescale_multiplier),
                                       #StandardizationLayer(),
                                       layers.Resizing(*image_size)])

    data_aug_val = keras.Sequential([layers.Rescaling(rescale_multiplier),
                                     layers.Resizing(*image_size)])

    train_ds = train_ds.map(lambda img, label: 
                                (data_aug_train(img), label),
                            num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda img, label:
                            (data_aug_val(img), label),
                        num_parallel_calls=tf.data.AUTOTUNE)
                        
    check_gpu()

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=[BalancedSparseCategoricalAccuracy(), "accuracy"])

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds,
              class_weight=get_class_weights(data_dir,
                                              csv_file, 
                                              use_thumbnails))

    model.save(save_model_path)

main = train
