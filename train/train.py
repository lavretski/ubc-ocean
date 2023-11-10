import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from train.tools import (get_labels, check_gpu, \
    BalancedSparseCategoricalAccuracy, \
    get_class_weights, StandardizationLayer)
from tools import cancer_to_number


def train(model: tf.keras.Model, data_dir: str, csv_file: str,
          image_size: tuple[int, int],
          batch_size: int, validation_split: int,
          random_seed: int, epochs: int, lr: float,
          save_model_path: str, rescale_multiplier: float,
          use_thumbnails: bool) -> None:
    labels = get_labels(data_dir, csv_file, use_thumbnails)
    integer_labels = [cancer_to_number[label] for label in labels]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=integer_labels,
        validation_split=validation_split,
        subset="both",
        seed=random_seed,
        batch_size=batch_size,
        label_mode='int')

    data_aug_train = keras.Sequential([#layers.RandomFlip("horizontal_and_vertical"),
                                       #layers.Rescaling(rescale_multiplier),
                                       layers.Resizing(*image_size),
                                       StandardizationLayer(),])

    data_aug_val = keras.Sequential([layers.Resizing(*image_size),
                                     StandardizationLayer(),
                                     #layers.Rescaling(rescale_multiplier)
                                     ])

    train_ds = train_ds.map(lambda img, label: 
                                (data_aug_train(img), label),
                            num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda img, label:
                            (data_aug_val(img), label),
                        num_parallel_calls=tf.data.AUTOTUNE)
                        
    check_gpu()

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds,
              class_weight=get_class_weights(data_dir,
                                             csv_file,
                                             use_thumbnails))

    model.save(save_model_path)

main = train
