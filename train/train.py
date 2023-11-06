import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from train.models import make_model
from train.tools import get_labels, check_gpu, BalancedSparseCategoricalAccuracy
from tools import cancer_to_number


def train(data_dir: str, csv_file: str,
          image_size: tuple[int, int],
          batch_size: int, validation_split: int,
          random_seed: int, epochs: int, lr: float,
          num_classes: int, save_model_path: str,
          rescale_multiplier: float) -> None:
    labels = get_labels(data_dir, csv_file)
    integer_labels = [cancer_to_number[label] for label in labels]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=integer_labels,
        validation_split=validation_split,
        subset="both",
        seed=random_seed,
        batch_size=batch_size,
        label_mode='int')

    data_aug_train = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                       layers.Rescaling(rescale_multiplier),
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

    model = make_model(image_size + [3], num_classes)

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=[BalancedSparseCategoricalAccuracy()])

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds)

    model.save(save_model_path)

main = train