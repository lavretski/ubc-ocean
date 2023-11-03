import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .models import make_model
from keras import backend as K
from .tools import get_labels, balanced_accuracy, check_gpu

image_size = (180, 180)
batch_size = 128
validation_split = 0.2
random_seed = 42
epochs = 25
lr = 1e-3


def train(train_data_dir: str, test_data_dir: str, train_csv_file: str,
          test_csv_file: str) -> None:
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        labels=get_labels(train_data_dir, train_csv_file),
        validation_split=validation_split,
        subset="both",
        seed=random_seed,
        image_size=image_size,
        batch_size=batch_size)

    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal_and_vertical")])

    model = make_model(image_size + (3,))

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=[balanced_accuracy])

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds)

    check_gpu()