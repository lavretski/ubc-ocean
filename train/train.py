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
num_classes = 5

def train(train_data_dir: str, test_data_dir: str, train_csv_file: str,
          test_csv_file: str) -> None:
    labels = get_labels(train_data_dir, train_csv_file)
    label_to_index = dict((name, index) for index, name in enumerate(set(labels)))
    integer_labels = [label_to_index[label] for label in labels]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        labels=integer_labels,
        validation_split=validation_split,
        subset="both",
        seed=random_seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical')

    data_augmentation = keras.Sequential([layers.RandomFlip("horizontal_and_vertical")])

    train_ds = train_ds.map(lambda img, label: 
                                        (data_augmentation(img), 
                                         label),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    check_gpu()

    model = make_model(image_size + (3,), num_classes)

    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=[balanced_accuracy])

    model.fit(train_ds, epochs=epochs,
              validation_data=val_ds)
