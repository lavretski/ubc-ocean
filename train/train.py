import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from pathlib import Path

image_size = (180, 180)
batch_size = 128
validation_split = 0.2
random_seed = 42

def get_labels(image_dir: str, csv_file: str) -> list[str]:
    df = pd.read_csv(csv_file)
    image_id_col = "image_id"
    df[image_id_col] = df[image_id_col].astype('str')
    df[image_id_col] = df[image_id_col] + "_thumbnails"
    image_files = [f.stem for f in Path(image_dir).glob('*.png')]
    df = df[df[image_id_col].isin(image_files)]
    return df[image_id_col].tolist()

def train(train_data_dir: str, test_data_dir: str, train_csv_file: str,
          test_csv_file: str) -> None:
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data_dir,
        labels = get_labels(train_data_dir, train_csv_file),
        validation_split=validation_split,
        subset="both",
        seed=random_seed,
        image_size=image_size,
        batch_size=batch_size)