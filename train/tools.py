from keras import backend as K
import pandas as pd
from pathlib import Path
import tensorflow as tf

def get_labels(image_dir: str, csv_file: str) -> list[str]:
    df = pd.read_csv(csv_file)
    image_id_col = "image_id"
    label_col = "label"
    df[image_id_col] = df[image_id_col].astype('str')
    df[image_id_col] = df[image_id_col] + "_thumbnail"
    image_files = [f.stem for f in Path(image_dir).glob('*.png')]
    df = df[df[image_id_col].isin(image_files)]
    return df[label_col].tolist()


def balanced_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    recall = K.mean(K.equal(y_true, y_pred)[K.equal(y_true, 1)])
    specificity = K.mean(K.equal(y_true, y_pred)[K.equal(y_true, 0)])
    
    return (recall + specificity) / 2


def check_gpu() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        print("GPU is available")
        print("Physical devices:", physical_devices)
    else:
        print("GPU is not available")