import pandas as pd
from pathlib import Path
import tensorflow as tf
from train.tools import read_image
import numpy as np
from tools import number_to_cancer


def make_submission(model: tf.keras.Model, model_weights: str,
                    test_csv_file: str, test_data_dir: str, 
                    submission_csv_file: str, use_thumbnails: bool,
                    image_size: tuple[int, int]) -> None:
    df_test = pd.read_csv(test_csv_file)
    labels = []

    for image_id in df_test['image_id']:
        path = Path(test_data_dir) / str(image_id)
        suffix = "_thumbnail" if use_thumbnails else ""

        model.load_weights(model_weights)
        proc_image = read_image(f"{path}{suffix}.png", image_size)
        proc_image = proc_image[None, ...]
        prediction = model.predict(proc_image)

        model_output = np.argmax(prediction, axis=-1)[0]        
        labels.append(number_to_cancer[model_output])

    df_test.drop(["image_width", "image_height"], axis=1, inplace=True)
    df_test['label'] = labels
    df_test.to_csv(submission_csv_file, index=False)

main = make_submission
