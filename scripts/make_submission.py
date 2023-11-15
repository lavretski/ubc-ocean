from keras.models import load_model
import pandas as pd
from models.model import Model
from pathlib import Path
import cv2

def make_submission(model: Model, test_csv_file: str, test_data_dir: str, 
                    submission_csv_file: str, use_thumbnails: bool) -> None:
    df_test = pd.read_csv(test_csv_file)
    labels = []

    for image_id in df_test['image_id']:
        path = Path(test_data_dir) / str(image_id)
        suffix = "_thumbnail" if use_thumbnails else ""
        label = model.predict(f"{path}{suffix}.png")
        labels.append(label)

    df_test.drop(["image_width", "image_height"], axis=1, inplace=True)
    df_test['label'] = labels
    df_test.to_csv(submission_csv_file, index=False)

main = make_submission
