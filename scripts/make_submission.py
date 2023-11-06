from keras.models import load_model
import pandas as pd
from .models.model import Model
from pathlibe import Path
import cv2


def make_submission(model: Model, test_csv_file: str, test_data_dir: str, submission_csv_file: str) -> None:
    df_test = pd.read_csv(test_csv_file)

    images = [cv2.imread(f"{Path(test_data_dir) / image_id}.png" 
                   for image_id in df['image_id'].tolist())]

    labels = model.predict(images)

    df_test.drop(["image_width", "image_height"], axis=1, inplace=True)
    df_test['label'] = labels
    df_test.to_csv(submission_csv_file, index=False)
    