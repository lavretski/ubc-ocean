from keras.models import load_model
import pandas as pd

def make_submission(model_file: str, csv_file: str) -> None:
    df = pd.read_csv(csv_file)
    