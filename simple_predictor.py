import pandas as pd
import fire
import os

COMPETITION_NAME = "UBC-OCEAN"

def simple_predictor(train_csv_file: str, test_csv_file: str, 
                     submission_csv_file: str, submission_message: str = "") -> None:
    df_train = pd.read_csv(train_csv_file)
    most_frequent = df_train['label'].value_counts().idxmax()

    df_test = pd.read_csv(test_csv_file)
    df_test.drop(["image_width", "image_height"], axis=1, inplace=True)
    df_test['label'] = most_frequent
    df_test.to_csv(submission_csv_file, index=False)

    os.system(f'kaggle competitions submit -c {COMPETITION_NAME} -f {submission_csv_file} -m {submission_message}')
    


if __name__ == "__main__":
    fire.Fire(simple_predictor)