import pandas as pd
import fire 

def analysis(csv_file: str) -> None:
    df = pd.read_csv(csv_file)
    print(df.head())

if __name__ == "__main__":
    fire.Fire(analysis)