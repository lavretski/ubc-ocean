import numpy as np
from keras.models import load_model


def inference(model_file: str, image: np.ndarray) -> str:
    model = load_model(model_file)
