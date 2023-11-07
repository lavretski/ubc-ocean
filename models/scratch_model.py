from models.model import Model
from keras.models import load_model
from keras import layers
import numpy as np
from tools import number_to_cancer
import keras


class ScratchModel(Model):
    def __init__(self, model_file: str, image_size: tuple[int, int],
                 rescale_multiplier: float):
        self._model = load_model(model_file, compile=False)
        self._preprocess_f = keras.Sequential([layers.Rescaling(rescale_multiplier),
                                               layers.Resizing(*image_size)])

    def predict(self, image: np.ndarray) -> str:
        proc_image = self._preprocess_f(image)
        proc_image = proc_image[None, ...]
        prediction = self._model.predict(proc_image)
        return self._postprocess(prediction)

    def _postprocess(self, model_output: np.ndarray):
        model_output = np.argmax(model_output, axis=-1)
        return [number_to_cancer[x] for x in model_output]