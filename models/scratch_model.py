from .models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras import layers
import numpy as np
from ..tools import number_to_cancer


class ScratchInferenceModel(Model):
    def __init__(self, model_file: str, image_size: tuple[int, int],
                 rescale_multiplier: float):
        self._model = load_model(model_file)
        self._preprocess_f = keras.Sequential([layers.Rescaling(rescale_multiplier),
                                          layers.Resizing(*image_size)])

    def predict(self, images: list[np.ndarray]) -> list[str]:
        proc_images = [self._preprocess_f(img) for img in images]
        proc_images = np.stack(proc_images, axis=0)
        predictions = self._model.predict(proc_images)
        return self._postprocess(predictions)

    def _postprocess(model_output: np.ndarray):
        model_output = np.argmax(model_output, axis=-1)
        return [number_to_cancer(x) for x in model_output]