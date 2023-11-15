from models.model import Model
from keras.models import load_model
from keras import layers
import numpy as np
from tools import number_to_cancer
import keras
from train.tools import read_image
from keras_cv.models import ResNetV2Backbone


class AritraModel(Model):
    def __init__(self, model_file: str, image_size: tuple[int, int]):
        custom_objects = {'ResNetV2Backbone': ResNetV2Backbone}
        self._model = load_model(model_file, custom_objects=custom_objects, compile=False)
        self._image_size = image_size

    def predict(self, image_path: str) -> str:
        proc_image = read_image(image_path, self._image_size)
        proc_image = proc_image[None, ...]
        prediction = self._model.predict(proc_image)
        return self._postprocess(prediction)

    def _postprocess(self, model_output: np.ndarray):
        model_output = np.argmax(model_output, axis=-1)[0]
        return number_to_cancer[model_output]