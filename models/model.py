from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> list[str]:
        pass