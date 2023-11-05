from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> list[str]:
        pass