from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def infer(self, image):
        pass
