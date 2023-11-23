"""Python file to instantite the model and the transform that goes with it."""
from model import Net
from data import data_transforms


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = Net()
        self.transform = data_transforms

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
