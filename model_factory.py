"""Python file to instantite the model and the transform that goes with it."""
from model import Net, Model1, FineTunedNet2
from data import *
from torchvision import disable_beta_transforms_warning
from torchvision.models import resnet50, ResNet50_Weights

class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Model1(resnet50(ResNet50_Weights.IMAGENET1K_V2))
        if self.model_name[:8] == "resnet50":
            return Model1(resnet50(ResNet50_Weights.IMAGENET1K_V2))
        if self.model_name == "tuned_resnet":
            return FineTunedNet2()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet50":
            #return data_transforms_resnet
            return data_transforms
        if self.model_name == "resnet50_augm":
            return data_transforms_augmented
        if self.model_name == "tuned_resnet":
            return data_transforms_augmented
        else:
            return data_transforms

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
