import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights



nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer of ResNet
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add your custom fully connected layer for classification
        self.fc = nn.Linear(resnet.fc.in_features, nclasses)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x