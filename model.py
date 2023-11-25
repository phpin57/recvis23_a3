import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights


nclasses = 250

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Model1(nn.Module):
    def __init__(self, backbone):
        super(Model1, self).__init__()
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


class FineTunedNet2(nn.Module):
    def __init__(self, num_classes=250, fc_hidden_size=1024, dropout_prob=0.5):
        super(FineTunedNet2, self).__init__()
        
        # Load the pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)
        
        # Remove the last fully connected layer of ResNet
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom fully connected layers
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(fc_hidden_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x