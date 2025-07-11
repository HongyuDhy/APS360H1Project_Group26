import torch
import torch.nn as nn
from torchvision import models

# pretrained VGG16
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(self.num_features, 4)

    def forward(self, x):
        out = self.model(x)
        return out

def pretrained_vgg16():
    return VGG16()

# pretrained ResNet50
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 4)

    def forward(self, x):
        out = self.model(x)
        return out

def pretrained_resnet50():
    return ResNet50()

# pretrained InceptionV3
class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 4)

    def forward(self, x):
        out = self.model(x)
        return out.logits if hasattr(out, 'logits') else out

def pretrained_inceptionv3():
    return InceptionV3()

