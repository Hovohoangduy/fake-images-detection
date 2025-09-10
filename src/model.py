import torch
from torchvision import models
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Classifier, self).__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)