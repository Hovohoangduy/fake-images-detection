import torch
from torchvision import models
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(Classifier, self).__init__()
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet152(weights=weights)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)