import torch
from torchvision import models
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(Classifier, self).__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vgg19(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[0].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)