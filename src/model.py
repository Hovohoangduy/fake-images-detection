import torch
from torchvision import models
import torch.nn as nn
from transformers import ViTForImageClassification, SwinForImageClassification

class Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        model_name = "microsoft/swin-base-patch4-window7-224"
        if pretrained:
            self.model = SwinForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SwinForImageClassification.config_class.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SwinForImageClassification(config)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:  # chỉ fine-tune head
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits