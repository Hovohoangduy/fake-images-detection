import torch
from torchvision import models
import torch.nn as nn
from transformers import ViTForImageClassification, SwinForImageClassification, DeiTForImageClassification

class Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        model_name = "facebook/deit-base-distilled-patch16-224"
        if pretrained:
            self.model = DeiTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = DeiTForImageClassification.config_class.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = DeiTForImageClassification(config)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "classifier" not in name:  # chỉ fine-tune head
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits