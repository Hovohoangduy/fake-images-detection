from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

data_dir = "./datasets"

class DatasetLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transforms = {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        }
    
    def get_dataloaders(self):
        train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=self.data_transforms["train"])
        val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "val"), transform=self.data_transforms["val"])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return train_loader, val_loader, len(train_dataset), len(val_dataset)