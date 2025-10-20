import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from src.data import DatasetLoader
from src.model import Classifier

class Trainer:
    def __init__(self, model, train_loader, val_loader, train_size, val_size,
                 device, lr=1e-4, epochs=100, save_path="moire.pth"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_size = train_size
        self.val_size = val_size
        self.device = device
        self.epochs = epochs
        self.save_path = save_path

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.model.fc.parameters(), lr=lr)
        self.best_acc = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss, running_corrects = 0.0, 0
        total_batches = len(self.train_loader)

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)
            # if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            #     print(f"[Batch {batch_idx+1}/{total_batches}] "
            #           f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / self.train_size
        epoch_acc = running_corrects.double() / self.train_size
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds==labels.data)
            val_loss /= self.val_size
            val_acc = val_corrects.double() / self.val_size
            return val_loss, val_acc

    def train(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(F"Epoch [{epoch+1}/{self.epochs}] "
                f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                onnx_path = self.save_path.replace(".pth", ".onnx")
                dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                )
        print(f"Best val acc: {self.best_acc:.4f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train fake image detection model")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="datasets path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="moire.onnx", help="Save path model")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    dataset_loader = DatasetLoader(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader, val_loader, train_size, val_size = dataset_loader.get_dataloaders()
    model = Classifier(num_classes=2, pretrained=True, freeze_backbone=True)
    trainer = Trainer(model, train_loader, val_loader, train_size, val_size, device,
                      lr=args.lr, epochs=args.epochs, save_path=args.save_path)
    trainer.train()