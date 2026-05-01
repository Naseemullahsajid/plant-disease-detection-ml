# src/train.py
import torch
import yaml
import torch.nn as nn
import torch.optim as optim

from data_loader import get_dataloaders
from model import SimpleCNN, get_resnet
from engine import train_one_epoch, validate
from earlystop import EarlyStopping, save_model

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = config["device"]

train_loader, val_loader, _, classes = get_dataloaders(batch_size=config["batch_size"])
num_classes = len(classes)
 
# Model selection
if config["model_name"] == "cnn":
    model = SimpleCNN(num_classes)
else:
    model = get_resnet(num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

early_stopper = EarlyStopping(patience=3)

for epoch in range(config["epochs"]):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if early_stopper.step(val_loss):
        print("⛔ Early stopping triggered")
        break

    save_model(model)  