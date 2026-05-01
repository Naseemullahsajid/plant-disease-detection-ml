# src/test_loader.py
from data_loader import get_dataloaders

train_loader, val_loader, test_loader, classes = get_dataloaders()

print("Classes:", classes)

for images, labels in train_loader:
    print("Batch shape:", images.shape)
    print("Labels:", labels)
    break