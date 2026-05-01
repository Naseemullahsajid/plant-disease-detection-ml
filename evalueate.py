# src/evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from data_loader import get_dataloaders
from model import get_resnet

device = "cpu"

# Load data
_, _, test_loader, classes = get_dataloaders()

# Load model
model = get_resnet(num_classes=len(classes))
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ✅ Classification Report
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ✅ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure()
sns.heatmap(cm, annot=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()