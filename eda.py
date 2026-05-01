# src/eda.py
import os
import matplotlib.pyplot as plt
from collections import Counter

DATA_DIR = "data/train"

class_counts = {}

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    count = len(os.listdir(class_path))
    class_counts[class_name] = count

# Print summary
print("Number of classes:", len(class_counts))
print("Total images:", sum(class_counts.values()))

# Plot class distribution
plt.figure()
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.tight_layout()
plt.show()