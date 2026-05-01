# src/check_data.py
import os

def count_images(folder):
    total = 0
    for cls in os.listdir(folder):
        total += len(os.listdir(os.path.join(folder, cls)))
    return total

print("Train:", count_images("data/train"))
print("Val:", count_images("data/val"))
print("Test:", count_images("data/test"))