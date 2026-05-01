# src/split_data.py
import os
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE_DIR = "data"
DEST_DIR = "data/"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_end = int(len(images) * TRAIN_RATIO)
    val_end = train_end + int(len(images) * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, img_list in splits.items():
        split_dir = os.path.join(DEST_DIR, split, class_name)
        Path(split_dir).mkdir(parents=True, exist_ok=True)

        for img in img_list:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("✅ Data split complete")