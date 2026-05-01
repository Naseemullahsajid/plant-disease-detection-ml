# src/utils.py
import torch
from collections import Counter

def compute_class_weights(dataset):
    counts = Counter(dataset.targets)
    total = sum(counts.values())

    weights = []
    for i in range(len(counts)):
        weights.append(total / counts[i])

    return torch.tensor(weights, dtype=torch.float)