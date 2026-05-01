# src/utils.py
import torch
import os

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


import torch
import os

def save_model(model, path="models/best_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)  
    torch.save(model.state_dict(), path)