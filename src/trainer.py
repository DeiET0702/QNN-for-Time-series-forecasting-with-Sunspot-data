import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import shuffle

class Trainer:
    def __init__(self, model, lr=0.01, batch_size=32):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def train_one_epoch(self, X, y):
        self.model.train()
        X, y = shuffle(X, y)
        total_loss = 0
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i:i+self.batch_size]
            batch_y = y[i:i+self.batch_size]
            # Training step
            self.optimizer.zero_grad()
            pred = self.model(batch_X)
            loss = self.criterion(pred, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / (len(X) // self.batch_size)
    
    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
            loss = self.criterion(pred, y).item()
        return loss
    
    def fit(self, X_train, y_train, X_val, y_val, max_epochs=1000, patience=50):
        best_val_loss = float('inf')
        best_model = None
        no_improve = 0

        train_losses = []
        val_losses = []

        print(f"Starting training for model: {self.model._get_name()}")
        for epoch in range(max_epochs):
            train_loss = self.train_one_epoch(X_train, y_train)
            val_loss = self.evaluate(X_val, y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                no_improve = 0
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f} (improved)")
            else:
                no_improve += 1
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        self.model.load_state_dict(best_model)
        return train_losses, val_losses, best_model