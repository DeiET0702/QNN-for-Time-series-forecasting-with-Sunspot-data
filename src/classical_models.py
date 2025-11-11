import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/sunspot_sequences.npz")
data = np.load(os.path.normpath(data_path))

seq_length = int(data["seq_length"])

class MLPModel(nn.Module):
    def __init__(self, seq_length, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(seq_length, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class RNNModel(nn.Module):
    def __init__(self, seq_length, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        x = x.unsqueeze(-1)  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0) 
        out = self.fc(out[:, -1, :])  
        return out
    