import torch
import numpy as np
import os
from quantum_models import DQNNModel, QRNNModel
from classical_models import MLPModel, RNNModel
from trainer import Trainer
from analyzer import analyze_and_save

# --- Ensure the data path is correct ---
data_path = os.path.join(os.path.dirname(__file__), "../data/sunspot_sequences.npz")
data_path = os.path.abspath(data_path)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

# --- Load data ---
data = np.load(data_path)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.float32).view(-1, 1)
seq_length = int(data["seq_length"])

# --- Hyperparameter grids ---
num_qubits_list = [2, 4, 6]
num_layers_list = [1, 2, 3]
ansatz_list = ["reset", "no_reset"]

model_configs = []

'''# --- Model configurations ---
model_configs = [
    {
        "name": "DQNN",
        "model": DQNNModel(seq_length=seq_length, num_qubits=4, num_layers=2)
    },
    {
        "name": "QRNN_Reset",
        "model": QRNNModel(num_qubits=4, num_qubits_data=1, ansatz="reset")
    },
    {
        "name": "QRNN_No_Reset",
        "model": QRNNModel(num_qubits=4, num_qubits_data=1, ansatz="no_reset")
    },
    {
        "name": "MLP", 
        "model": MLPModel(seq_length=seq_length, hidden_size=64)
    },
    {   "name": "RNN", 
        "model": RNNModel(seq_length=seq_length, hidden_size=32)
    },
]'''

model_configs = []

# --- Generate quantum model configs dynamically ---
for nq in num_qubits_list:
    for nl in num_layers_list:
        model_configs.append({
            "name": f"DQNN_q{nq}_l{nl}",
            "model": DQNNModel(seq_length=seq_length, num_qubits=nq, num_layers=nl)
        })

for nq in num_qubits_list:
    for ansatz in ansatz_list:
        model_configs.append({
            "name": f"QRNN_q{nq}_{ansatz}",
            "model": QRNNModel(num_qubits=nq, num_qubits_data=1, ansatz=ansatz)
        })

# --- Training and analysis loop ---
for cfg in model_configs:
    model_name = cfg["name"]
    model = cfg["model"]

    print(f"\n{'='*70}")
    print(f"Starting training: {model_name}")
    print(f"{'='*70}")

    trainer = Trainer(model, lr=0.01, batch_size=32)
    train_losses, val_losses, _ = trainer.fit(
        X_train, y_train, X_val, y_val,
        max_epochs=500, patience=30
    )

    # Store training/validation losses for analysis
    trainer.train_losses = train_losses
    trainer.val_losses = val_losses

    # Create separate result directory for each model
    save_dir = os.path.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Analyze and save ---
    analyze_and_save(
        model,            
        trainer,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        save_dir=save_dir,
        model_name=model_name
    )

print("\nTraining completed for all models!")

