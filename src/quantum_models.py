import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/sunspot_sequences.npz")
data = np.load(os.path.normpath(data_path))

seq_length = int(data["seq_length"])


init_method = {"weights": lambda x: torch.nn.init.uniform_(x, 0, 2 * torch.pi)}

def d_qnn_circuit(num_qubits, num_layers, backend="default.qubit", diff_method="best"):
    dev = qml.device(backend, wires=num_qubits)

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(inputs, weights): 
        # Data encoding (RY rotations)
        for i in range(num_qubits):
            qml.RY(torch.pi * inputs[:, i], wires=i)
        # Variational layers
        for j in range(num_layers):
            # Rotation layer
            for i in range(num_qubits):
                qml.RX(weights[i, j, 0], wires=i)
                qml.RY(weights[i, j, 1], wires=i)
                qml.RZ(weights[i, j, 2], wires=i)
            # Entangling layer (CNOTs in a ring)
            for i in range (num_qubits -1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[num_qubits-1, 0])
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
    
    # Define weight shapes and initialization
    weight_shapes = {"weights": (num_qubits, num_layers, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

class DQNNModel(nn.Module):
    def __init__(self, seq_length, num_qubits, num_layers):
        super().__init__()
        self.input_layer = nn.Linear(seq_length, num_qubits, bias=False)
        self.q_layer = d_qnn_circuit(num_qubits, num_layers)
        self.output_layer = nn.Linear(num_qubits, 1)

    def forward(self, inputs):
        x = self.input_layer(inputs)
        q_out = self.q_layer(x)
        out = self.output_layer(q_out)
        return out    
    
    def _get_name(self):
        return "DQNN"

def qrnn_circuit(num_qubits, num_qubits_data, ansatz, backend="default.qubit", diff_method="best"):
    dev = qml.device(backend, wires=num_qubits)

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(inputs, weights):
        for i in range(seq_length):
            for j in range(num_qubits_data):
                qml.RY(torch.arccos(inputs[:, i]), wires=j)
            for j in range(num_qubits):
                qml.RX(weights[j, 0], wires=j)
                qml.RZ(weights[j, 1], wires=j)
                qml.RX(weights[j, 2], wires=j)
            for j in range(num_qubits - 1):
                qml.CNOT(wires=[j, j+1])
                qml.RZ(weights[j+1, 3], wires=j+1)
                qml.CNOT(wires=[j, j+1])
            qml.CNOT(wires=[num_qubits - 1, 0])
            qml.RZ(weights[0, 3], wires=0)
            qml.CNOT(wires=[num_qubits - 1, 0])

        if ansatz == "reset" and i < seq_length - 1:
            for j in range(num_qubits_data):
                qml.measure(wires=j, reset=True)
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits_data)]
    
    weight_shapes = {"weights": (num_qubits, 4)}
    return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

class QRNNModel(nn.Module):
    def __init__(self, num_qubits, num_qubits_data, ansatz):
        super().__init__()
        # If dimension > 1, reshape = 1 (else squeeze(-1) in forward)
        self.q_layer = qrnn_circuit(num_qubits, num_qubits_data, ansatz)
        self.output_layer = nn.Linear(num_qubits_data, 1)
    def forward(self, inputs):
        # TorchLayer tự xử lý batch, ta chỉ cần bỏ chiều dư
        x = inputs.squeeze(-1) if inputs.ndim > 2 else inputs
        x = self.q_layer(x)
        out = self.output_layer(x)
        return out
    
    def _get_name(self):
        return "QRNN"