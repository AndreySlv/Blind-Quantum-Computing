#INSTALE AS DEPENDÊNCIAS
#pip install qiskit qiskit-machine-learning scikit-learn torch matplotlib

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import collections
print("Distribuição das classes no y_train e y_test:")
print("Treino:", collections.Counter(y_train))
print("Teste:", collections.Counter(y_test))

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_qubits = X_train.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

print("Circuito usado no QML")

qnn = EstimatorQNN(
    circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
)
model = TorchConnector(qnn)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.BCELoss()

for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_func(torch.sigmoid(outputs.squeeze()), y_train_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
    y_pred = torch.sigmoid(outputs).round().numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nDesempenho do Modelo QML:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")

from quantumnet.components import Network, Logger

rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)

Logger.activate(Logger)

quantum_circuit = qc
num_qubits = quantum_circuit.num_qubits
circuit_depth = quantum_circuit.depth()

print(f"\nEnviando circuito para a rede:")
print(f"Qubits: {num_qubits} | Profundidade: {circuit_depth}")

# Executa o protocolo AC_BQC com o circuito do QML
resultado = rede.application_layer.run_app(
    "AC_BQC",
    alice_id=6,
    bob_id=0,
    num_qubits=num_qubits,
    scenario=1,
    circuit_depth=circuit_depth
)

print("\nResultado da comunicação via rede:")
print(resultado)
