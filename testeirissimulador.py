# INSTALE AS DEPENDÊNCIAS
# pip install qiskit qiskit-machine-learning scikit-learn torch matplotlib

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

# =========================
# PARTE 1 — QNN COM IRIS
# =========================

# Carrega o dataset Iris (binário)
iris = load_iris()
X = iris.data
y = iris.target

# Reduz para duas classes
X = X[y != 2]
y = y[y != 2]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalização
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Número de qubits
num_qubits = X_train.shape[1] 

feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

model = TorchConnector(qnn)

# Dados para PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Otimizador e função de perda
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = torch.nn.BCELoss()

# Treinamento
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_func(torch.sigmoid(output.squeeze()), y_train_tensor)
    loss.backward()
    optimizer.step()

# Avaliação
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
    y_pred = torch.sigmoid(outputs).round().numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Resultado QNN com Estimator:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")


# ==================================================
# PARTE 2 — ENVIANDO O CIRCUITO QNN NA REDE BQC
# ==================================================

# Instala se não tiver
# pip install quantumnet

import random
import matplotlib.pyplot as plt
from quantumnet.components import Network, Logger

# Cria a rede quântica simulada
rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)  # 8 hosts, 3x3 grid

Logger.activate(Logger)

# Gerar um circuito aleatório (opcional, só para testar)
# quantum_circuit, num_qubits, circuit_depth = rede.generate_random_circuit(num_qubits=6, num_gates=20)

# ✅ Usar o circuito do QNN que treinamos

# Calcular profundidade do circuito treinado
circuit_depth = qc.depth()

print(f"Circuito do QNN tem profundidade: {circuit_depth}")

# Executa o protocolo Childs BQC simulando enviar o circuito do QNN
rede.application_layer.run_app(
    app_name="AC_BQC",
    alice_id=6,         # Escolha do nó Alice
    bob_id=0,           # Escolha do nó Bob (servidor quântico)
    num_qubits=num_qubits,
    scenario=1,         # Pode ser 1, 2, 3 dependendo do protocolo definido na sua rede
    circuit_depth=circuit_depth
)
