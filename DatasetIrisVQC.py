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
import collections
from quantumnet.components import Network, Logger

iris = load_iris()
X = iris.data
y = iris.target

X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Distribuição das classes:")
print("Treino:", collections.Counter(y_train))
print("Teste :", collections.Counter(y_test))

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_qubits = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

from qiskit.quantum_info import SparsePauliOp
observable = SparsePauliOp("Z" * num_qubits)

from qiskit.primitives import Estimator
#from qiskit.algorithms.gradients import ParamShiftEstimatorGradient

estimator = Estimator()
#gradient = ParamShiftEstimatorGradient(estimator)

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    observables=observable,
    estimator=estimator,
    input_gradients=True
)

model = TorchConnector(qnn)

rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)
rede.start_eprs(num_eprs=10)
Logger.activate(Logger)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.BCELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print("\nINICIANDO TREINAMENTO VQC (manualmente)...")

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(torch.sigmoid(outputs.squeeze()), y_train_tensor)
    loss.backward()
    optimizer.step()

    rede.start_eprs(num_eprs=5)

    # Extrai pesos aprendidos
    trained_weights = model.weight.detach().numpy()

    # Circuito final para envio para rede
    qc_treinado = QuantumCircuit(num_qubits)
    qc_treinado.compose(feature_map, inplace=True)
    qc_treinado.compose(ansatz.assign_parameters(trained_weights), inplace=True)

    # Envio
    try:
        rede.application_layer.run_app(
            "AC_BQC",
            alice_id=6,
            bob_id=0,
            num_qubits=num_qubits,
            scenario=2,
            circuit_depth=qc_treinado.depth(),
            circuit=qc_treinado
        )
        print(f"[Epoch {epoch+1}] Circuito enviado. Profundidade: {qc_treinado.depth()}")
    except Exception as e:
        print(f"[Epoch {epoch+1}] Erro ao enviar circuito: {str(e)}")

    print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")

with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    y_pred = torch.sigmoid(outputs).squeeze().round().numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMÉTRICAS FINAIS:")
print(f"Acurácia : {accuracy:.4f}")
print(f"Precisão : {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
