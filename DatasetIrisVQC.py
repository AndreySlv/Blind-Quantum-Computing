# pip install matplotlib qiskit qiskit-machine-learning scikit-learn torch

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from quantumnet.components import Network, Logger

import torch
import numpy as np
import random
import collections

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)

rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)
rede.draw()
Logger.activate(Logger)

iris = load_iris()
X = iris.data
y = iris.target
X = X[y != 2]
y = y[y != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print("Distribuição das classes:")
print("Treino:", collections.Counter(y_train))
print("Teste:", collections.Counter(y_test))

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_qubits = X_train.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)
sampler = Sampler()

num_epochs = 5
for epoch in range(num_epochs):
    print(f"\nÉpoca {epoch + 1}")

    # Inicializa VQC e treina
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        sampler=sampler
    )
    vqc.fit(X_train, y_train)

    # Obtém os pesos treinados
    trained_weights = vqc._fit_result.x

    # Monta o circuito final com os pesos aprendidos
    final_circuit = QuantumCircuit(num_qubits)
    final_circuit.compose(feature_map, inplace=True)
    final_circuit.compose(ansatz.assign_parameters(trained_weights), inplace=True)

    # Envia o circuito à rede BQC
    quantum_circuit = final_circuit
    circuit_depth = quantum_circuit.depth()

    rede.application_layer.run_app(
        "AC_BQC",
        alice_id=6,
        bob_id=0,
        num_qubits=num_qubits,
        scenario=2,
        circuit_depth=circuit_depth
    )

    print(f"Circuito com profundidade {circuit_depth} enviado para rede BQC.")

# Avaliação final com o último VQC treinado, dentro do vqc (encapsulado)
y_pred = vqc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n MÉTRICAS FINAIS:")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
