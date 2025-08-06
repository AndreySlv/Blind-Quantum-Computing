# INSTALE AS DEPEDÊNCIAS
# pip install --upgrade qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn matplotlib numpy

import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_algorithms.optimizers import COBYLA

from qiskit_aer import AerSimulator

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 40
random.seed(SEED)
np.random.seed(SEED)

try:
    from qiskit_algorithms.utils import algorithm_globals
    algorithm_globals.random_seed = SEED
except ImportError:
    print("Não foi possível configurar a semente global do Qiskit, continuando...")

def enviar_circuito_por_epoca(circuito, epoch, num_qubits, circuit_depth):
    print(f"[Epoch {epoch+1}] Circuito com {num_qubits} qubits e profundidade {circuit_depth}")

simulator = AerSimulator(method='statevector')
path = "base_test_mnist_784_f90/qasm/"
file_list = sorted(os.listdir(path))[:20]  

states = []
labels = []

for i, file_name in enumerate(file_list):
    try:
        full_path = os.path.join(path, file_name)
        with open(full_path) as f:
            qasm = f.read()
            qc = QuantumCircuit.from_qasm_str(qasm)
            qc.save_statevector()
            compiled = transpile(qc, simulator)
            job = simulator.run(compiled)
            result = job.result()
            state = result.data(0)['statevector']
            features = np.abs(state) ** 2
            features = features[:4]  
            states.append(features)
            labels.append(0 if i < 10 else 1)  
    except Exception as e:
        print(f"Erro ao processar {file_name}: {str(e)}")

X = np.array(states)
y = np.array(labels)

if len(X) == 0:
    raise ValueError("Nenhum dado foi carregado corretamente!")

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

num_qubits = X.shape[1]

feature_map = ZZFeatureMap(num_qubits, reps=10, entanglement="full")
ansatz = RealAmplitudes(num_qubits, reps=10, entanglement="full")

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

estimator = Estimator()

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
    input_gradients=True
)

vqc = NeuralNetworkClassifier(
    neural_network=qnn,
    loss=CrossEntropyLoss(),
    optimizer=COBYLA(maxiter=50),
    warm_start=True,
    callback=lambda weights, loss, step: enviar_circuito_por_epoca(
        circuito=feature_map.compose(ansatz.assign_parameters(weights)),
        epoch=step,
        num_qubits=num_qubits,
        circuit_depth=feature_map.compose(ansatz).depth()
    )
)

print("\nTREINANDO VQC...")
vqc.fit(X_train, y_train)

print("\nCALCULANDO MÉTRICAS...")

y_pred = vqc.predict(X_test)

y_pred = np.round(y_pred).astype(int)
y_pred = np.clip(y_pred, 0, 1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("\nMÉTRICAS FINAIS:")
print(f"Acurácia : {accuracy:.4f}")
print(f"Precisão : {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nResultados Detalhados:")
print("Entradas:", X_test)
print("Saídas Previstas:", y_pred)
print("Saídas Reais:    ", y_test)