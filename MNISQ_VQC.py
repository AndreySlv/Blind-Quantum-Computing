# pip install --upgrade qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms scikit-learn matplotlib numpy

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils.loss_functions import CrossEntropyLoss
from qiskit_algorithms.optimizers import ADAM

from qiskit_aer import AerSimulator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from quantumnet.components import Network, Logger

# Inicializa rede quântica do seu ambiente
rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)
Logger.activate(Logger)

def enviar_circuito_por_epoca(circuito, epoch, num_qubits, circuit_depth):
    print(f"[Epoch {epoch+1}] Enviando circuito para a rede...")
    try:
        rede.application_layer.run_app(
            "AC_BQC",
            alice_id=6,
            bob_id=0,
            num_qubits=num_qubits,
            scenario=2,
            circuit_depth=circuit_depth,
            circuito=circuito
        )
        print(f"[Epoch {epoch+1}] Envio concluído.")
    except Exception as e:
        print(f"[Epoch {epoch+1}] Erro ao enviar circuito: {str(e)}")

# --- Preparação dos dados ---

simulator = AerSimulator(method='statevector')
path = "base_test_mnist_784_f90/qasm/"
file_list = sorted(os.listdir(path))[:20]  # 10 de cada classe, por exemplo

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
            features = features[:4]  # só 4 features para simplificar
            states.append(features)
            labels.append(0 if i < 10 else 1)  # classe 0 para primeiros 10, classe 1 para próximos 10
    except Exception as e:
        print(f"Erro ao processar {file_name}: {str(e)}")

X = np.array(states)
y = np.array(labels)

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Definindo o circuito quântico ---

num_qubits = 4
feature_map = ZZFeatureMap(num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

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

# --- Modelo VQC ---

vqc = NeuralNetworkClassifier(
    neural_network=qnn,
    loss=CrossEntropyLoss(),
    optimizer=ADAM(maxiter=25),
    callback=lambda weights, loss, step: enviar_circuito_por_epoca(
        circuito=feature_map.compose(ansatz.assign_parameters(weights)),
        epoch=step,
        num_qubits=num_qubits,
        circuit_depth=feature_map.compose(ansatz).depth()
    )
)

print("\nTREINANDO VQC...")
vqc.fit(X_train, y_train)

# --- Avaliação ---

print("\nCALCULANDO MÉTRICAS...")

y_pred = vqc.predict(X_test)

print("Classes verdadeiras únicas:", np.unique(y_test))
print("Classes previstas únicas:", np.unique(y_pred))

# Garantir inteiros (0 ou 1)
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

accuracy = accuracy_score(y_test, y_pred)

# Usar 'weighted' para métricas, para evitar erros com multiclass
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMÉTRICAS FINAIS:")
print(f"Acurácia : {accuracy:.4f}")
print(f"Precisão : {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nResultados Detalhados:")
print("Entradas:", X_test)
print("Saídas Previstas:", y_pred)
print("Saídas Reais:    ", y_test)
