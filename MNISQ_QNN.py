##
# INSTALE AS DEPENDÊNCIAS
# pip install qiskit qiskit-aer qiskit-machine-learning numpy matplotlib torch scikit-learn

import matplotlib
matplotlib.use('Agg')

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from quantumnet.components import Network, Logger

# Rede de comunicação quântica
rede = Network()
rede.set_ready_topology('grade', 8, 3, 3)
Logger.activate(Logger)

# Função de envio do circuito treinado
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
            circuit=circuito  # <-- AQUI ESTÁ O QUE FALTAVA!
        )
        print(f"[Epoch {epoch+1}] Envio concluído.")
    except Exception as e:
        print(f"[Epoch {epoch+1}] Erro ao enviar circuito: {str(e)}")


# Caminho dos arquivos QASM
path = "base_test_mnist_784_f90/qasm/"

def show_figure(pict, index=0):
    try:
        pict = np.asarray(pict, dtype=np.float64).ravel()
        if pict.max() > 0:
            pict = pict / pict.max()
        size = len(pict)
        dim = int(np.ceil(np.sqrt(size)))
        padded = np.zeros((dim * dim,))
        padded[:size] = pict
        plt.imshow(padded.reshape(dim, dim), cmap="Greys")
        plt.axis('off')
        plt.savefig(f"figura_{index}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Erro ao salvar figura {index}: {str(e)}")

def show_state_figure(statevector, index=0):
    try:
        if hasattr(statevector, 'data'):
            statevector = statevector.data
        statevector = np.asarray(statevector, dtype=np.complex128)
        probs = np.abs(statevector) ** 2
        if len(probs) < 784:
            padded = np.zeros(784)
            padded[:len(probs)] = probs
            probs = padded
        show_figure(probs, index=index)
    except Exception as e:
        print(f"Erro ao processar estado {index}: {str(e)}")

# Simulador
simulator = AerSimulator(method='statevector')

# Leitura dos arquivos QASM
file_list = sorted(os.listdir(path))[:20]  # 20 arquivos: 10 de cada classe
states = []
labels = []

for i, file_name in enumerate(file_list):
    try:
        full_path = os.path.join(path, file_name)
        with open(full_path) as f:
            qasm = f.read()
            qc = QuantumCircuit.from_qasm_str(qasm)
            qc.save_statevector()
            compiled_circuit = transpile(qc, simulator)
            job = simulator.run(compiled_circuit)
            result = job.result()
            state = result.data(0)['statevector']
            show_state_figure(state, index=i)
            features = np.abs(state)**2
            features = features[:4]  # Apenas 4 features
            states.append(features)
            labels.append(0 if i < 10 else 1)
    except Exception as e:
        print(f"Erro ao processar arquivo {file_name}: {str(e)}")

# Conversão para tensores
X = torch.tensor(states, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

X_train_tensor = X
y_train_tensor = y

# Configuração do circuito quântico e QNN
num_qubits = 4
feature_map = ZZFeatureMap(num_qubits)
ansatz = RealAmplitudes(num_qubits, reps=1)

qc = QuantumCircuit(num_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

observable = SparsePauliOp("Z" * num_qubits)
estimator = Estimator()
gradient = ParamShiftEstimatorGradient(estimator)

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    observables=observable,
    estimator=estimator,
    gradient=gradient,
    input_gradients=True
)

# Modelo e otimizador
model = TorchConnector(qnn)
loss_func = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Treinamento com envio a cada época
print("\nINICIANDO TREINAMENTO...")
for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_func(torch.sigmoid(outputs.squeeze()), y_train_tensor.squeeze())
    loss.backward()
    optimizer.step()

    rede.start_eprs(num_eprs=5)

    # Envio do circuito
    enviar_circuito_por_epoca(
        circuito=qc,
        epoch=epoch,
        num_qubits=qc.num_qubits,
        circuit_depth=qc.depth()
    )

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

# Predição e métricas
print("\nCALCULANDO MÉTRICAS...")
with torch.no_grad():
    preds = model(X)
    y_pred = torch.sigmoid(preds).squeeze().round().detach().numpy()
    y_true = y.squeeze().numpy()

    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nMÉTRICAS FINAIS:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nResultados Detalhados:")
    print("Entradas:", X.numpy())
    print("Saídas Previstas:", y_pred)
    print("Saídas Reais:    ", y_true)
