# INSTALE AS DEPENDÊNCIAS NECESSÁRIAS
# pip install qiskit qiskit-aer qiskit-machine-learning numpy matplotlib scikit-learn qiskit-algorithms

import matplotlib
matplotlib.use('Agg')

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
import os

# Configuração do simulador
simulator = AerSimulator(method='statevector')

def load_qasm_files(path, max_files=20):
    """Carrega arquivos QASM de forma robusta"""
    states = []
    labels = []
    
    # Correção aplicada aqui: parênteses balanceados
    qasm_files = sorted([f for f in os.listdir(path) if f.endswith('.qasm')])[:max_files]
    
    for i, filename in enumerate(qasm_files):
        try:
            with open(os.path.join(path, filename), 'r') as f:
                qc = QuantumCircuit.from_qasm_str(f.read())
                
            # Simulação
            qc.save_statevector()
            result = simulator.run(transpile(qc, simulator)).result()
            state = result.get_statevector()
            
            # Features
            features = np.abs(state)[:8]**2  # Primeiras 8 amplitudes
            states.append(features)
            labels.append(0 if i < max_files//2 else 1)
            
            print(f"Arquivo {filename} processado com sucesso")
            
        except Exception as e:
            print(f"Erro no arquivo {filename}: {str(e)}")
            continue
    
    return np.array(states), np.array(labels)

# Carregamento dos dados
try:
    X, y = load_qasm_files("base_test_mnist_784_f90/qasm/")
    if len(X) == 0:
        raise ValueError("Nenhum arquivo QASM válido foi encontrado!")
except Exception as e:
    print(f"Falha ao carregar dados: {str(e)}")
    exit()

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuração do VQC
vqc = VQC(
    sampler=Sampler(),
    feature_map=ZZFeatureMap(8, reps=2),
    ansatz=RealAmplitudes(8, reps=1),
    optimizer=COBYLA(maxiter=50)
)

# Treinamento
print("\nIniciando treinamento...")
vqc.fit(X_train, y_train)

# Avaliação
y_pred = vqc.predict(X_test)

# Métricas
print("\nMétricas de Desempenho:")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))