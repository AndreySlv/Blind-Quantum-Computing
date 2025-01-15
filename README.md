# Quantumnet - Blind Quantum Computing 

Este projeto simula uma rede quântica para transmissão de informações entre um remetente e um destinatário. A rede é projetada com diferentes topologias, abrangendo todas as camadas necessárias para um funcionamento eficiente. O repositório contém os códigos em Python para execução da simulação.

## Descrição

O repositório inclui os componentes essenciais da rede e do host, além dos qubits e pares EPR. Pode haver visualização de três tipos de topologia: anel, linha e grade. O projeto, também, abrange todas as camadas necessárias, desde a física até a aplicação, garantindo o funcionamento completo da rede quântica.

## Diretórios 
- ``quantumnet``: 
  - ``/components`` : arquivos básicos para o realização das simulações.
  - ``/layers`` : arquivos que contém as camadas para o funciomento básico da rede.
  - ``/objects`` : elementos essenciais para a funcionamento dos componentes.

## Camada de Aplicação

A camada de aplicação é responsável por gerenciar as interações entre clientes e servidores na rede quântica, além de implementar os principais protocolos de computação quântica cega. Essa camada utiliza qubits e pares EPR para simular a transmissão de informações, realizar operações quânticas e garantir a segurança na comunicação.

### Principais Funcionalidades

#### 1. **Execução de Protocolos**
A camada suporta a execução de diferentes protocolos quânticos. Os dois principais são:
- **Andrew Childs Protocol (AC-BQC):**
  - Alice prepara e envia qubits para Bob.
  - Bob realiza operações baseadas em instruções clássicas enviadas por Alice.
  - Os qubits são devolvidos para Alice após as operações.
- **BFK Protocol (BFK-BQC):**
  - Alice prepara qubits e envia para Bob.
  - Bob cria um estado de "brickwork" para computação baseada em medição.
  - Alice instrui Bob a realizar medições adaptativas em rodadas sucessivas.

Esses protocolos simulam computação quântica cega segura e eficiente.

#### 2. **Gestão de Fidelidade**
A camada permite registrar e calcular métricas de fidelidade, garantindo a qualidade das transmissões e operações:
- Registro das fidelidades de rotas durante a transmissão.
- Cálculo da fidelidade média para avaliar a confiabilidade da rede.
- Impressão de fidelidades registradas para análise.

#### 3. **Operações Quânticas**
- Geração de operações quânticas aleatórias, como X, Y e Z.
- Aplicação de decodificações quânticas (Clifford) para garantir a correção das operações.

## Classe Network

A classe `Network` é a base para a simulação de redes quânticas. Ela gerencia a topologia da rede, os nós (hosts), os canais de comunicação, e os pares EPR. Além disso, integra todas as camadas necessárias para simular a transmissão de qubits, incluindo física, enlace, rede, transporte e aplicação.

#### 1. **Gerenciamento da Rede**
- **Topologia**:
  - Suporta diferentes tipos de grafos: grade, linha e anel.
  - Permite configurar e visualizar a topologia, incluindo clientes, servidor e nós regulares.
- **Hosts**:
  - Gerencia os hosts da rede, armazenando-os em um dicionário com base em seus IDs.
  - Adiciona, recupera e inicializa hosts com qubits.

#### 2. **Gerenciamento de Canais**
- Configura canais de comunicação entre os nós.
- Inicializa as probabilidades de criação de pares EPR (sob demanda e de replay).
- Associa pares EPRs a cada canal.

#### 3. **Gerenciamento de Pares EPR**
- **Criação e Registro**:
  - Gera pares EPR e os registra nos canais da rede.
- **Manipulação**:
  - Remove EPRs de canais específicos.
  - Aplica decoerência a pares EPR para simular os efeitos do tempo.

#### 4. **Simulação de Slices**
- **Configuração de Slices**:
  - Calcula e armazena os caminhos (rotas) mais curtos para cada cliente em relação ao servidor.
  - Penaliza arestas para evitar sobreposição de rotas em diferentes slices.
- **Visualização**:
  - Exibe o grafo da rede com as rotas dos slices destacadas por cores.
- **Execução de Simulações**:
  - Roda simulações para configurar rotas finais de slices, retornando os caminhos calculados.

#### 5. **Simulação de Requisições**
- **Geração de Circuitos**:
  - Cria circuitos quânticos aleatórios com qubits e portas.
  - Registra as instruções do circuito para análise.
- **Requisições**:
  - Gera e armazena requisições de teletransporte de qubits, associando informações como IDs de nós, número de qubits, protocolos, rotas e circuitos.
  - Envia requisições para o controlador da rede.

#### 6. **Métricas e Monitoramento**
- **Métricas**:
  - Coleta dados como número de qubits e EPRs usados, fidelidade média das camadas, e tamanho médio das rotas.
  - Exporta métricas para arquivos CSV, exibe no console ou retorna como variável.
- **Monitoramento de Timeslots**:
  - Registra e exibe informações de timeslots usados por qubits e canais.

#### 7. **Decoerência e Reinicialização**
- Aplica decoerência a qubits e EPRs para simular perdas de fidelidade ao longo do tempo.
- Reinicia a rede, restaurando EPRs e redefinindo qubits nos hosts.

## Controlador (Controller)

O controlador é o responsável por gerenciar a alocação, execução e monitoramento de requisições na rede quântica. Ele realiza o planejamento de rotas, a priorização de requisições e garante a eficiência no uso dos recursos da rede.

### Principais Funcionalidades

#### 1. **Inicialização**
- Configura os **slices** (segmentos de rede) com base nos clientes, protocolos e rotas fornecidos.
- Garante que cada slice esteja associado a um cliente, um servidor e um caminho específico na topologia da rede.

#### 2. **Gerenciamento de Requisições**
- **Recepção e Processamento**:
  - Recebe requisições contendo informações como IDs dos nós, número de qubits e protocolo.
  - Processa as requisições, tentando agendá-las em **timeslots** disponíveis.
- **Agendamento de Requisições**:
  - Prioriza as requisições com base no número de qubits e na complexidade do circuito.
  - Busca reutilizar timeslots disponíveis ou encontra o próximo timeslot livre para evitar conflitos.
- **Execução de Requisições**:
  - Valida e executa cada requisição no timeslot apropriado.
  - Registra requisições bem-sucedidas e falhas, permitindo a análise posterior.

#### 3. **Gerenciamento de Rotas**
- **Verificação de Disponibilidade**:
  - Confirma se uma rota está livre para uso em um determinado timeslot.
- **Reserva e Liberação**:
  - Reserva rotas para requisições agendadas, evitando conflitos de uso.
  - Libera rotas após a execução das requisições, permitindo o reuso eficiente dos recursos.

#### 4. **Simulação em Slices**
- **Mapeamento para Slices**:
  - Associa requisições a slices com base nos protocolos definidos.
- **Agendamento em Timeslots**:
  - Alterna entre slices para distribuir requisições ao longo dos timeslots, garantindo um balanceamento eficiente.

#### 5. **Relatórios e Métricas**
- Gera relatórios detalhados com informações sobre requisições:
  - Sucessos, falhas e detalhes das rotas utilizadas.
- Exibe métricas úteis para avaliar a eficiência do agendamento e a qualidade da rede.

---

Este projeto foi elaborado como parte de um artigo em processo de revisão para o SBRC 2025, com o objetivo de contribuir para o avanço das técnicas de simulação e gerenciamento de redes quânticas de alta eficiência e confiabilidade. 

As funcionalidades desenvolvidas asseguram que o sistema gerencie de maneira eficiente as requisições, alocação de rotas e execução de protocolos, proporcionando uma comunicação quântica cega confiável e com alta fidelidade entre os diferentes nós da rede durante as simulações e operações em slices.
