import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset de transações de cartão e amostrar 100 registros aleatórios
df = pd.read_csv(r"C:\Users\Pedro Kruta\Desktop\Zoox\NVIDIA\card_transaction.v1.csv").sample(n=1000000, random_state=13)

# Criar uma nova coluna 'card_id' combinando 'User' e 'Card' como string
df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)

# Converter a coluna 'Amount' de string para float, removendo o símbolo de dólar
df["Amount"] = df["Amount"].str.replace("$", "").astype(float)

# Extrair a hora e o minuto da coluna 'Time' e criar colunas separadas 'Hour' e 'Minute'
df["Hour"] = df["Time"].str[0:2].astype(int)
df["Minute"] = df["Time"].str[3:5].astype(int)

# Remover as colunas 'Time', 'User' e 'Card', pois não são mais necessárias
df = df.drop(["Time", "User", "Card"], axis=1)

# Preencher valores ausentes na coluna 'Errors?' com 'No error'
df["Errors?"] = df["Errors?"].fillna("No error")

# Remover colunas desnecessárias 'Merchant State' e 'Zip'
df = df.drop(columns=["Merchant State", "Zip"], axis=1)

# Converter a coluna 'Is Fraud?' para valores binários (1 para 'Yes', 0 para 'No')
df["Is Fraud?"] = df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)

# Aplicar LabelEncoder para converter valores categóricos em numéricos nas colunas selecionadas
df["Merchant City"] = LabelEncoder().fit_transform(df["Merchant City"])
df["Use Chip"] = LabelEncoder().fit_transform(df["Use Chip"])
df["Errors?"] = LabelEncoder().fit_transform(df["Errors?"])

# Criar um grafo direcionado
G = nx.DiGraph()

# Adicionar nós e propriedades aos nós
for _, row in df.iterrows():
    G.add_node(str(row['card_id']), feature=[row['Amount'], row['Use Chip'], row['Hour'], row['Minute'], 0, 0])  # Defina as features como desejar
    G.add_node(str(row['Merchant Name']), feature=[0, 0, 0, 0, row['MCC'], row['Errors?']])  # Defina as features como desejar

# Adicionar arestas e propriedades às arestas
for _, row in df.iterrows():
    G.add_edge(str(row['card_id']), str(row['Merchant Name']),
               year=row['Year'], month=row['Month'], day=row['Day'],
               hour=row['Hour'], minute=row['Minute'], amount=row['Amount'],
               use_chip=row['Use Chip'], merchant_city=row['Merchant City'],
               errors=row['Errors?'], mcc=row['MCC'])

# Função para obter embeddings dos nós do grafo
def get_node_embeddings(model, G):
    features = torch.tensor([G.nodes[node]['feature'] for node in G.nodes], dtype=torch.float)
    adjacency_matrix = nx.to_numpy_array(G)
    adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float)
    model.eval()
    with torch.no_grad():
        embeddings = model(features, adjacency_tensor)
    return embeddings

# Definição do Modelo GNN para gerar embeddings
class FraudDetectionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudDetectionGNN, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, 1)  # Nova camada para reduzir a saída para um único valor

    def forward(self, features, adjacency_matrix):
        x = F.relu(self.fc1(features))
        x = torch.mm(adjacency_matrix, x)  # Multiplicação pelo grafo de adjacência
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Saída final com um único valor por nó
        return x

# Instanciar o modelo
model = FraudDetectionGNN(in_channels=6, hidden_channels=64, out_channels=32)

# Função de treinamento
def train(model, features, adjacency_matrix, labels, epochs=201):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model.train()

    labeled_nodes = [i for i, node in enumerate(G.nodes) if node in df['card_id'].values]
    labeled_features = features[labeled_nodes]
    labeled_labels = labels[:len(labeled_nodes)]

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(features, adjacency_matrix).squeeze()  # Ajustar a saída para remover dimensões extras
        loss = F.mse_loss(out[labeled_nodes], labeled_labels.float())  # Usar apenas os nós rotulados para calcular a perda
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Preparar dados para o treinamento
features = torch.tensor([G.nodes[node]['feature'] for node in G.nodes], dtype=torch.float)
adjacency_matrix = torch.tensor(nx.to_numpy_array(G), dtype=torch.float)
labels = torch.tensor(df['Is Fraud?'].values, dtype=torch.long)

# Treinar o modelo
train(model, features, adjacency_matrix, labels)

# Gerar embeddings após o treinamento
embeddings = get_node_embeddings(model, G)
embeddings_np = embeddings.numpy()  # Converter para NumPy

# Filtrar os embeddings e rótulos para apenas os nós rotulados
labeled_nodes = [i for i, node in enumerate(G.nodes) if node in df['card_id'].values]
embeddings_np_labeled = embeddings_np[labeled_nodes]
labels_labeled = labels[:len(labeled_nodes)].numpy()

# Dividir os embeddings em treino e teste
X_train, X_test, y_train, y_test = train_test_split(embeddings_np_labeled, labels_labeled, test_size=0.2, random_state=42)

# Treinar o modelo XGBoost
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)

# Fazer previsões
y_pred = xgb_model.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo XGBoost: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))
