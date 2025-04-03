
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import pickle


def load_data(graph_path, feature_path):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    df = pd.read_csv(feature_path)

    X = torch.tensor([G.nodes[node]['features'] for node in G.nodes()], dtype=torch.float)
    y = torch.tensor(df['Drought_Risk'].values, dtype=torch.long)

    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    data = Data(x=X, edge_index=edge_index, y=y)
    return data


class STGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(STGNN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)

        x = self.fc(x)
        return x

def train(model, data, optimizer, criterion, epochs):
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')


config = {
    'input_dim': 5,
    'hidden_dim': 64,
    'output_dim': 4,
    'num_layers': 2,
    'dropout': 0.2,
}

lr = 0.0001
epochs = 250

data = load_data("data/graph.pkl", "data/drought_dataset.csv")

model = STGNN(**config)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train(model, data, optimizer, criterion, epochs)

# Save the model
torch.save(model.state_dict(), "data/stgnn_model.pt")
print("\n✅ Model saved.")
