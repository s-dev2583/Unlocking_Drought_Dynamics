# ============================================
# 📌 1. IMPORT LIBRARIES
# ============================================
import torch
import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ============================================
# 📌 2. STGNN MODEL CLASS
# ============================================
class STGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(STGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)

        x = self.fc(x)
        return x

# ============================================
# 📌 3. LOAD MODEL AND GRAPH
# ============================================
# Load the graph and dataset
with open("data/graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load the trained model
config = {
    'input_dim': 5,
    'hidden_dim': 64,
    'output_dim': 4,
    'num_layers': 2,
    'dropout': 0.2,
}

# Initialize and load the model
model = STGNN(**config)
model.load_state_dict(torch.load("data/stgnn_model.pt"))
model.eval()

# ============================================
# 📌 4. SINGLE PREDICTION INPUT
# ============================================
# Example single sample input (use actual values from your dataset)
single_sample = torch.tensor([[35.6, 0.45, 28.0, 0.6, 15]], dtype=torch.float)

# Edge index (dummy edge since it's a single prediction)
edge_index = torch.tensor([[0], [0]], dtype=torch.long)

# Format into PyTorch Geometric Data format
single_data = Data(x=single_sample, edge_index=edge_index)

# ============================================
# 📌 5. MAKE PREDICTION
# ============================================
with torch.no_grad():
    output = model(single_data)
    prediction = torch.argmax(output, dim=1).item()

# ============================================
# 📌 6. DISPLAY RESULT
# ============================================
# Map predictions to labels
risk_labels = ["Severe Drought", "Moderate Drought", "Mild Drought", "No Drought"]
predicted_label = risk_labels[prediction]

# Display the result
print("\n🔥 Single Prediction Result:")
print(f"➡️  Input Features: {single_sample.tolist()[0]}")
print(f"➡️  Predicted Drought Risk: {predicted_label}")
