# ============================================
# 📌 1. IMPORT LIBRARIES
# ============================================
import torch
import pandas as pd
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import folium

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
# 📌 3. LOAD MODEL AND DATA
# ============================================
# Load the graph and dataset
with open("data/graph.pkl", "rb") as f:
    G = pickle.load(f)

df = pd.read_csv("data/drought_dataset.csv")

# Prepare data for prediction
X = torch.tensor([G.nodes[node]['features'] for node in G.nodes()], dtype=torch.float)
y = torch.tensor(df['Drought_Risk'].values, dtype=torch.long)
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
data = Data(x=X, edge_index=edge_index, y=y)

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
# 📌 4. PREDICTION
# ============================================
with torch.no_grad():
    predictions = model(data).argmax(dim=1).numpy()

# Add predictions to the DataFrame
df['Predicted_Risk'] = predictions
df.to_csv("data/drought_predictions.csv", index=False)

print("\n✅ Predictions saved to 'data/drought_predictions.csv'")

# ============================================
# 📌 5. VISUALIZATION
# ============================================

# 🔥 1️⃣ Actual vs Predicted Visualization
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['Drought_Risk'], label='Actual', color='blue', marker='o')
plt.plot(df['date'], df['Predicted_Risk'], label='Predicted', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Drought Risk')
plt.title('Actual vs Predicted Drought Risk')
plt.legend()
plt.grid(True)
plt.show()

# 🔥 2️⃣ Confusion Matrix Visualization
cm = confusion_matrix(df['Drought_Risk'], df['Predicted_Risk'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Severe", "Moderate", "Mild", "No Drought"],
            yticklabels=["Severe", "Moderate", "Mild", "No Drought"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 🔥 3️⃣ Classification Report
print("\n📊 Classification Report:\n")
print(classification_report(df['Drought_Risk'], df['Predicted_Risk']))

# ============================================
# 📌 6. OPTIONAL: GEOSPATIAL VISUALIZATION
# ============================================
# This part assumes you have lat/lon data for plotting

if 'latitude' in df.columns and 'longitude' in df.columns:
    print("\n🌍 Geospatial Visualization using Folium...")

    # Create Folium Map
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=7)

    # Add markers
    for _, row in df.iterrows():
        color = 'red' if row['Predicted_Risk'] == 0 else 'orange' if row['Predicted_Risk'] == 1 else 'yellow' if row['Predicted_Risk'] == 2 else 'green'

        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"Drought Risk: {row['Predicted_Risk']}"
        ).add_to(m)

    # Save the map
    m.save("data/drought_map.html")
    print("\n✅ Geospatial map saved as 'data/drought_map.html'")
