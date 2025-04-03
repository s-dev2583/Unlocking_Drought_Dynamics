# ============================================
# 👈 1. IMPORT LIBRARIES
# ============================================
import streamlit as st
import torch
import pickle
import folium
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# 👈 2. STGNN MODEL CLASS
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
# 👈 3. LSTM MODEL CLASS
# ============================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x

# ============================================
# 👈 4. LOAD MODELS AND GRAPH
# ============================================
@st.cache_resource
def load_models():
    try:
        # Load the graph and dataset
        with open("/content/drive/MyDrive/data/graph.pkl", "rb") as f:
            G = pickle.load(f)

        # Load STGNN Model
        stgnn_config = {
            'input_dim': 5,
            'hidden_dim': 64,
            'output_dim': 4,
            'num_layers': 2,
            'dropout': 0.2,
        }
        stgnn_model = STGNN(**stgnn_config)
        stgnn_model.load_state_dict(torch.load("/content/drive/MyDrive/data/stgnn_model.pt", map_location=torch.device("cpu")))
        stgnn_model.eval()

        # Load LSTM Model (Placeholder Weights)
        lstm_model = LSTMModel(input_dim=5, hidden_dim=64, num_layers=2, output_dim=4)
        lstm_model.eval()
        
        return stgnn_model, lstm_model, G
    except Exception as e:
        st.error(f"⚠️ Model Loading Failed: {e}")
        return None, None, None

# ============================================
# 👈 5. USER INPUT: SELECT LOCATION & FEATURES
# ============================================
st.set_page_config(page_title="Drought Prediction", page_icon="🌍", layout="wide")

st.title("🌍 STGNN & LSTM-Based Drought Prediction")
st.markdown("Predict drought severity using Graph Neural Networks and LSTMs.")

# Sidebar: Select Location & Features
st.sidebar.header("📍 Select Location & Input Features")

location = st.sidebar.selectbox("📍 Choose a Location", ["Aurangabad", "Beed", "Jalna", "Latur", "Usmanabad"])

# Dynamic Feature Inputs (Sliders)
temperature = st.sidebar.slider("🌡 Temperature (°C)", 20, 60, 40)
ndvi = st.sidebar.slider("🌱 NDVI Index", 0.0, 1.0, 0.3)
soil_moisture = st.sidebar.slider("💧 Soil Moisture (%)", 0, 50, 25)
precipitation = st.sidebar.slider("🌧 Precipitation (mm)", 0.0, 1.0, 0.5)
humidity = st.sidebar.slider("🌬 Humidity (%)", 0, 100, 15)

# ============================================
# 👈 6. LOAD MODELS & MAKE PREDICTIONS
# ============================================
stgnn_model, lstm_model, G = load_models()
if stgnn_model is None or lstm_model is None:
    st.error("🚨 Models not loaded. Please check the files and try again.")
    st.stop()

# Convert user input to tensors
input_features = torch.tensor([[temperature, ndvi, soil_moisture, precipitation, humidity]], dtype=torch.float)
edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Single-node prediction

data = Data(x=input_features, edge_index=edge_index)

# Run STGNN Prediction
with torch.no_grad():
    stgnn_prediction = stgnn_model(data).argmax().item()

# Run LSTM Prediction
lstm_input = input_features.unsqueeze(0)
with torch.no_grad():
    lstm_prediction = lstm_model(lstm_input).argmax().item()

# Drought Risk Categories
drought_classes = {0: "Severe Drought", 1: "Moderate Drought", 2: "Mild Drought", 3: "No Drought"}

# Display Prediction Results
st.subheader(f"📍 Location: **{location}**")
st.subheader(f"🔥 STGNN Predicted Drought Risk: **{drought_classes[stgnn_prediction]}**")
st.subheader(f"📈 LSTM Predicted Trend: **{drought_classes[lstm_prediction]}**")

# ============================================
# 👈 7. FEATURE VISUALIZATION
# ============================================
feature_data = pd.DataFrame({"Feature": ["Temperature", "NDVI", "Soil Moisture", "Precipitation", "Humidity"],"Value": [temperature, ndvi, soil_moisture, precipitation, humidity]})
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=feature_data, x="Feature", y="Value", palette="coolwarm", ax=ax)
ax.set_ylabel("Value")
ax.set_title("Feature Distribution")
st.pyplot(fig)