
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import folium
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler

# ✅ Define STGNN Model
class STGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ✅ Load Trained STGNN Model
@st.cache_resource
def load_model():
    model = STGNN(input_dim=5, hidden_dim=64, output_dim=4)
    model.load_state_dict(torch.load("stgnn_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ✅ Generate Fake Data for Locations
def get_location_data(location):
    data = {
        "Aurangabad": [50, 0.3, 25, 0.5, 15],
        "Beed": [40, 0.2, 30, 0.6, 10],
        "Jalna": [60, 0.4, 20, 0.4, 20],
        "Latur": [55, 0.35, 28, 0.55, 12],
        "Usmanabad": [45, 0.25, 22, 0.45, 18]
    }
    return data.get(location, [50, 0.3, 25, 0.5, 15])  # Default values

# ✅ Create Map Visualization
def create_map(predictions):
    m = folium.Map(location=[19.7515, 75.7139], zoom_start=7)

    color_map = {0: "red", 1: "orange", 2: "yellow", 3: "green"}
    locations = {
        "Aurangabad": [19.8762, 75.3433],
        "Beed": [18.9894, 75.7600],
        "Jalna": [19.8410, 75.8860],
        "Latur": [18.4088, 76.5604],
        "Usmanabad": [18.1860, 76.0410]
    }

    for loc, coords in locations.items():
        folium.Marker(
            location=coords,
            popup=f"{loc}: Risk Level {predictions[loc]}",
            icon=folium.Icon(color=color_map[predictions[loc]])
        ).add_to(m)
    
    return m

# ✅ Streamlit App UI
st.title("🌍 STGNN-Based Drought Prediction")

# Sidebar: Select Location
location = st.sidebar.selectbox("Select a Location", ["Aurangabad", "Beed", "Jalna", "Latur", "Usmanabad"])

# Load Model
model = load_model()

# Get Data for Selected Location
input_features = get_location_data(location)

# Predict
input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
prediction = model(input_tensor).argmax().item()

# Drought Risk Categories
drought_classes = {0: "Severe Drought", 1: "Moderate Drought", 2: "Mild Drought", 3: "No Drought"}

# Display Prediction
st.write(f"### 📍 Location: {location}")
st.write(f"### 🔥 Predicted Drought Risk: **{drought_classes[prediction]}**")

# Map Visualization
predictions = {location: prediction}
st.write("### 🌍 Drought Risk Map")
folium_static(create_map(predictions))

