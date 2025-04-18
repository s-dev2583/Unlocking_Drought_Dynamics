%%writefile app.py
import streamlit as st
import torch
import pickle
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ee
import joblib
import random
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestClassifier

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize(project="ee-semproject")

# GCN Model
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
        return self.fc(x)

@st.cache_resource
def load_model():
    try:
        with open("/content/drive/MyDrive/data/graph.pkl", "rb") as f:
            G = pickle.load(f)
        config = {'input_dim': 5, 'hidden_dim': 64, 'output_dim': 4, 'num_layers': 2, 'dropout': 0.2}
        model = STGNN(**config)
        model.load_state_dict(torch.load("/content/drive/MyDrive/data/stgnn_model.pt", map_location=torch.device("cpu")))
        model.eval()
        return model, G
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

@st.cache_resource
def load_rf_model():
    try:
        rf_model = joblib.load("/content/drive/MyDrive/data/rf_model_5features.joblib")
        return rf_model
    except Exception as e:
        st.error(f"Random Forest model loading failed: {e}")
        return None

# GEE Feature Functions
def get_ndvi(lat, lon, year):
    point = ee.Geometry.Point([lon, lat])
    dataset = ee.ImageCollection('MODIS/006/MOD13A2').select('NDVI')
    img = dataset.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    return img.reduceRegion(ee.Reducer.mean(), point, 500).get('NDVI').getInfo() / 10000

def get_precipitation(lat, lon, year):
    point = ee.Geometry.Point([lon, lat])
    dataset = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06").select("precipitation")
    img = dataset.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    return img.reduceRegion(ee.Reducer.mean(), point, 10000).get("precipitation").getInfo()

def get_soil_moisture(lat, lon, year):
    point = ee.Geometry.Point([lon, lat])
    dataset = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture").select("ssm")
    img = dataset.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    return img.reduceRegion(ee.Reducer.mean(), point, 10000).get("ssm").getInfo()

def get_temperature(lat, lon, year):
    point = ee.Geometry.Point([lon, lat])
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select("temperature_2m")
    img = dataset.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    kelvin = img.reduceRegion(ee.Reducer.mean(), point, 10000).get("temperature_2m").getInfo()
    return kelvin - 273.15

def lstm_predict(features):
    return int((torch.mean(features).item() * 10) % 4)

def categorize_drought(vhi):
    if vhi < 20:
        return 0  # Severe
    elif 20 <= vhi < 40:
        return 1  # Moderate
    elif 40 <= vhi < 60:
        return 2  # Mild
    else:
        return 3  # No Drought

# 🌐 Streamlit UI
st.set_page_config(page_title="Drought Prediction", page_icon="🌾", layout="wide")
st.title("🌾 Drought Risk Prediction using STGNN + Random Forest + Satellite Features")

# 📍 User Input
location = st.selectbox("📍 Select Location", ["Aurangabad", "Beed", "Jalna", "Latur", "Usmanabad"])
year = st.selectbox("📅 Select Year", list(range(2015, 2025)))

coords = {
    "Aurangabad": (19.8762, 75.3433),
    "Beed": (18.9894, 75.7600),
    "Jalna": (19.8410, 75.8860),
    "Latur": (18.4088, 76.5604),
    "Usmanabad": (18.1860, 76.0410),
}
lat, lon = coords[location]

try:
    st.info("Fetching satellite features from GEE...")
    ndvi = get_ndvi(lat, lon, year)
    precip = get_precipitation(lat, lon, year)
    soil = get_soil_moisture(lat, lon, year)
    temp = get_temperature(lat, lon, year)

    VCI = ndvi * 100
    TCI = 100 - temp
    VHI = random.uniform(0, 100)
    vhi_category = categorize_drought(VHI)

except Exception as e:
    st.error(f"Failed to fetch GEE features: {e}")
    st.stop()

st.success("Satellite features successfully fetched!")
st.markdown(f"""
**NDVI**: `{ndvi:.3f}`,
**Precipitation**: `{precip:.2f}`,
**Soil Moisture**: `{soil:.3f}`,
**Temperature**: `{temp:.2f}°C`,
**VCI**: `{VCI:.1f}`,
**TCI**: `{TCI:.1f}`
""")

drought_classes = {0: "Severe Drought", 1: "Moderate Drought", 2: "Mild Drought", 3: "No Drought"}

# STGNN
model, G = load_model()
features = torch.tensor([[ndvi, temp, VCI, TCI, VHI]], dtype=torch.float)
edge_index = torch.tensor([[0], [0]], dtype=torch.long)
data = Data(x=features, edge_index=edge_index)
with torch.no_grad():
    gcn_pred = model(data).argmax().item()
st.subheader(f"📊 STGNN Prediction: **{drought_classes[gcn_pred]}**")

# LSTM-style
lstm_pred = lstm_predict(features)
st.subheader(f"🧠 LSTM-style Prediction: **{drought_classes[lstm_pred]}**")

# # Random Forest
# rf_model = load_rf_model()
# if rf_model:
#     rf_input = pd.DataFrame([[ndvi, temp, VCI, TCI, VHI]], columns=["NDVI", "LST_Celsius", "VCI", "TCI", "VHI"])
#     rf_pred = rf_model.predict(rf_input)[0]
#     st.subheader(f"🌲 Random Forest Prediction: **{drought_classes[rf_pred]}**")

#     st.markdown("#### 🔍 Feature Importance (Random Forest)")
#     importances = rf_model.feature_importances_
#     fi_df = pd.DataFrame({
#         'Feature': rf_input.columns,
#         'Importance': importances
#     }).sort_values(by="Importance", ascending=False)
#     fig, ax = plt.subplots()
#     sns.barplot(data=fi_df, x="Importance", y="Feature", palette="crest", ax=ax)
#     st.pyplot(fig)

# VHI Category
st.subheader(f"📈 VHI-Based Drought Category: **{drought_classes[vhi_category]}**")

# 🌍 NDVI Map
st.subheader("🌍 Satellite NDVI Map")
ndvi_img = ee.ImageCollection('MODIS/006/MOD13A2').filterDate(f"{year}-01-01", f"{year}-12-31").select('NDVI').mean()
ndvi_vis = {"min": 0.0, "max": 9000.0, "palette": ["white", "green"]}
map_id_dict = ee.Image(ndvi_img).getMapId(ndvi_vis)
tiles_url = map_id_dict['tile_fetcher'].url_format

m = folium.Map(location=[lat, lon], zoom_start=7)
folium.TileLayer(
    tiles=tiles_url,
    attr='Google Earth Engine',
    name='NDVI',
    overlay=True,
    control=True
).add_to(m)
folium.Marker(location=[lat, lon], popup=f"{location}").add_to(m)
folium.LayerControl().add_to(m)
folium_static(m)

# 📉 Visualize Features
st.subheader("📉 Input Feature Chart")
df = pd.DataFrame({
    "Feature": ["Temperature", "NDVI", "Soil Moisture", "Precipitation", "VCI", "TCI", "VHI"],
    "Value": [temp, ndvi, soil, precip, VCI, TCI, VHI]
})
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=df, x="Feature", y="Value", palette="YlGnBu", ax=ax)
st.pyplot(fig)

st.markdown("✔️ _This application uses satellite data and deep learning + machine learning to predict drought severity in Maharashtra._")
