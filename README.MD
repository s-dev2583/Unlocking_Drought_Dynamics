# 🌾 Drought Prediction System using STGNN (Spatio-Temporal Graph Neural Network)

This project is a complete pipeline to predict **drought severity** in regions of Maharashtra, India, using a **Spatio-Temporal Graph Neural Network (STGNN)**. It combines **satellite-based feature inputs** with **graph-based learning** to analyze spatial and temporal patterns effectively.

The system includes:

- A **Streamlit-based web interface**
- A **pretrained STGNN model**
- Interactive visualizations
- Location-based inference using preprocessed features

---

## 📌 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Features Used](#features-used)
- [Graph Construction](#graph-construction)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Visual Output](#visual-output)
- [Model Training (Optional)](#model-training-optional)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)
- [License](#license)

---

## 🧠 Overview

> 🌍 "Water scarcity is a growing challenge, especially in drought-prone regions. With timely predictions, we can mitigate the impact."

This project uses:

- **Remote Sensing data** (NDVI, LST, SAR, VHI, etc.)
- A **Graph Neural Network** (PyTorch Geometric)
- **Time series modeling** using node-level features over months
- **Geo-visualization** for showing severity on maps

---

## 🛡️ Model Architecture

The STGNN consists of:

- **2 GCN layers** (`torch_geometric.nn.GCNConv`)
- **ReLU activation**
- **Fully Connected layer** for output

```python
class STGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STGNN, self).__init__()
        self.convs = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return self.fc(x)
```

---

## 🚁 Features Used

| Feature           | Source/Satellite | Description              |
| ----------------- | ---------------- | ------------------------ |
| NDVI              | MODIS/Sentinel-2 | Vegetation health        |
| LST (Day/Night)   | MODIS            | Land Surface Temperature |
| VHI               | NOAA             | Vegetation Health Index  |
| SAR (HH/HH Ratio) | Sentinel-1       | Soil moisture proxy      |
| Clear Sky Days    | MODIS QA         | Cloud cover estimation   |

---

## 🧠 Graph Construction

We represent regions as **nodes**, and geographical or climatic proximity defines **edges**.

- The graph is saved as a pickled dictionary `graph.pkl`:

```python
{
  "edge_index": torch.tensor([[0, 1], [1, 2], ...], dtype=torch.long)
}
```

- `edge_index` is used for message passing in GCNs.

---

## 📂 Folder Structure

```
stgnn-drought-app/
├── app.py               # Streamlit frontend
├── stgnn_model.pth      # Trained model
├── graph.pkl            # Graph edge_index
├── utils.py             # Feature extraction, data prep
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/stgnn-drought-app.git
cd stgnn-drought-app
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

(Optional for Colab):

```python
!pip install pyngrok
from pyngrok import ngrok
!streamlit run app.py &
print(ngrok.connect(8501))
```

---

## 🖥️ Usage

1. Launch the Streamlit app.
2. Input features (NDVI, LST, etc.) manually or via a form.
3. Select a location (or enter lat/lon).
4. Press **Predict**.
5. See results on a map and as a textual classification.

---

## 📊 Visual Output

The result includes:

- 🌐 An interactive **Folium map** showing location
- 📈 Drought severity classification:
  - `0`: No Drought 🟢
  - `1`: Mild Drought 🟡
  - `2`: Moderate Drought 🟠
  - `3`: Severe Drought 🔴

More visualizations you can add:

- Time series of NDVI/VHI
- Heatmap of prediction confidence
- Animated drought spread across regions

---

## 🏋️ Model Training (Optional)

You can retrain the model using PyTorch and PyTorch Geometric:

```python
model = STGNN(input_dim=5, hidden_dim=32, output_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
```

Save the trained model:

```python
torch.save(model.state_dict(), "stgnn_model.pth")
```

---

## 🤯 Troubleshooting

| Issue                     | Fix                                                  |
| ------------------------- | ---------------------------------------------------- |
| `Missing state_dict keys` | Ensure same model architecture as used in training   |
| `GEE asset not found`     | We removed all GEE dependencies in the final version |
| `Ngrok ERR_NGROK_8012`    | Make sure Streamlit server is running before ngrok   |
| `KeyError: 'edge_index'`  | Ensure `graph.pkl` contains `edge_index` key         |

---

## 🙏 Credits

- NASA MODIS / Sentinel Data
- NOAA CDR for VHI
- PyTorch Geometric Team
- Streamlit + Folium for frontend mapping

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

