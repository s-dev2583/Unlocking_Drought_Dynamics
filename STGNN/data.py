import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import pickle
import os


file_path = "Maharashtra_Drought_2023_2024_Full_Drive.csv"
df = pd.read_csv(file_path)


num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# if vhi is not presesnt calculate
df['VHI'] = 0.5 * df['VCI'] + 0.5 * df['TCI']

# Normalize NDVI if present
if 'NDVI' in df.columns:
    scaler = MinMaxScaler()
    df['NDVI'] = scaler.fit_transform(df[['NDVI']])

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month


def categorize_drought(vhi):
    if vhi < 20:
        return 0  # Severe
    elif 20 <= vhi < 40:
        return 1  # Moderate
    elif 40 <= vhi < 60:
        return 2  # Mild
    else:
        return 3  # No Drought

df['Drought_Risk'] = df['VHI'].apply(categorize_drought)


def create_graph(df):
    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i, features=df.iloc[i][['VHI', 'NDVI', 'LST_Celsius', 'SAR_Mean', 'Clear_sky_days']].values)

    for i in range(len(df) - 1):
        G.add_edge(i, i + 1, weight=1)

    return G

G = create_graph(df)


os.makedirs('data', exist_ok=True)

# Save graph with pickle
with open("data/graph.pkl", "wb") as f:
    pickle.dump(G, f)

# Save dataset
df.to_csv("data/drought_dataset.csv", index=False)

print("\n✅ Preprocessing completed. Graph and dataset saved.")