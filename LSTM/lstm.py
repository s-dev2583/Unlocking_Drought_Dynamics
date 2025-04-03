import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ✅ Load Dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("✅ Data Loaded:", df.shape)
    print(df.head())
    return df

# ✅ Preprocessing (Scaling, Handling Missing Data)
def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)  # Fill missing values
    scaler = MinMaxScaler()
    
    # Selecting Features for Prediction
    features = ["VHI", "NDVI", "LST_Celsius", "SAR_Mean", "Clear_sky_days"]
    
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

# ✅ Convert Data to LSTM Format
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i : i + seq_length])
        labels.append(data[i + seq_length, -1])  # Predicting drought risk
    return np.array(sequences), np.array(labels)

# ✅ LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Taking last output
        return out

# ✅ Train Model
def train_lstm(model, train_loader, criterion, optimizer, epochs=50):
    loss_history = []
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        loss_history.append(loss.item())
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return loss_history

# ✅ Plot Training Loss
def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.show()

# ✅ Predict Drought Risk
def predict(model, test_data):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data).numpy()
    return predictions

# ✅ Main Pipeline
def main():
    # Load & Preprocess Data
    df = load_data("Maharashtra_Drought_2023_2024_Full_Drive.csv")
    df, scaler = preprocess_data(df)
    
    # Convert to Sequences for LSTM
    seq_length = 5
    features = ["VHI", "NDVI", "LST_Celsius", "SAR_Mean", "Clear_sky_days", "Drought_Risk"]
    sequences, labels = create_sequences(df[features].values, seq_length)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Data Loader
    train_loader = [(X_train_tensor, y_train_tensor)]

    # Initialize Model
    input_size = X_train.shape[2]  # Number of features
    model = LSTMModel(input_size=input_size, hidden_size=64, output_size=1)

    # Define Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    loss_history = train_lstm(model, train_loader, criterion, optimizer)
    
    # Plot Training Loss
    plot_loss(loss_history)

    # Predictions
    predictions = predict(model, X_test_tensor)

    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Drought Risk")
    plt.title("LSTM Drought Prediction")
    plt.legend()
    plt.show()

# ✅ Run Script
if __name__ == "__main__":
    main()
