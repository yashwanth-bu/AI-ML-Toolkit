import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

df = pd.read_csv('my_diabetes_data.csv')

features = ['Glucose', 'BMI', 'BloodPressure', 'Insulin', 'Age']

label_encoder = LabelEncoder()
encoded_targets = label_encoder.fit_transform(df['Diabetes']).reshape(-1, 1)

scaler_encoder = StandardScaler()
scaled_features = scaler_encoder.fit_transform(df[features])

X_train, X_test, y_train, y_test = train_test_split(scaled_features, encoded_targets, test_size=0.1, random_state=42)

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.tensor_x = torch.tensor(X, dtype=torch.float32)
        self.tensor_y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.tensor_x)
    
    def __getitem__(self, idx):
        return self.tensor_x[idx], self.tensor_y[idx]

dataset = DiabetesDataset(X_train, y_train)

loader = DataLoader(dataset, batch_size=10, shuffle=True)

test_tensor_x = torch.tensor(X_test, dtype=torch.float32)
test_tensor_y = torch.tensor(y_test, dtype=torch.float32)

class DiabetesNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
input_size = len(features)
model = DiabetesNN(input_dim=input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    
    model.train()
    epoch_loss = 0
    for input, target in loader:
        optimizer.zero_grad()
        output = model(input)
        loss_value = criterion(output, target)
        loss_value.backward()
        optimizer.step()
        epoch_loss += loss_value.item()
        
    avg_train_loss = epoch_loss / len(loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(test_tensor_x)
        val_loss = criterion(val_outputs, test_tensor_y)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")
    
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    raw_preds = model(test_tensor_x)
    preds_binary = (raw_preds > 0.5).float()
    
    y_true = test_tensor_y.squeeze().numpy()
    y_pred = preds_binary.squeeze().numpy()
    
    print("Accuracy Score:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))