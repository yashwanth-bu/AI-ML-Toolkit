import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('house_price_data_v2.csv')

# Define features
numerical_columns = ['Num_Rooms', 'Square_Feet', 'Year_Built',
    'Lot_Size', 'Num_Bathrooms', 'Garage', 'Pool', 'Walk_Score']
categorical_column = ['Condition', 'Location']

# Separate features and target
features = df.drop(columns=['Price'])
target = df['Price']

# Preprocessing
preprocesser = ColumnTransformer(
    transformers=[
        ('OH-encoder', OneHotEncoder(), categorical_column),
        ('SD-scaler', StandardScaler(), numerical_columns)
    ]
)
pipeline = Pipeline(steps=[('preprocesser', preprocesser)])
scaled_features = pipeline.fit_transform(features)

# Scale target
target_scaler = StandardScaler()
scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, scaled_target, test_size=0.1, random_state=42
)

# PyTorch Dataset
class HouseDataset(Dataset):
    def __init__(self, X, y):
        self.tensor_x = torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)
        self.tensor_y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.tensor_x)
    
    def __getitem__(self, idx):
        return self.tensor_x[idx], self.tensor_y[idx]

# Dataloader
train_dataset = HouseDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Neural Network Model
class HousePriceNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Setup
input_size = scaled_features.shape[1]
model = HousePriceNN(input_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam instead of SGD for better convergence

# Prepare test tensors
X_test_tensor = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Training loop with validation tracking
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_losses.append(val_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f}")

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    original_y = target_scaler.inverse_transform(y_test_tensor)
    original_preds = target_scaler.inverse_transform(predictions)

    print("Final Evaluation:")
    print("MSE:", mean_squared_error(original_y, original_preds))
    print("R2 Score:", r2_score(original_y, original_preds))
