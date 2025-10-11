import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load and preprocess data
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)  # Make sure target shape is (samples, 1)

# Normalize features to zero mean and unit variance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to torch tensors
tensor_X = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32)

# Train-validation split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(tensor_X, tensor_y, test_size=0.2, random_state=42)

# Create DataLoader objects for batching
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

# 2. Define the neural network model
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # input layer
            nn.ReLU(),
            nn.Linear(128, 64),         # hidden layer
            nn.ReLU(),
            nn.Linear(64, 1)            # output layer
        )
        
    def forward(self, x):
        return self.model(x)

# 3. Initialize model, loss function, optimizer, and scheduler
model = RegressionNN(X.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Scheduler to reduce LR on plateau of validation loss (no verbose argument)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # Minimize validation loss
    patience=5,      # Wait 5 epochs before reducing LR
    factor=0.5       # Reduce LR by factor of 0.5
)

# 4. Training with early stopping and checkpointing
n_epochs = 100
patience = 10                       # Early stopping patience
best_val_loss = float('inf')        # Track best validation loss
epochs_no_improve = 0               # Count epochs without improvement
train_losses, val_losses = [], []   # Lists to store losses for plotting

checkpoint_path = "best_model.pth"  # Path to save best model

for epoch in range(n_epochs):
    model.train()  # Set to training mode
    total_train_loss = 0.0
    
    # Training loop over batches
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()                # Reset gradients
        output = model(batch_X)              # Forward pass
        loss = loss_fn(output, batch_y)     # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        total_val_loss = 0.0
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print losses
    print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # Optionally print learning rate manually since verbose is off
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']:.6f}")

    # Check for improvement to save best model and reset early stopping counter
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Best model saved (Val Loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("⏹️ Early stopping triggered.")
            break

# 5. Plot training and validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Load best model weights for inference or further evaluation
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
print("✅ Best model reloaded for final use.")

# -------------------------------------------
# 4. Loading model and making predictions on NEW DATA
# -------------------------------------------

# Load the saved model weights into a new instance of the model
loaded_model = RegressionNN(X.shape[1])
loaded_model.load_state_dict(torch.load(checkpoint_path))
loaded_model.eval()  # Set to eval mode

# Example: New raw data to predict on (must have same features, unscaled)
# Let's create some new sample data (randomly or use some existing data)
# Here we take a few samples from the original dataset for demonstration:
new_raw_data = data.data[:5]  # first 5 samples (numpy array)

# IMPORTANT: Must apply the SAME scaler used during training!
new_data_scaled = scaler.transform(new_raw_data)

# Convert to tensor
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = loaded_model(new_tensor)

print("Predictions on new data:")
print(predictions.numpy())