# Re-import necessary modules if in a new script
import torch
import torch.nn as nn
import pandas as pd


# Redefine your model class (must match training)
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

# Get input size (must match what was used before)
input_size =  scaled_features.shape[1]  # Or save this value during training

# Load model
model = HousePriceNN(input_size)
model.load_state_dict(torch.load('house_price_model.pt'))
model.eval()

# Example new data (must have same columns)
new_data = pd.DataFrame([{
    'Num_Rooms': 4,
    'Square_Feet': 2200,
    'Year_Built': 2010,
    'Lot_Size': 0.25,
    'Num_Bathrooms': 2,
    'Garage': 1,
    'Pool': 0,
    'Walk_Score': 85,
    'Condition': 'Good',  # Categorical
    'Location': 'Suburban'  # Categorical
}])

# Preprocess features
scaled_new = pipeline.transform(new_data)

# Convert to tensor
input_tensor = torch.tensor(scaled_new.toarray() if hasattr(scaled_new, "toarray") else scaled_new, dtype=torch.float32)

# Predict
with torch.no_grad():
    predicted_scaled = model(input_tensor).numpy()

# Convert back to original price scale
predicted_price = target_scaler.inverse_transform(predicted_scaled)

print("Predicted House Price: $", predicted_price[0][0])


def predict_house_price(new_data: pd.DataFrame):
    # Preprocess
    scaled = pipeline.transform(new_data)
    tensor = torch.tensor(scaled.toarray() if hasattr(scaled, "toarray") else scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        scaled_pred = model(tensor).numpy()
    unscaled_pred = target_scaler.inverse_transform(scaled_pred)
    return unscaled_pred[0][0]


price = predict_house_price(new_data)
print(f"Predicted Price: ${price:,.2f}")
