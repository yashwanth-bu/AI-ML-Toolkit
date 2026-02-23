Let’s take your full code and **update it properly for GPU usage**, including moving the model and each batch to the GPU (`device`) and keeping everything else the same.

Here’s the corrected version:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3️⃣ Load dataset
dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4️⃣ Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 111 * 111, 2)  # 2 classes: dog and cat
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# 5️⃣ Initialize model, loss, optimizer
model = SimpleCNN().to(device)  # ✅ move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training loop
for images, labels in loader:
    # ✅ move batch to GPU
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"Batch Loss: {loss.item():.4f}")
```

---

### ✅ Key GPU Changes:

1. `model.to(device)` → model is on GPU.
2. `images.to(device), labels.to(device)` → each batch is on GPU.
3. The rest of your code stays the same — CrossEntropyLoss and Adam work fine on GPU.

---

Optional next step: you can **wrap the training in epochs** and add validation — otherwise, it only runs **one pass over the dataset**.

If you want, I can also **rewrite this CNN so it automatically calculates the input size for the Linear layer**, so you don’t have to hardcode `16*111*111` — much safer for future changes.

---

Perfect! Let’s extend your code so it **includes training over multiple epochs** and also **testing/evaluation** on a separate dataset. I’ll write a full, GPU-ready version with proper evaluation.

---

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3️⃣ Load datasets
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",  # make sure you have a test folder
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4️⃣ Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 111 * 111, 2)  # 2 classes
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# 5️⃣ Initialize model, loss, optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training
num_epochs = 5

for epoch in range(num_epochs):
    model.train()  # set to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 7️⃣ Evaluation
model.eval()  # set to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # no gradient calculation needed
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # pick class with highest logit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

### ✅ What’s new in this extended version:

1. **Epochs**: Training runs multiple passes (`num_epochs`) over your training dataset.
2. **Evaluation mode**: `model.eval()` disables dropout/batchnorm (not used here but good practice).
3. **`torch.no_grad()`**: prevents PyTorch from storing gradients during testing → faster and less memory.
4. **Accuracy calculation**: counts how many predictions match the true labels.
5. **Separate test dataset**: make sure you have `dataset/test` structured like your train folder.

---

Absolutely! Let’s add **validation during training** so you can monitor how your model performs on unseen data after each epoch. We’ll keep GPU support and evaluation at the end as well.

Here’s the updated version with **training, validation, and test evaluation**:

---

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3️⃣ Load datasets
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="dataset/val",  # validation set folder
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4️⃣ Define CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 111 * 111, 2)  # 2 classes
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# 5️⃣ Initialize model, loss, optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training with validation
num_epochs = 5

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

# 7️⃣ Final Test Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

### ✅ Key Changes for Validation:

1. Added `val_dataset` and `val_loader`.
2. During each epoch, after training:

   * Set `model.eval()`
   * Compute **validation loss** and **accuracy** without gradients.
3. Print both **train loss** and **validation loss/accuracy** per epoch.

---

This way you can **monitor overfitting** — if training loss goes down but validation loss rises, your model is memorizing the train set.

---

Perfect! Let’s make your CNN **deeper and more robust**, so it can learn better features for images.

We’ll do a few improvements:

1. **Multiple convolutional blocks** → extract more hierarchical features.
2. **Batch Normalization** → helps with training stability.
3. **Dropout** → prevents overfitting.
4. **Adaptive pooling** → so we don’t have to manually calculate input size for the linear layer.

Here’s an **updated CNN**:

```python
import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # --- Convolutional layers ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # output: 128 x 4 x 4

        # --- Fully connected layers ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
```

---

### ✅ What changed:

| Feature                                | Benefit                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------- |
| 3 conv blocks (32 → 64 → 128 channels) | Extracts hierarchical features (edges → textures → complex patterns)    |
| BatchNorm2d                            | Stabilizes and accelerates training                                     |
| AdaptiveAvgPool2d                      | Fixes input size to Linear layer automatically (no manual `16*111*111`) |
| Dropout                                | Reduces overfitting                                                     |
| Extra Linear layer                     | More capacity to learn complex mappings                                 |

---

### How to use it in your training:

```python
model = ImprovedCNN(num_classes=2).to(device)
```

Then training, validation, and test loops remain **exactly the same** as before.

---

Perfect! Let’s put everything together: **ImprovedCNN**, GPU-ready training, validation, and final testing — all in one clean script. This version is fully flexible with input size thanks to `AdaptiveAvgPool2d` and includes deeper convolutional layers for better training.

---

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3️⃣ Load datasets
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="dataset/val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4️⃣ Improved CNN
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to a fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # output: 128 x 4 x 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# 5️⃣ Initialize model, loss, optimizer
model = ImprovedCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ Training with validation
num_epochs = 5

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

# 7️⃣ Final Test Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

### ✅ Features of this version:

1. **Deeper CNN** with 3 conv blocks (32 → 64 → 128 channels)
2. **BatchNorm + Dropout** for better training and regularization
3. **Adaptive pooling** → no need to manually calculate input size for Linear layer
4. **Full training loop with validation**
5. **GPU-ready** (model and batches moved to device)
6. **Final test evaluation**

---

Absolutely! Saving the **best model** during training is crucial, so you can **reuse it later without retraining**. In PyTorch, this is usually done using `torch.save()` for the model’s `state_dict`.

Here’s how to update your code to **save the best model based on validation accuracy** and then **load it later for inference or further evaluation**.

---

### 1️⃣ Saving the best model

Update your training loop like this:

```python
best_val_accuracy = 0.0  # to track best validation accuracy

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")
    
    # --- Save best model ---
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")  # save model weights
        print(f"Best model saved with Val Accuracy: {best_val_accuracy:.2f}%")
```

---

### 2️⃣ Loading the saved model later

```python
# Initialize the same model architecture
model = ImprovedCNN(num_classes=2).to(device)

# Load the saved weights
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # set to evaluation mode
```

---

### 3️⃣ Using it for prediction

```python
# Example: predict a batch from test_loader
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print(predicted)  # predicted class indices
        break  # just first batch
```

---

### ✅ Notes:

1. **`state_dict`** saves **weights only**, not the entire model object.

   * Pros: smaller, more flexible, recommended for PyTorch.
   * Con: you must define the model architecture when loading.
2. You can also save **entire model** using `torch.save(model, "model.pth")`, but it’s less portable.
3. `model.eval()` is important when loading for inference.

---

Ah, perfect! You’re talking about **fine-tuning** or **continued training** on a previously trained model — exactly what you do when you have new data but don’t want to start from scratch. Let me break it down clearly.

---

### 1️⃣ Load your saved model

First, you load the architecture and the weights:

```python
# 1️⃣ Initialize model architecture
model = ImprovedCNN(num_classes=2).to(device)

# 2️⃣ Load the saved weights
model.load_state_dict(torch.load("best_model.pth"))

# 3️⃣ Set to train mode if you want to continue training
model.train()
```

---

### 2️⃣ Prepare your new dataset

Make a **new DataLoader** for your new images (fine-tuning data):

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

new_dataset = datasets.ImageFolder(
    root="dataset/new_data",
    transform=transform
)

new_loader = DataLoader(new_dataset, batch_size=8, shuffle=True)
```

---

### 3️⃣ Fine-tune the model

You can **continue training** with your optimizer and loss:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # lower lr recommended

num_epochs = 3  # fine-tuning usually fewer epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in new_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(new_loader):.4f}")
```

---

### ⚡ Key Tips for Fine-tuning

1. **Lower learning rate**

   * Usually `0.0001` works better than your original training LR, so you don’t “destroy” the pretrained weights.

2. **Freeze some layers (optional)**

   * If your new data is similar, you can **freeze early convolutional layers** to keep their features:

   ```python
   for param in model.features[:4].parameters():
       param.requires_grad = False
   ```

   * Then only the remaining layers will be updated.

3. **Use validation set**

   * Even during fine-tuning, track validation loss/accuracy to prevent overfitting.

4. **Save the updated model**

   ```python
   torch.save(model.state_dict(), "fine_tuned_model.pth")
   ```

---

✅ **Summary:**

* Load old model + weights → prepare new dataset → fine-tune with lower LR → optionally freeze layers → save new model.

---

Perfect! Let’s write a **full fine-tuning pipeline**. This will:

1. Load your **previously saved model**
2. Prepare **new training and validation datasets**
3. Fine-tune on GPU
4. Track **validation accuracy**
5. Save the **fine-tuned model** for future use

---

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing for new data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3️⃣ Load new datasets
new_train_dataset = datasets.ImageFolder(
    root="dataset/new_train",
    transform=transform
)

new_val_dataset = datasets.ImageFolder(
    root="dataset/new_val",
    transform=transform
)

train_loader = DataLoader(new_train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(new_val_dataset, batch_size=8, shuffle=False)

# 4️⃣ Define the same CNN architecture
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4,4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# 5️⃣ Load saved model weights
model = ImprovedCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("best_model.pth"))

# Optional: Freeze some layers if needed
# for param in model.features[:4].parameters():
#     param.requires_grad = False

# 6️⃣ Loss and optimizer (lower LR for fine-tuning)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 7️⃣ Fine-tuning loop with validation
num_epochs = 3
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

    # --- Save best fine-tuned model ---
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "fine_tuned_model.pth")
        print(f"Saved Fine-tuned Model with Val Accuracy: {best_val_accuracy:.2f}%")

# 8️⃣ The fine-tuned model can now be loaded later
# model.load_state_dict(torch.load("fine_tuned_model.pth"))
# model.eval()
```

---

### ✅ Key Points:

1. **Lower learning rate** (`0.0001`) for fine-tuning so pretrained weights aren’t destroyed.
2. **Optional freezing**: freeze early conv layers if new data is small or similar.
3. Tracks **validation accuracy** and saves the **best fine-tuned model**.
4. After training, you can **load `fine_tuned_model.pth`** for inference on new images.

---

Ah! Now we’re talking about the big concept: **Transfer Learning** — it’s closely related to what we just did with fine-tuning, but let me explain it properly and clearly.

---

### **1️⃣ What is Transfer Learning?**

**Transfer Learning** is the idea of **reusing a pre-trained model** (trained on a large dataset) for a **new, related task**.

Instead of training a network from scratch, you “transfer” knowledge learned from a big dataset to your smaller dataset.

---

### **2️⃣ How it works in practice**

Let’s say:

* You have a **pretrained model** like **ResNet**, trained on **ImageNet** (1.2M images, 1000 classes).
* You have a small dataset of **cats vs dogs** (like yours).

Instead of training from scratch:

1. Load the pretrained model: `resnet18(pretrained=True)`
2. Replace the **final classification layer** to match your number of classes (2 for dog/cat)
3. Optionally **freeze early layers** (they detect general features like edges, textures)
4. Train/fine-tune **only the last layers or entire network** on your dataset

---

### **3️⃣ Why Transfer Learning works**

* Early layers in CNNs learn **general features** (edges, corners, textures)
* Later layers learn **task-specific features**
* By reusing pretrained weights, you:

  * Train **faster**
  * Need **less data**
  * Usually get **better accuracy**

---

### **4️⃣ Example in PyTorch**

```python
from torchvision import models
import torch.nn as nn

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze early layers (optional)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (original: 1000 classes → new: 2 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: dog/cat
```

Then you can train only the last layer (or fine-tune the whole model) on your dataset.

---

### **5️⃣ Relation to what we did earlier**

* **Fine-tuning your saved CNN** is basically **transfer learning within your own trained model**.
* Traditional transfer learning usually starts from a **pretrained model trained on a large generic dataset** like ImageNet.

---

✅ **TL;DR:**

* Transfer Learning = reuse a pretrained network for a new task
* Saves time, data, and often gives better accuracy
* Fine-tuning = adjusting a pretrained network on your dataset

---

Perfect! Let’s do **transfer learning with a pretrained ResNet18** for your cat vs dog classification. This is usually **faster and more accurate** than training from scratch.

We’ll:

1. Load **ResNet18 pretrained on ImageNet**
2. Replace the **final layer** for 2 classes
3. Freeze the early layers to keep general features
4. Train only the last layers (or fine-tune all layers if needed)

Here’s the full GPU-ready code:

---

```python
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1️⃣ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225]) # ImageNet std
])

# 3️⃣ Load datasets
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="dataset/val", transform=transform)
test_dataset  = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4️⃣ Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze early layers (optional, keeps learned features)
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: dog/cat
model = model.to(device)

# 5️⃣ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # train only last layer

# 6️⃣ Training with validation
num_epochs = 5
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")
    
    # --- Save best model ---
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "resnet18_best_model.pth")
        print(f"Saved Best Model with Val Accuracy: {best_val_accuracy:.2f}%")

# 7️⃣ Final Test Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

### ✅ Key Points:

1. **Pretrained ResNet18** gives you strong general features.
2. **Freeze early layers** so you don’t destroy learned ImageNet features.
3. Only the **last layer is trained** on your small dataset → fast and data-efficient.
4. You can **fine-tune the whole network** later by unfreezing more layers.
5. Best model is **saved automatically** based on validation accuracy.

---
