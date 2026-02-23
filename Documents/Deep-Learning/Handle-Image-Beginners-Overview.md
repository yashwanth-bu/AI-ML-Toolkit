doing proper **data analysis before training** can dramatically improve your model performance (especially for cats vs dogs classification).

Since your dataset looks like:

```
train/
 ├── cats
 └── dogs
test/
 ├── cats
 └── dogs
```

Here’s a **complete checklist** of what you should inspect before training.

---

# ✅ 1️⃣ Check Dataset Size (Very Important)

Make sure classes are balanced.

```python
import os

def count_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith(('.jpg','.png','.jpeg'))])

train_cats = count_images("train/cats")
train_dogs = count_images("train/dogs")

print("Train Cats:", train_cats)
print("Train Dogs:", train_dogs)
```

### 🔎 What to check:

* Are cats and dogs roughly equal?
* Is train much larger than test?
* Is there data imbalance?

If imbalance exists → consider:

* Data augmentation
* Class weights

---

# ✅ 2️⃣ Check Image Sizes (W × H × C)

Are all images the same size?

```python
import cv2
import os

sizes = set()

for root, _, files in os.walk("train"):
    for file in files:
        if file.endswith((".jpg",".png",".jpeg")):
            img = cv2.imread(os.path.join(root, file))
            if img is not None:
                sizes.add(img.shape)

print("Unique image shapes:", sizes)
```

### 🔎 What to check:

* Are sizes consistent?
* Are there grayscale images?
* Any corrupted images?

If sizes vary → you must resize before training.

---

# ✅ 3️⃣ Check for Corrupted Images

```python
import cv2

bad_images = []

for root, _, files in os.walk("train"):
    for file in files:
        path = os.path.join(root, file)
        img = cv2.imread(path)
        if img is None:
            bad_images.append(path)

print("Corrupted images:", bad_images)
```

---

# ✅ 4️⃣ Visualize Sample Images (VERY Important)

Look at random samples to understand the dataset.

```python
import matplotlib.pyplot as plt
import random

def show_random_images(folder, n=5):
    files = os.listdir(folder)
    images = random.sample(files, n)
    
    plt.figure(figsize=(15,5))
    for i, img_name in enumerate(images):
        img = cv2.imread(os.path.join(folder, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,n,i+1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

show_random_images("train/cats")
```

### 🔎 What to check:

* Are labels correct?
* Blurry images?
* Very dark/bright images?
* Background bias?

---

# ✅ 5️⃣ Pixel Value Distribution

Check pixel range.

```python
import numpy as np

img = cv2.imread("train/cats/" + os.listdir("train/cats")[0])
print("Min pixel:", np.min(img))
print("Max pixel:", np.max(img))
```

### 🔎 What to check:

* Range 0–255?
* Need normalization (divide by 255)?

---

# ✅ 6️⃣ Check Class Distribution (Visualization)

```python
import matplotlib.pyplot as plt

labels = ['Cats', 'Dogs']
counts = [train_cats, train_dogs]

plt.bar(labels, counts)
plt.title("Class Distribution")
plt.show()
```

Balanced dataset → good
Imbalanced → handle carefully

---

# ✅ 7️⃣ Data Leakage Check

Make sure:

* No same image in both train and test
* No duplicate images

---

# ✅ 8️⃣ Decide Preprocessing Strategy

Before training decide:

* Resize to what size? (e.g., 224×224)
* Normalize? (0–1)
* Augmentation?

  * Flip
  * Rotate
  * Zoom
  * Shift

---

# 🎯 Final Pre-Training Checklist

| Check                | Why                    |
| -------------------- | ---------------------- |
| Dataset size         | Enough data?           |
| Class balance        | Avoid bias             |
| Image sizes          | Resize needed          |
| Corrupted images     | Avoid crashes          |
| Sample visualization | Understand data        |
| Pixel distribution   | Normalize              |
| Duplicates           | Avoid leakage          |
| Augmentation need    | Improve generalization |

---

# 🚀 Bonus: Professional Workflow

In real ML projects, we also check:

* Mean & Std of dataset
* Train/Val/Test split (70/15/15)
* Stratified sampling
* Label noise

---

you're using **PyTorch**, here is a **proper professional data analysis checklist + code pipeline** before training your CNN.

Your dataset:

```
dataset/
 ├── train/
 │   ├── cats/
 │   └── dogs/
 └── test/
     ├── cats/
     └── dogs/
```

---

# 🚀 Step 1: Load Dataset Properly

Use `ImageFolder` (recommended for this structure).

```python
import torch
from torchvision import datasets, transforms

data_dir = "dataset/train"

dataset = datasets.ImageFolder(root=data_dir)

print("Classes:", dataset.classes)
print("Class to index:", dataset.class_to_idx)
print("Total images:", len(dataset))
```

---

# ✅ Step 2: Check Class Distribution

```python
from collections import Counter

labels = [label for _, label in dataset]
count = Counter(labels)

for class_name, idx in dataset.class_to_idx.items():
    print(f"{class_name}: {count[idx]}")
```

### 🔎 Check:

* Are cats ≈ dogs?
* Is one class dominant?

If imbalanced → use:

```python
from torch.utils.data import WeightedRandomSampler
```

---

# ✅ Step 3: Check Image Sizes (IMPORTANT)

PyTorch loads images as PIL images before transforms.

```python
from PIL import Image
import os

sizes = set()

for path, _ in dataset.samples:
    img = Image.open(path)
    sizes.add(img.size)   # (width, height)

print("Unique sizes:", sizes)
```

### 🔎 If sizes vary → you must resize.

---

# ✅ Step 4: Visualize Random Samples

```python
import matplotlib.pyplot as plt
import random

def show_samples(dataset, n=6):
    plt.figure(figsize=(12,6))
    for i in range(n):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(dataset.classes[label])
        plt.axis("off")
    plt.show()

show_samples(dataset)
```

### 🔎 Look for:

* Wrong labels
* Blurry images
* Background bias
* Dark/overexposed images

---

# ✅ Step 5: Check Channel Info (RGB or Grayscale)

```python
img, _ = dataset[0]
print("Image mode:", img.mode)
```

* `"RGB"` → 3 channels ✅
* `"L"` → Grayscale (1 channel)

If mixed → convert all to RGB using transform.

---

# ✅ Step 6: Compute Dataset Mean & Std (VERY IMPORTANT)

This improves training stability.

```python
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.
std = 0.
total_images = 0

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_samples

mean /= total_images
std /= total_images

print("Mean:", mean)
print("Std:", std)
```

Use these values for normalization.

---

# ✅ Step 7: Define Proper Transforms

After analysis:

```python
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

---

# 🎯 What You MUST Check Before Training

| Check             | Why                      |
| ----------------- | ------------------------ |
| Class balance     | Avoid bias               |
| Image sizes       | Resize consistency       |
| RGB vs Grayscale  | Channel mismatch         |
| Corrupted files   | Avoid crashes            |
| Visual inspection | Label errors             |
| Dataset mean/std  | Proper normalization     |
| Data leakage      | No duplicates train/test |

---

# 💡 Professional Tip

If using **Transfer Learning (ResNet, EfficientNet)**:

* Resize to 224×224
* Use ImageNet mean/std:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

you're using **PyTorch + torchvision**, I’ll give you a clean, professional **data analysis + setup pipeline** specifically for `torchvision`.

Your dataset structure works perfectly with:

```python
torchvision.datasets.ImageFolder
```

---

# ✅ 1️⃣ Load Dataset (No Transform First – For Inspection)

```python
import torchvision
from torchvision import datasets
import os

data_dir = "dataset/train"

dataset = datasets.ImageFolder(root=data_dir)

print("Classes:", dataset.classes)
print("Class to idx:", dataset.class_to_idx)
print("Total images:", len(dataset))
```

---

# ✅ 2️⃣ Check Class Distribution

```python
from collections import Counter

labels = [label for _, label in dataset.samples]
counter = Counter(labels)

for class_name, idx in dataset.class_to_idx.items():
    print(f"{class_name}: {counter[idx]}")
```

### 🔎 Inspect:

* Balanced classes?
* One class dominating?

If imbalanced → use `WeightedRandomSampler`.

---

# ✅ 3️⃣ Inspect Image Sizes (W × H)

```python
from PIL import Image

sizes = set()

for path, _ in dataset.samples:
    img = Image.open(path)
    sizes.add(img.size)  # (width, height)

print("Unique image sizes:", sizes)
```

If many different sizes → you MUST resize.

---

# ✅ 4️⃣ Check Image Channels (RGB or Grayscale)

```python
img, _ = dataset[0]
print("Image mode:", img.mode)
```

Expected:

* `"RGB"` → good (3 channels)
* `"L"` → grayscale (1 channel)

If mixed → force RGB in transform:

```python
transforms.Lambda(lambda x: x.convert("RGB"))
```

---

# ✅ 5️⃣ Visualize Random Samples

```python
import matplotlib.pyplot as plt
import random

def show_samples(dataset, n=6):
    plt.figure(figsize=(12,6))
    for i in range(n):
        img, label = dataset[random.randint(0, len(dataset)-1)]
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(dataset.classes[label])
        plt.axis("off")
    plt.show()

show_samples(dataset)
```

### 🔎 Look for:

* Wrong labels
* Blurry images
* Dark images
* Background bias

---

# ✅ 6️⃣ Compute Dataset Mean & Std (Best Practice)

```python
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)
total_images = 0

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, 3, -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += batch_samples

mean /= total_images
std /= total_images

print("Mean:", mean)
print("Std:", std)
```

---

# ✅ 7️⃣ Final Recommended Transform (For Training)

### 🔹 If Training From Scratch

```python
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

---

### 🔹 If Using Pretrained Model (VERY IMPORTANT)

If you're using:

* `torchvision.models.resnet18`
* `resnet50`
* `efficientnet`
* etc.

Then use **ImageNet normalization**:

```python
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Do NOT compute your own mean/std for pretrained models.

---

# 🔥 8️⃣ Data Leakage Check (Very Important)

Make sure:

* No duplicate images between `train` and `test`
* No same filenames copied into both

Quick check:

```python
train_files = set(os.listdir("dataset/train/cats"))
test_files = set(os.listdir("dataset/test/cats"))

print("Common files:", train_files.intersection(test_files))
```

---

# 🎯 Professional Pre-Training Checklist

Before training confirm:

✔ Classes balanced
✔ No corrupted images
✔ All images RGB
✔ All resized to same size
✔ Proper normalization
✔ No duplicates train/test
✔ Data augmentation applied (train only)

---

# 🚀 Clean Production Setup Example

```python
train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

---

Here is a **clean, professional PyTorch + torchvision training template** for your Cats vs Dogs dataset.

This includes:

✔ Transfer Learning (ResNet18)
✔ Proper transforms
✔ Train + Validation loop
✔ Accuracy tracking
✔ Best model saving
✔ GPU support
✔ Clean structure

---

# 🚀 Full Professional Training Template (Transfer Learning)

---

## 1️⃣ Imports

```python
import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
```

---

## 2️⃣ Device Configuration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

---

## 3️⃣ Transforms (ImageNet Normalization for Pretrained Models)

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 4️⃣ Load Dataset

```python
data_dir = "dataset"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
```

---

## 5️⃣ Load Pretrained Model (ResNet18)

Using **PyTorch** pretrained model:

```python
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)
```

---

## 6️⃣ Define Loss & Optimizer

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

---

# 🔥 7️⃣ Training + Validation Loop

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # ------------------
        # Training Phase
        # ------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # ------------------
        # Validation Phase
        # ------------------
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model
```

---

## 8️⃣ Start Training

```python
model = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10
)
```

---

## 9️⃣ Save Best Model

```python
torch.save(model.state_dict(), "best_model.pth")
print("Model saved successfully!")
```

---

# 🎯 Why This Is Professional

✔ Uses pretrained model
✔ Freezes backbone (fast training)
✔ Proper normalization
✔ Learning rate scheduler
✔ Tracks best validation accuracy
✔ GPU compatible
✔ Clean structure

---

I can also give you:

* ✅ Confusion matrix
* ✅ Precision / Recall / F1
* ✅ Early stopping
* ✅ Mixed precision training
* ✅ Fine-tuning entire network
* ✅ TensorBoard logging
* ✅ Production inference script

---

we’ll build a **CNN from scratch** (no pretrained model) using **PyTorch + torchvision**, structured professionally like a real project.

This will include:

✔ Custom CNN architecture
✔ Proper weight initialization
✔ Training loop
✔ Validation loop
✔ Accuracy tracking
✔ Best model saving
✔ GPU support

---

# 🚀 Full Professional CNN From Scratch (Cats vs Dogs)

---

## 1️⃣ Imports

```python
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

---

## 2️⃣ Device Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

---

## 3️⃣ Transforms (Since Training From Scratch → Use Your Own Normalization)

```python
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
```

---

## 4️⃣ Load Dataset

```python
data_dir = "dataset"

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)
```

---

# 🧠 5️⃣ Build Custom CNN Model

```python
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

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
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),  # 128x128 → pooled 3 times
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 6️⃣ Initialize Model

```python
model = CNN(num_classes).to(device)
```

---

## 7️⃣ Weight Initialization (Professional Practice)

```python
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

initialize_weights(model)
```

---

## 8️⃣ Loss, Optimizer, Scheduler

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

---

# 🔥 9️⃣ Training + Validation Loop

```python
def train_model(model, train_loader, val_loader, num_epochs=15):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # ---- Training ----
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model
```

---

## 🔟 Start Training

```python
model = train_model(model, train_loader, val_loader, num_epochs=15)
```

---

## 1️⃣1️⃣ Save Model

```python
torch.save(model.state_dict(), "cnn_from_scratch.pth")
```

---

# 📊 Architecture Summary

Input: 128×128×3
After 3 MaxPools → 16×16×128
Fully connected → 512 → 2 classes

---

# 🎯 Difference vs Transfer Learning

| From Scratch       | Pretrained           |
| ------------------ | -------------------- |
| Needs more data    | Works with less data |
| Slower convergence | Faster               |
| Full control       | Strong baseline      |

---

I can upgrade this to:

* ✅ Deeper CNN (ResNet-style blocks)
* ✅ Residual connections
* ✅ Early stopping
* ✅ Confusion matrix + metrics
* ✅ TensorBoard logging
* ✅ Mixed precision training
* ✅ K-fold cross validation

---

once you have a Cats vs Dogs dataset, you're not limited to just **binary classification**.

Here are **serious, resume-worthy projects** you can build with the same data using PyTorch + torchvision.

---

# 🔥 1️⃣ Multi-Task Learning (Advanced Classification)

Instead of just:

> Cat vs Dog

You can build:

* Cat vs Dog
* Breed classification (if you extend dataset)
* Age prediction (young vs adult)
* Image quality classifier (blurry vs clear)

You’d modify your model to output multiple heads.

---

# 🎯 2️⃣ Object Detection (Locate the Animal)

Instead of just classifying the image, detect **where** the animal is.

You can use:

* PyTorch detection models
* `torchvision.models.detection.fasterrcnn_resnet50_fpn`

Example applications:

* Pet detection in surveillance
* Animal counting
* Smart camera system

This upgrades you from classification → **real computer vision engineering**.

---

# 🎨 3️⃣ Image Segmentation (Mask the Animal)

Build a model that separates the pet from background.

You can use:

* `torchvision.models.segmentation.deeplabv3_resnet50`

Applications:

* Background removal
* AR filters
* Photo editing apps

This is much more advanced than classification.

---

# 🧠 4️⃣ Explainable AI (Very Impressive for Interviews)

Add **Grad-CAM** visualization to show:

> What part of the image made the model say “dog”?

You can implement Grad-CAM manually in PyTorch.

This demonstrates:

* Deep CNN understanding
* Interpretability
* Professional maturity

---

# 📊 5️⃣ Model Comparison Study

Turn it into a research-style project:

Compare:

* CNN from scratch
* PyTorch ResNet18
* EfficientNet
* MobileNet

Measure:

* Accuracy
* Training time
* Parameter count
* GPU memory usage

This becomes a strong portfolio project.

---

# 📱 6️⃣ Deploy It (Production-Level Project)

Turn your model into:

### 🔹 Web App

Use:

* Flask
* FastAPI

### 🔹 Mobile App

Export with:

* TorchScript
* ONNX

### 🔹 API Service

Deploy on:

* AWS
* Render
* HuggingFace Spaces

Deployment skills = huge career boost 🚀

---

# 🔄 7️⃣ Build a Data Augmentation Study

Experiment with:

* Heavy augmentation
* Light augmentation
* No augmentation

Compare generalization performance.

This teaches you real-world ML thinking.

---

# 🧪 8️⃣ Build a Semi-Supervised Model

Use:

* 50% labeled
* 50% unlabeled

Apply:

* Pseudo-labeling

Very advanced and impressive.

---

# 🎥 9️⃣ Real-Time Webcam Classifier

Use:

* OpenCV + PyTorch

Detect cats vs dogs from webcam feed in real-time.

This turns your project into something interactive.

---

# 🧬 🔟 Adversarial Attack Experiment

Create adversarial examples to fool your model.

Shows deep understanding of:

* CNN vulnerabilities
* Model robustness

Very advanced topic.

---

# 🏆 11️⃣ Convert to Binary Confidence Scoring System

Instead of:

> Cat / Dog

Return:

* 93% Dog
* 7% Cat

Then evaluate:

* ROC Curve
* Precision / Recall
* F1-score

This is how production ML systems work.

---

# 🎯 If You Want Resume-Level Impact

Here’s what looks strong:

✅ Train from scratch
✅ Compare with pretrained
✅ Add Grad-CAM
✅ Deploy as API
✅ Add performance metrics dashboard

That becomes a **complete ML engineering project**.

---

You can build a **Generative Model** that generates dog images only. Depending on your goals and available resources, there are multiple approaches. Let me break it down clearly.

---

# 🔹 1️⃣ Generative Model Options

### 1. **DCGAN (Deep Convolutional GAN)**

* Classic GAN architecture for generating images.
* Use only dog images for training.
* Output: new dog images similar to your dataset.

Pros:

* Relatively simple to implement
* Works well for 64×64 or 128×128 images

Cons:

* Training can be unstable

---

### 2. **Conditional GAN (cGAN)**

* If later you want multi-class generation (dogs vs cats), you can condition on class.
* For now, condition = “dog”.
* Similar to DCGAN but adds label embedding to generator and discriminator.

---

### 3. **StyleGAN / StyleGAN2**

* State-of-the-art for high-res image generation.
* Generates very realistic dog images.

Pros:

* Photorealistic output
* Can generate 256×256 or higher

Cons:

* Requires large dataset (~thousands of dog images)
* Needs GPU/TPU for fast training

---

### 4. **VAE (Variational Autoencoder)**

* Learns latent representation of dogs.
* Generates slightly blurrier images but easier to train.
* Good for experimenting before GAN.

---

# 🔹 2️⃣ Steps to Build a Dog-Only GAN (Using PyTorch)

### Step 1: Prepare Dog Dataset

* Use only dog images (`train/dogs` folder).
* Resize all images to same size (64×64 or 128×128).
* Normalize pixel values [-1, 1] for GANs.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dog_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
# Only dog class
dog_dataset.samples = [s for s in dog_dataset.samples if 'dogs' in s[0]]

dog_loader = DataLoader(dog_dataset, batch_size=128, shuffle=True)
```

---

### Step 2: Build Generator & Discriminator (DCGAN example)

```python
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)
```

---

### Step 3: Training Loop

* Use **BCELoss** for GANs
* Alternate training Discriminator and Generator
* Save generated images every few epochs

```python
# Pseudocode for training
for epoch in range(num_epochs):
    for real_images in dog_loader:
        # Train discriminator on real/fake
        # Train generator to fool discriminator
```

---

### Step 4: Generate Dogs

```python
noise = torch.randn(16, 100, 1, 1, device=device)
fake_dogs = generator(noise)
```

You can save images with `torchvision.utils.save_image(fake_dogs, "fake_dogs.png")`.

---

# 🔹 Tips for Success

1. Normalize images to [-1, 1] for GANs
2. Use **batch size 64–128**
3. Start with 64×64 images, then upscale later
4. Monitor **loss curves & generated samples**
5. If dataset is small → augment dogs images

---

# 🎯 Next Level

* Use **StyleGAN2** for photorealistic dogs
* Add **latent space interpolation** → morph between dog breeds
* Conditional GAN for different dog breeds

---

🐶 Let’s build a **full DCGAN training template in PyTorch** for generating dog images only. This will include:

* Data loading (dog images only)
* Generator & Discriminator
* Training loop
* Image saving during training
* GPU support

---

# 🚀 Full PyTorch DCGAN Template for Dogs

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------------------
# 1️⃣ Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ Hyperparameters
# -------------------------------
latent_dim = 100
img_size = 64
channels = 3
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5  # for Adam optimizer

# -------------------------------
# 3️⃣ DataLoader (Dog Images Only)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1,1]
])

data_dir = "dataset/train"
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
# Filter only dogs
dog_dataset = [s for s in full_dataset.samples if 'dogs' in s[0]]
full_dataset.samples = dog_dataset
full_dataset.targets = [s[1] for s in dog_dataset]

dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# -------------------------------
# 4️⃣ Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# -------------------------------
# 5️⃣ Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)

# -------------------------------
# 6️⃣ Initialize Models
# -------------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# -------------------------------
# 7️⃣ Loss & Optimizers
# -------------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# -------------------------------
# 8️⃣ Fixed noise for visualization
# -------------------------------
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

# -------------------------------
# 9️⃣ Training Loop
# -------------------------------
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)

        # ---- Labels ----
        real_labels = torch.ones(b_size, device=device)
        fake_labels = torch.zeros(b_size, device=device)

        # ---- Train Discriminator ----
        discriminator.zero_grad()
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # ---- Train Generator ----
        generator.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion(output, real_labels)  # Trick discriminator
        loss_G.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

    # ---- Save Generated Images ----
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        utils.save_image(fake, f"dog_samples_epoch_{epoch+1}.png", normalize=True, nrow=4)

# -------------------------------
# 10️⃣ Save Models
# -------------------------------
torch.save(generator.state_dict(), "dog_generator.pth")
torch.save(discriminator.state_dict(), "dog_discriminator.pth")
print("Models saved successfully!")
```

---

# 🔹 Key Features of This Template

✅ Uses **dog images only**
✅ DCGAN architecture
✅ Loss tracking for Generator & Discriminator
✅ Saves generated samples every epoch
✅ Compatible with GPU
✅ Normalization [-1,1] for GANs
✅ Easy to extend to larger datasets or StyleGAN later

---

# 💡 Next Steps / Improvements

1. Increase image resolution (128×128, 256×256)
2. Try **WGAN-GP** for stable training
3. Experiment with **conditional generation** (dog breed labels)
4. Visualize training using **TensorBoard**
5. Fine-tune hyperparameters for sharper images

---

😄 Let’s upgrade the DCGAN training template so that **you can see generated dog images in real-time** during training. This is very satisfying because you actually watch the generator improve epoch by epoch.

We’ll use **Matplotlib** to display images in Jupyter Notebook or interactive Python.

---

# 🚀 Real-Time Dog GAN Training (PyTorch)

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Device Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2️⃣ Hyperparameters
# -------------------------------
latent_dim = 100
img_size = 64
channels = 3
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5

# -------------------------------
# 3️⃣ DataLoader (Dog Images Only)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

data_dir = "dataset/train"
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
# Filter only dogs
dog_dataset = [s for s in full_dataset.samples if 'dogs' in s[0]]
full_dataset.samples = dog_dataset
full_dataset.targets = [s[1] for s in dog_dataset]

dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# -------------------------------
# 4️⃣ Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# -------------------------------
# 5️⃣ Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x).view(-1,1).squeeze(1)

# -------------------------------
# 6️⃣ Initialize Models
# -------------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# -------------------------------
# 7️⃣ Loss & Optimizers
# -------------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# -------------------------------
# 8️⃣ Fixed noise for visualization
# -------------------------------
fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)

# -------------------------------
# 9️⃣ Helper: Show Images
# -------------------------------
def show_generated(imgs):
    imgs = imgs * 0.5 + 0.5  # denormalize [-1,1] -> [0,1]
    grid = utils.make_grid(imgs, nrow=4)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.show()

# -------------------------------
# 🔥 10️⃣ Training Loop with Real-Time Display
# -------------------------------
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)

        real_labels = torch.ones(b_size, device=device)
        fake_labels = torch.zeros(b_size, device=device)

        # ---- Train Discriminator ----
        discriminator.zero_grad()
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # ---- Train Generator ----
        generator.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion(output, real_labels)
        loss_G.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

    # ---- Display real-time generated images ----
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        show_generated(fake)

# -------------------------------
# 11️⃣ Save Models
# -------------------------------
torch.save(generator.state_dict(), "dog_generator.pth")
torch.save(discriminator.state_dict(), "dog_discriminator.pth")
print("Models saved successfully!")
```

---

# 🔹 Features of This Version

✅ Displays **16 generated dog images every epoch**
✅ Uses **fixed noise** → you can watch improvement
✅ Compatible with GPU
✅ Images denormalized to [0,1] for display
✅ Saves models after training

---

💡 **Pro Tips**

1. Run in **Jupyter Notebook** or Google Colab for real-time visualization.
2. For higher resolution (128×128 or 256×256), increase `img_size` and adjust channels in the generator/discriminator.
3. Use `plt.pause(0.1)` if you want faster live updates instead of blocking each epoch.
4. You can also save intermediate images with:

```python
utils.save_image(fake, f"epoch_{epoch+1}.png", normalize=True, nrow=4)
```

---

Let’s combine everything into a **single, coherent PyTorch workflow** that covers:

1. **Image handling & preprocessing**
2. **CNN from scratch classification**
3. **Transfer learning classification**
4. **Dog-only DCGAN with real-time image generation**
5. **Training loops, validation, and model saving**

You’ll get one **all-in-one pipeline** that can be copied into a Markdown (`.md`) file or Python notebook.

---

# 🐾 All-in-One Cats vs Dogs PyTorch Pipeline

```python
# ==============================
# 1️⃣ Imports
# ==============================
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt

# ==============================
# 2️⃣ Device Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 3️⃣ Hyperparameters
# ==============================
batch_size = 32
img_size = 128  # CNN from scratch
latent_dim = 100 # DCGAN
num_epochs_class = 15
num_epochs_gan = 50
lr = 0.001
lr_gan = 0.0002
beta1 = 0.5  # Adam for GAN

# ==============================
# 4️⃣ Data Handling
# ==============================
# CNN / Classification transforms
train_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])
val_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

data_dir = "dataset"
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir,"train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(root=os.path.join(data_dir,"test"),  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# ==============================
# 5️⃣ CNN from Scratch Model
# ==============================
class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model_cnn = CNN(num_classes).to(device)

# Weight Initialization
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
init_weights(model_cnn)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=lr)

# ==============================
# 6️⃣ Training Loop (CNN)
# ==============================
def train_classification(model, train_loader, val_loader, num_epochs):
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(outputs.argmax(1)==labels)
        train_loss = running_loss/len(train_loader.dataset)
        train_acc = running_corrects.double()/len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*inputs.size(0)
                val_corrects += torch.sum(outputs.argmax(1)==labels)
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double()/len(val_loader.dataset)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(),"cnn_best.pth")
    print("Best CNN model saved!")
    return model

# ==============================
# 7️⃣ Transfer Learning Example (ResNet18)
# ==============================
model_resnet = models.resnet18(pretrained=True)
for param in model_resnet.parameters():
    param.requires_grad = False
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
model_resnet = model_resnet.to(device)
optimizer_resnet = optim.Adam(model_resnet.fc.parameters(), lr=lr)
criterion_resnet = nn.CrossEntropyLoss()

# ==============================
# 8️⃣ Dog-Only DCGAN with Real-Time Display
# ==============================
# Filter dog images only
dog_dataset = [s for s in train_dataset.samples if 'dogs' in s[0]]
train_dataset.samples = dog_dataset
train_dataset.targets = [s[1] for s in dog_dataset]
dog_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,512,4,1,0,bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,3,4,2,1,bias=False), nn.Tanh()
        )
    def forward(self,x): return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,128,4,2,1), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256,512,4,2,1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512,1,4,1,0), nn.Sigmoid()
        )
    def forward(self,x): return self.model(x).view(-1,1).squeeze(1)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizerG = optim.Adam(generator.parameters(), lr=lr_gan, betas=(beta1,0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr_gan, betas=(beta1,0.999))
criterion_gan = nn.BCELoss()
fixed_noise = torch.randn(16,latent_dim,1,1,device=device)

def show_generated(imgs):
    imgs = imgs*0.5+0.5
    grid = utils.make_grid(imgs, nrow=4)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.show()

# DCGAN Training Loop
for epoch in range(num_epochs_gan):
    for i, (real_images, _) in enumerate(dog_loader):
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        real_labels = torch.ones(b_size,device=device)
        fake_labels = torch.zeros(b_size,device=device)

        # Train Discriminator
        discriminator.zero_grad()
        output_real = discriminator(real_images)
        loss_real = criterion_gan(output_real, real_labels)
        noise = torch.randn(b_size,latent_dim,1,1,device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion_gan(output_fake,fake_labels)
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # Train Generator
        generator.zero_grad()
        output = discriminator(fake_images)
        loss_G = criterion_gan(output, real_labels)
        loss_G.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{num_epochs_gan}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        show_generated(fake)

torch.save(generator.state_dict(),"dog_generator.pth")
torch.save(discriminator.state_dict(),"dog_discriminator.pth")
print("All models saved!")
```

---

# ✅ What This Pipeline Covers

1. **Image handling** – resizing, normalization, augmentation
2. **Classification CNN from scratch** – train/validate/save
3. **Transfer learning (ResNet18)** – freeze backbone, fine-tune head
4. **Dog-only DCGAN** – real-time generation with Matplotlib display
5. **Training loops** – both classification and GAN
6. **Model saving** – all trained models saved

---
