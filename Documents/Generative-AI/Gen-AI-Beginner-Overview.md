## **1️⃣ From CNN Classifier → Generative AI**

For image generation, we typically use models like:

* **GANs (Generative Adversarial Networks)** – two networks: Generator + Discriminator
* **Variational Autoencoders (VAEs)**
* **Diffusion Models** (state-of-the-art, but more complex)

Since you’re starting simple, we can implement a **GAN for dogs**.

---

## **2️⃣ Basic GAN Architecture**

1. **Generator (G)** – takes random noise (`z`) and generates fake dog images.
2. **Discriminator (D)** – tries to distinguish between **real dog images** and **fake ones** from G.

During training:

* G learns to produce realistic dog images to “fool” D
* D learns to get better at telling real vs fake

---

### **3️⃣ Example GAN for Dog Images (PyTorch)**

```python id="gd2n3l"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Image preprocessing
transform = transforms.Compose([
    transforms.Resize(64),  # small size for faster GAN training
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1, 1]
])

# 2️⃣ Dataset
dataset = datasets.ImageFolder(root="dataset/dogs_train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3️⃣ Generator
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

# 4️⃣ Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# 5️⃣ Initialize models
noise_dim = 100
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

# 6️⃣ Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 7️⃣ Training loop (simplified)
num_epochs = 50

for epoch in range(num_epochs):
    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_real = D(real_images)
        D_fake = D(fake_images.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_fake = D(fake_images)
        loss_G = criterion(D_fake, real_labels)  # want D to think fake is real

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Save sample generated images
    if (epoch+1) % 10 == 0:
        save_image(fake_images[:16], f"generated_dogs_epoch{epoch+1}.png", normalize=True)
```

---

### ✅ Key Notes:

1. **Generator** produces new dog images from random noise.
2. **Discriminator** tries to detect real vs fake dogs.
3. **Tanh output** in generator and normalization in dataset ensures pixel values in [-1,1].
4. Training is **adversarial** — G improves by trying to fool D.
5. You can increase `num_epochs` or image size (64→128→224) for better quality but slower training.

---

If you want, I can **upgrade this GAN to a full high-quality Generative AI pipeline** using:

* **Larger images (224x224)**
* **More advanced architectures like DCGAN or StyleGAN2**
* **Save best generator and discriminator for later image generation**

---

Perfect! For a **generative model** like a GAN, “testing and evaluation” is a bit different from classification:

* We **don’t have labels** for the generated images, so accuracy doesn’t apply.
* Instead, we evaluate **how good the generated images look** using:

  1. **Visual inspection** – generate images and look at them
  2. **Metrics** (optional advanced): FID, IS, or LPIPS scores for quality/diversity

I’ll show you a **practical way to “test” your GAN** on GPU and save samples for evaluation.

---

### **1️⃣ Load trained generator**

```python
# Initialize generator
G = Generator(noise_dim=100).to(device)

# Load trained weights (from saved model)
G.load_state_dict(torch.load("generator_best.pth"))
G.eval()  # set to evaluation mode
```

---

### **2️⃣ Generate new dog images**

```python
import torch
from torchvision.utils import save_image

num_samples = 16  # number of dog images to generate
z = torch.randn(num_samples, 100).to(device)  # random noise
with torch.no_grad():
    fake_images = G(z)

# Save images for visual inspection
save_image(fake_images, "test_generated_dogs.png", nrow=4, normalize=True)
print("Generated dog images saved as test_generated_dogs.png")
```

* `normalize=True` converts the output from `[-1,1]` to `[0,1]` for viewing.
* `nrow=4` saves 4 images per row in a grid.

---

### **3️⃣ Optional: Advanced evaluation metrics**

If you want **numerical evaluation**:

1. **FID (Fréchet Inception Distance)** – measures similarity between generated and real images.
2. **IS (Inception Score)** – evaluates quality and diversity of generated images.

Example libraries:

```bash
pip install pytorch-fid
```

```bash
# Run FID between your generated images and real dog dataset
pytorch-fid path_to_real_dogs path_to_generated_dogs
```

---

### ✅ Summary

* **Testing a GAN** = generate images from random noise + inspect visually
* Optionally compute metrics like FID/IS for automated evaluation
* Can save multiple batches to see diversity

---

Perfect! Let’s build a **full end-to-end GAN pipeline for dogs** with:

1. **Generator & Discriminator**
2. **Training loop with adversarial loss**
3. **Validation-like evaluation** – we generate a batch of images every few epochs to visually check quality
4. **Automatic saving of the best generator** based on discriminator loss or a simple heuristic

We’ll also keep it **GPU-ready**.

---

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1️⃣ Dataset and Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(64),   # 64x64 for faster GAN training
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1,1]
])

dataset = datasets.ImageFolder(root="dataset/dogs_train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create folder to save generated images
os.makedirs("generated_images", exist_ok=True)

# -------------------------------
# 2️⃣ Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# -------------------------------
# 3️⃣ Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# -------------------------------
# 4️⃣ Initialize models, loss, optimizers
# -------------------------------
noise_dim = 100
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5,0.999))

# -------------------------------
# 5️⃣ Training loop with evaluation
# -------------------------------
num_epochs = 50
best_generator_loss = float("inf")

for epoch in range(num_epochs):
    G.train()
    D.train()
    running_loss_G = 0.0
    running_loss_D = 0.0

    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_real = D(real_images)
        D_fake = D(fake_images.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

        # --- Train Generator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_fake = D(fake_images)
        loss_G = criterion(D_fake, real_labels)  # want discriminator to think fake is real

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()

    avg_loss_D = running_loss_D / len(loader)
    avg_loss_G = running_loss_G / len(loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}")

    # -------------------------------
    # 6️⃣ Validation-like evaluation: generate sample images every 5 epochs
    # -------------------------------
    if (epoch+1) % 5 == 0:
        G.eval()
        with torch.no_grad():
            z = torch.randn(16, noise_dim).to(device)
            fake_images = G(z)
            save_image(fake_images, f"generated_images/epoch{epoch+1}.png", nrow=4, normalize=True)
        print(f"Saved generated images for visual evaluation at epoch {epoch+1}")

    # -------------------------------
    # 7️⃣ Save best generator automatically
    # -------------------------------
    if avg_loss_G < best_generator_loss:
        best_generator_loss = avg_loss_G
        torch.save(G.state_dict(), "best_generator.pth")
        print(f"Saved best generator at epoch {epoch+1} with loss {best_generator_loss:.4f}")

# -------------------------------
# 8️⃣ Final evaluation: generate test images
# -------------------------------
G.eval()
with torch.no_grad():
    z = torch.randn(16, noise_dim).to(device)
    fake_images = G(z)
    save_image(fake_images, "generated_images/final_generated_dogs.png", nrow=4, normalize=True)
print("Final generated dog images saved.")
```

---

### ✅ Features of this pipeline:

1. **Generator & Discriminator training** in adversarial loop
2. **Validation-like evaluation** – generates 16 sample images every 5 epochs for visual inspection
3. **Automatic saving of the best generator** based on generator loss
4. **GPU-ready** with `device`
5. **Final test images** saved after all epochs

---

Perfect! Once you’ve **trained your GAN and saved the best generator**, you can **load it later** and generate new dog images anytime — no retraining needed. Here’s how to do it:

---

### **1️⃣ Load the saved generator**

```python
import torch
from torchvision.utils import save_image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the Generator architecture (must match training)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Initialize model
noise_dim = 100
G = Generator(noise_dim=noise_dim).to(device)

# Load saved weights
G.load_state_dict(torch.load("best_generator.pth", map_location=device))
G.eval()  # set to evaluation mode
```

---

### **2️⃣ Generate new dog images**

```python
import torch
from torchvision.utils import save_image

num_samples = 16  # number of dog images to generate
z = torch.randn(num_samples, noise_dim).to(device)  # random noise

with torch.no_grad():
    fake_images = G(z)

# Save generated images to disk
save_image(fake_images, "generated_dogs.png", nrow=4, normalize=True)
print("Generated dog images saved as generated_dogs.png")
```

* `normalize=True` converts pixel values from `[-1, 1]` to `[0,1]` so you can view them.
* `nrow=4` arranges images in a 4×4 grid.

---

### **3️⃣ Optional: Generate a single image for preview**

```python
single_z = torch.randn(1, noise_dim).to(device)
with torch.no_grad():
    single_image = G(single_z)
save_image(single_image, "generated_single_dog.png", normalize=True)
```

* This is useful for **interactive testing** or embedding in an app.

---

### ✅ Summary

1. **Recreate the generator architecture** exactly as during training
2. **Load the saved weights** using `load_state_dict`
3. **Set model to `eval()`** mode
4. **Generate images from random noise** with `G(z)`
5. **Save or display** images for evaluation

---

Perfect! Let’s make a **reusable function** to generate any number of dog images from your saved generator. This way, you can just call it anytime without repeating code.

---

```python
import torch
from torchvision.utils import save_image
import os

def generate_dog_images(generator_path, num_images=16, noise_dim=100, save_dir="generated_dogs"):
    """
    Generate dog images from a saved GAN generator.

    Args:
        generator_path (str): Path to the saved generator weights (.pth file)
        num_images (int): Number of images to generate
        noise_dim (int): Dimension of random noise vector
        save_dir (str): Directory to save generated images
    """
    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------
    # Recreate Generator architecture
    # -------------------------------
    class Generator(torch.nn.Module):
        def __init__(self, noise_dim=noise_dim, img_channels=3):
            super().__init__()
            self.gen = torch.nn.Sequential(
                torch.nn.Linear(noise_dim, 256*8*8),
                torch.nn.BatchNorm1d(256*8*8),
                torch.nn.ReLU(True),
                torch.nn.Unflatten(1, (256, 8, 8)),
                torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(True),
                torch.nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
                torch.nn.Tanh()
            )

        def forward(self, x):
            return self.gen(x)

    # -------------------------------
    # Initialize and load weights
    # -------------------------------
    G = Generator().to(device)
    G.load_state_dict(torch.load(generator_path, map_location=device))
    G.eval()

    # Create save directory if it doesn’t exist
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------
    # Generate images
    # -------------------------------
    z = torch.randn(num_images, noise_dim).to(device)
    with torch.no_grad():
        fake_images = G(z)

    # Save generated images in a grid
    save_path = os.path.join(save_dir, "generated_grid.png")
    save_image(fake_images, save_path, nrow=4, normalize=True)
    print(f"{num_images} dog images generated and saved to {save_path}")
```

---

### **Usage Example**

```python
# Generate 16 dog images using the saved generator
generate_dog_images("best_generator.pth", num_images=16, noise_dim=100)

# Generate 32 images and save to a different folder
generate_dog_images("best_generator.pth", num_images=32, noise_dim=100, save_dir="dogs_batch2")
```

---

### ✅ Advantages of this function:

1. **Reusable** – you don’t need to rewrite loading and generating logic
2. **Customizable** – choose number of images, noise dimension, and save directory
3. **GPU-ready** – automatically uses CUDA if available
4. **Saves images in a grid** for easy visual inspection

---

Perfect — now we’re back to **fine-tuning a saved GAN generator** on a **new dataset of dog images**. This is similar to how we fine-tuned a classification model, but for a generative model, it’s a bit different.

We’ll go **step by step**.

---

## **1️⃣ Load the saved generator and discriminator**

You need **both** models (G and D) because GAN training is **adversarial**.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate Generator (must match saved architecture)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256*8*8),
            nn.BatchNorm1d(256*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

# Recreate Discriminator (must match original)
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)

# Initialize
noise_dim = 100
G = Generator(noise_dim=noise_dim).to(device)
D = Discriminator().to(device)

# Load saved generator weights
G.load_state_dict(torch.load("best_generator.pth", map_location=device))
G.train()  # fine-tuning requires train mode
D.train()
```

---

## **2️⃣ Load new dataset**

```python
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

new_dataset = datasets.ImageFolder(root="dataset/new_dogs", transform=transform)
loader = DataLoader(new_dataset, batch_size=64, shuffle=True)
```

---

## **3️⃣ Set up optimizers and loss**

```python
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))  # lower LR for fine-tuning
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
```

* Lower learning rate is important so you **don’t destroy the previous generator’s knowledge**.

---

## **4️⃣ Fine-tuning loop**

```python
num_epochs = 10  # usually fewer epochs for fine-tuning

for epoch in range(num_epochs):
    running_loss_G = 0.0
    running_loss_D = 0.0

    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_real = D(real_images)
        D_fake = D(fake_images.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

        # --- Train Generator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_fake = D(fake_images)
        loss_G = criterion(D_fake, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {running_loss_D/len(loader):.4f} | Loss G: {running_loss_G/len(loader):.4f}")

    # Optionally save generated images for visual evaluation
    if (epoch+1) % 2 == 0:
        with torch.no_grad():
            z = torch.randn(16, noise_dim).to(device)
            fake_images = G(z)
            from torchvision.utils import save_image
            save_image(fake_images, f"new_generated_epoch{epoch+1}.png", nrow=4, normalize=True)

# Save fine-tuned generator
torch.save(G.state_dict(), "fine_tuned_generator.pth")
print("Fine-tuned generator saved!")
```

---

### **✅ Key Points**

1. **Load saved generator** → keep learned features
2. **Use new dataset** → fine-tune on new styles of dogs
3. **Lower learning rate** → prevent forgetting previous knowledge
4. **Optionally freeze layers** → you could freeze early layers of G if dataset is small
5. **Save fine-tuned generator** → can be used to generate new images anytime

---

Ahhh, got it! You want to **upgrade your GAN / generative model** so it can produce **better dog images** — basically improving your Generator & overall GAN training. Let’s break it down **step by step** and give you a practical upgrade plan.

---

## **1️⃣ Make Generator & Discriminator Stronger**

### **Generator Upgrades**

* **Use more layers and channels** → allows generating higher-res details
* **Residual blocks or skip connections** → helps gradient flow
* **BatchNorm** in all layers except the last
* **Tanh output** for images normalized between [-1,1]

**Example Upgrade (Generator)**

```python id="gen_upgraded"
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
```

* Now outputs **64x64 → can go 128x128** by adding more layers

---

### **Discriminator Upgrades**

* **Deeper CNN** → more channels: 64 → 128 → 256 → 512
* **LeakyReLU** for stable gradients
* **Spectral Normalization** → helps D not overpower G

```python id="disc_upgraded"
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
```

---

## **2️⃣ Training Improvements**

1. **Wasserstein GAN (WGAN) + Gradient Penalty**

   * More stable than normal BCE loss
   * Helps G produce sharper images

2. **TTUR (Two Time-scale Update Rule)**

   * Use slightly different learning rates:
     `lr_G = 0.0001, lr_D = 0.0004`

3. **Label Smoothing**

   * Real labels = 0.9, Fake = 0.1 → prevents D from being too confident

4. **Noise Injection**

   * Add tiny Gaussian noise to inputs of D → stabilizes training

5. **Progressive Growing**

   * Start with 64x64 → then 128x128 → 224x224
   * Generates higher-res images gradually

---

## **3️⃣ Data Augmentation for GAN**

* Random flips, rotations, color jitter for **real images only**
* Makes Discriminator more robust → Generator learns faster

```python id="data_aug"
transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

---

## **4️⃣ Save & Evaluate Frequently**

* Generate images every few epochs
* Save the **best generator** based on visual quality or discriminator loss
* Optionally compute **FID score** for objective evaluation

---

### ✅ Summary of GAN Improvements

* **Generator**: deeper, more channels, BatchNorm, maybe residual blocks
* **Discriminator**: deeper, spectral normalization, leaky ReLU
* **Training tricks**: WGAN loss, TTUR, label smoothing, progressive growing
* **Dataset**: augment real images
* **Evaluation**: save generated images + compute FID

---

Perfect! Let’s upgrade your GAN to a **stronger, more production-ready version** for generating dog images. This includes:

* **Deeper Generator & Discriminator**
* **BatchNorm, LeakyReLU, SpectralNorm**
* **Data augmentation for real images**
* **Training loop with automatic saving of best generator**
* **Sample image generation every few epochs**

Here’s the full code:

---

```python id="upgraded_gan_dogs"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1️⃣ Dataset and Transform (with augmentation)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root="dataset/dogs_train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

os.makedirs("generated_images", exist_ok=True)

# -------------------------------
# 2️⃣ Upgraded Generator
# -------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 512*4*4),
            nn.BatchNorm1d(512*4*4),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# -------------------------------
# 3️⃣ Upgraded Discriminator
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# -------------------------------
# 4️⃣ Initialize models, loss, optimizers
# -------------------------------
noise_dim = 100
G = Generator(noise_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0004, betas=(0.5, 0.999))

# -------------------------------
# 5️⃣ Training loop with evaluation and saving
# -------------------------------
num_epochs = 50
best_loss_G = float("inf")

for epoch in range(num_epochs):
    running_loss_G = 0.0
    running_loss_D = 0.0

    G.train()
    D.train()

    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(device) * 0.9  # label smoothing
        fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1

        # --- Train Discriminator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_real = D(real_images)
        D_fake = D(fake_images.detach())
        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        running_loss_D += loss_D.item()

        # --- Train Generator ---
        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(z)
        D_fake = D(fake_images)
        loss_G = criterion(D_fake, real_labels)  # want D to think fake is real

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        running_loss_G += loss_G.item()

    avg_loss_D = running_loss_D / len(loader)
    avg_loss_G = running_loss_G / len(loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}")

    # -------------------------------
    # Validation-like evaluation: generate sample images every 5 epochs
    # -------------------------------
    if (epoch+1) % 5 == 0:
        G.eval()
        with torch.no_grad():
            z = torch.randn(16, noise_dim).to(device)
            fake_images = G(z)
            save_image(fake_images, f"generated_images/epoch{epoch+1}.png", nrow=4, normalize=True)
        print(f"Saved sample images for epoch {epoch+1}")

    # -------------------------------
    # Save best generator automatically
    # -------------------------------
    if avg_loss_G < best_loss_G:
        best_loss_G = avg_loss_G
        torch.save(G.state_dict(), "best_generator.pth")
        print(f"Saved best generator at epoch {epoch+1}")

# -------------------------------
# 6️⃣ Final evaluation: generate test images
# -------------------------------
G.eval()
with torch.no_grad():
    z = torch.randn(16, noise_dim).to(device)
    fake_images = G(z)
    save_image(fake_images, "generated_images/final_generated_dogs.png", nrow=4, normalize=True)
print("Final generated dog images saved!")
```

---

### **✅ Key Upgrades**

1. **Deeper Generator & Discriminator** → more capacity for realistic images
2. **SpectralNorm & BatchNorm** → stabilize training
3. **LeakyReLU** → better gradient flow in D
4. **Data Augmentation** → random flips/rotation for real images
5. **Label Smoothing** → prevents D from overpowering G
6. **Automatic best generator saving** → based on generator loss
7. **Periodic sample generation** → easy visual inspection

---
