# ----------------------------------------------------------
# save_embeddings.py
# Precompute embeddings for large image dataset
# ----------------------------------------------------------
import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from tqdm import tqdm

# -------------------------
# Folders
# -------------------------
folders = ["Data", "images"]

# -------------------------
# Load ResNet-50 model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()
model.to(device)

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# -------------------------
# Loop through images & extract features
# -------------------------
embeddings = []
image_paths = []

for folder in folders:
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for file in tqdm(files, desc=f"Processing {folder}"):
        path = os.path.join(folder, file)
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).cpu().numpy().squeeze()
            embeddings.append(feat)
            image_paths.append(path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

# -------------------------
# Save embeddings & paths
# -------------------------
np.save("embeddings.npy", np.array(embeddings))
np.save("image_paths.npy", np.array(image_paths))

print("✅ Saved embeddings.npy and image_paths.npy")
