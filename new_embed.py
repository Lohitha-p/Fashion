# save_embeddings_first_10k.py
import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from tqdm import tqdm  # progress bar

# ---------------------------------
# Settings
# ---------------------------------
DATA_FOLDER = "images"      # your folder with 40k images
OUTPUT_EMBEDDINGS = os.path.join(DATA_FOLDER, "embeddings.npy")
OUTPUT_NAMES = os.path.join(DATA_FOLDER, "names.npy")
MAX_IMAGES = 10000          # only process first 10k images

# ---------------------------------
# Device setup
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# Load model
# ---------------------------------
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # remove final classification layer
model.eval()
model.to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# ---------------------------------
# Collect image files
# ---------------------------------
all_images = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith((".jpg",".jpeg",".png"))])
all_images = all_images[:MAX_IMAGES]  # first 10k images

print(f"Processing {len(all_images)} images...")

# ---------------------------------
# Generate embeddings
# ---------------------------------
embeddings = []
names = []

with torch.no_grad():
    for img_name in tqdm(all_images):
        img_path = os.path.join(DATA_FOLDER, img_name)
        try:
            img_tensor = preprocess_image(img_path)
            emb = model(img_tensor).squeeze().cpu().numpy()
            embeddings.append(emb)
            names.append(img_name)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# ---------------------------------
# Save embeddings and names
# ---------------------------------
embeddings = np.array(embeddings)
names = np.array(names)

np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_NAMES, names)

print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
print(f"Saved image names to {OUTPUT_NAMES}")
