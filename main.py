# ---------------------------------
# Import required libraries
# ---------------------------------
import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------
# Load API key for Groq
# ---------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
else:
    st.error("❌ Missing GROQ_API_KEY in .env file. Styling advice will be unavailable.")

# ---------------------------------
# Streamlit App Layout
# ---------------------------------
st.set_page_config(page_title="AI Fashion Stylist", page_icon="🧥")

st.title("👗 AI Fashion Styling Assistant")
st.write("Upload an outfit image and/or enter a fashion question to get personalized advice and styling ideas!")

# File uploader widget
uploaded_file = st.file_uploader("📸 1. Upload an outfit image (optional):", type=["jpg", "jpeg", "png"])

# Optional text query input
user_query = st.text_input(
    "💬 2. Ask a styling question or describe an outfit idea (optional):",
    placeholder="e.g., 'What goes well with a black dress?' or 'Suggest an outfit for a beach vacation.'"
)

# ---------------------------------
# Load and Prepare the CNN Model
# ---------------------------------
@st.cache_resource
def load_model():
    """Loads the pretrained ResNet-50 model (feature extractor)."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

@st.cache_data
def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def get_embedding(model, image):
    """Extract embedding (feature vector) for the image."""
    with torch.no_grad():
        return model(preprocess_image(image)).squeeze().numpy()

# ---------------------------------
# Find Similar Images
# ---------------------------------
def find_similar_images(uploaded_image, data_folder, model, top_n=5):
    """Find top-N similar images based on cosine similarity."""
    uploaded_embedding = get_embedding(model, uploaded_image)
    similarities = []
    image_paths = []

    for filename in os.listdir(data_folder):
        if filename.startswith("train_image_") and filename.lower().endswith(".png"):
            img_path = os.path.join(data_folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                emb = get_embedding(model, img)
                sim = cosine_similarity(
                    uploaded_embedding.reshape(1, -1),
                    emb.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
                image_paths.append(img_path)
            except Exception as e:
                st.warning(f"⚠️ Could not process {filename}: {e}")

    if not similarities:
        st.error("❌ No valid images found in Data/ folder.")
        return []

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [image_paths[i] for i in top_indices]

# ---------------------------------
# Main Functionality
# ---------------------------------
if st.button("✨ Generate Styling Advice + Similar Looks"):
    # Handle different input modes
    has_image = uploaded_file is not None
    has_text = bool(user_query.strip())

    if not has_image and not has_text:
        st.warning("⚠️ Please upload an image or enter a fashion query first.")
        st.stop()

    model = load_model()
    data_folder = "Data"
    similar_images = []

    # If image provided → process and show similar looks
    if has_image:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Outfit", width='stretch')

        if not os.path.exists(data_folder):
            st.error("❌ Folder 'Data' not found! Please create it and add images named train_image_1.png, train_image_2.png, etc.")
        else:
            with st.spinner("Finding similar outfits..."):
                similar_images = find_similar_images(uploaded_image, data_folder, model)

            if similar_images:
                st.subheader("🖼️ Similar Looks Found in Dataset")
                cols = st.columns(len(similar_images))
                for i, img_path in enumerate(similar_images):
                    with cols[i]:
                        st.image(img_path, caption=os.path.basename(img_path),  width='stretch')
            else:
                st.info("No similar images found in dataset.")

    # ---------------------------------
    # Groq Styling Advice Generation
    # ---------------------------------
    if client:
        prompt_lines = ["You are a professional fashion stylist."]

        # Case 1: Only text
        if has_text and not has_image:
            prompt_lines.append(f"The user asks: '{user_query.strip()}'.")
            prompt_lines.append("Provide fashion ideas, outfit combinations, and styling suggestions related to the query.")

        # Case 2: Only image
        elif has_image and not has_text:
            prompt_lines.append("Analyze the uploaded outfit image and suggest styling ideas, matching accessories, and color recommendations.")

        # Case 3: Both image + text
        elif has_image and has_text:
            prompt_lines.append(f"The user also asks: '{user_query.strip()}'.")
            prompt_lines.append("Answer their question while incorporating styling advice for the uploaded outfit.")

        # Add image similarity context if applicable
        if similar_images:
            similar_image_names = [os.path.basename(p) for p in similar_images]
            prompt_lines.append(
                f"The most visually similar outfits from the dataset are: {', '.join(similar_image_names)}."
            )

        prompt = " ".join(prompt_lines)

        try:
            with st.spinner("🪄 Generating personalized styling advice..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a creative fashion stylist. Keep your advice concise and actionable."},
                        {"role": "user", "content": prompt},
                    ],
                )

            st.subheader("💡 AI Styling Suggestions:")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Error while calling Groq API: {e}")
    else:
        st.warning("Skipping styling advice generation because Groq client could not be initialized.")
