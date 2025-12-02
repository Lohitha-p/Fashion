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
    st.error("❌ GROQ_API_KEY missing in .env")

# ---------------------------------
# App UI
# ---------------------------------
st.set_page_config(page_title="AI Fashion Stylist", page_icon="👗")
st.title("👗 AI Fashion Styling Assistant")
st.write("Upload an outfit image and ask any style question!")

uploaded_file = st.file_uploader("📸 Upload image (optional):", type=["jpg", "jpeg", "png"])
user_query = st.text_input(
    "💬 Ask a styling question (optional):",
    placeholder="Example: Suggest an outfit for a dinner date."
)

# ---------------------------------
# CNN Model
# ---------------------------------
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(img):
    return transform(img).unsqueeze(0)

def get_embedding(model, img):
    with torch.no_grad():
        return model(preprocess_image(img)).squeeze().numpy()

# ---------------------------------
# Load Precomputed Embeddings
# ---------------------------------
@st.cache_resource
def load_precomputed_embeddings():
    embeddings = np.load("embeddings.npy")
    names = np.load("names.npy")
    return embeddings, names

# ---------------------------------
# Find Similar Images (FAST)
# ---------------------------------
def find_similar_images(upload_img, embeddings, names, model, top_n=5):
    upload_emb = get_embedding(model, upload_img)

    sims = cosine_similarity(
        upload_emb.reshape(1, -1),
        embeddings
    )[0]

    top_idx = np.argsort(sims)[::-1][:top_n]
    return [os.path.join("Data", names[i]) for i in top_idx]

# ---------------------------------
# Main Logic
# ---------------------------------
if st.button("✨ Generate Styling Advice + Similar Looks"):
    has_image = uploaded_file is not None
    has_text = len(user_query.strip()) > 0

    if not has_image and not has_text:
        st.warning("⚠️ Upload an image or enter a question!")
        st.stop()

    model = load_model()
    embeddings, names = load_precomputed_embeddings()

    similar_images = []

    # ------------------------
    # IMAGE PROCESSING
    # ------------------------
    if has_image:
        uploaded_img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_img, caption="Uploaded Outfit", use_container_width=True)

        with st.spinner("Finding similar looks..."):
            similar_images = find_similar_images(uploaded_img, embeddings, names, model)

        if similar_images:
            st.subheader("🖼️ Similar Looks")
            cols = st.columns(len(similar_images))
            for i, img_path in enumerate(similar_images):
                with cols[i]:
                    st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

    # ------------------------
    # AI STYLING ADVICE
    # ------------------------
    if client:
        prompt = "You are a professional stylist. "

        if has_text:
            prompt += f"The user asks: {user_query}. "

        if has_image:
            prompt += "Analyze the uploaded outfit and give styling suggestions. "

        if similar_images:
            prompt += f"Similar outfits found: {', '.join([os.path.basename(x) for x in similar_images])}. "

        try:
            with st.spinner("🪄 Generating Fashion Advice..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Be creative, helpful, and concise."},
                        {"role": "user", "content": prompt}
                    ],
                )

            st.subheader("💡 Styling Suggestions")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"❌ Groq API error: {e}")
