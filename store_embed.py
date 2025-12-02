
import os
import numpy as np
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Init ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="fashion_images",
    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

def embed_images(folder="Data"):
    images = [f for f in os.listdir(folder) if f.endswith(".png")]
    for idx, img_file in enumerate(images):
        img_path = os.path.join(folder, img_file)
        collection.add(
            ids=[f"img_{idx}"],
            documents=[img_file],
            metadatas=[{"path": img_path}]
        )
    print(f"✅ Stored {len(images)} image embeddings in ChromaDB.")

if __name__ == "__main__":
    embed_images()
