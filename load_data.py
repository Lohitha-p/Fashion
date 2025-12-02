# load_data.py
from datasets import load_dataset
import os

def save_images(dataset_name="detection-datasets/fashionpedia",
                split="train",
                dataset_folder="Data",
                num_images=500):
    """
    Download Fashionpedia dataset and save images locally.
    """
    dataset = load_dataset(dataset_name, split=split)
    total = len(dataset)
    os.makedirs(dataset_folder, exist_ok=True)

    limit = min(num_images, total)
    print(f"Split '{split}' has {total} images. Saving {limit}...")

    for i in range(limit):
        image = dataset[i]['image']  # Hugging Face gives PIL.Image
        out_path = os.path.join(dataset_folder, f"{split}_image_{i+1}.png")
        image.save(out_path)

    print(f"✅ Saved {limit} images to '{dataset_folder}'.")
    return dataset_folder

if __name__ == "__main__":
    save_images(num_images=500, split="train")

