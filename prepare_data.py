import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import os


def image_to_vector(image, size=(28, 28)):
    # MNIST images are already 28x28 grayscale, but convert to numpy just in case
    arr = np.array(image, dtype=np.float32)
    # Normalize pixel values [0, 255] to [0, 1]
    vector = arr.flatten() / 255.0
    return vector


def save_vectors_labels(vectors, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "images.npy"), vectors)
    np.save(os.path.join(save_dir, "labels.npy"), labels)
    print(f"Saved {len(vectors)} vectors and labels to '{save_dir}'")


def process_and_save_mnist(save_dir="./data"):
    os.makedirs(save_dir, exist_ok=True)

    ds = load_dataset("ylecun/mnist")

    # Process train and test splits separately
    for split in ["train", "test"]:
        vectors = []
        labels = []
        print(f"Processing {split} split:")
        for item in tqdm(ds[split], desc=f"{split} images"):
            img = item["image"]
            label = item["label"]
            vec = image_to_vector(img)
            vectors.append(vec)
            labels.append(label)

        vectors = np.array(vectors, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        split_save_dir = os.path.join(save_dir, split)
        os.makedirs(split_save_dir, exist_ok=True)

        save_vectors_labels(vectors, labels, split_save_dir)

process_and_save_mnist()
