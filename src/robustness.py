from sklearn.metrics import f1_score
import numpy as np
import os
import json
from helper import load_data, remap_labels
from model import BinaryClassifier

# -----------------------------
# Gaussian Noise Injection
# -----------------------------


def get_gaussian_set(x_val):
    """
    Adds Gaussian noise with different standard deviations (sigmas) to input data.
    Returns a list of noisy versions of x_val.
    """
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    x_val_noisy = []

    for sigma in sigmas:
        noise = np.random.normal(0, sigma, x_val.shape)
        x_val_noisy_elem = np.clip(x_val + noise, 0, 1)  # keep input valid
        x_val_noisy.append(x_val_noisy_elem)

    return x_val_noisy


pair = "4_and_9"
sizes = [24, 48, 64]
seeds = [0, 42, 78]
results_gaussian = []
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]

for seed in seeds:
    _, _, x_val, y_val = load_data(pair, seed)
    y_val = remap_labels(y_val)

    x_val_noisy = get_gaussian_set(x_val)
    seed_result = {"seed": seed}

    for size in sizes:
        size_score = {"size": size}

        # Load trained model
        model = BinaryClassifier.load_model(
            f"./models/training/4_and_9_lr0.03_hs{size}_seed{seed}.pkl"
        )

        # Evaluate on each noise level
        for noise, sigma in zip(x_val_noisy, sigmas):
            y_pred = model.predict(noise)
            score = f1_score(y_val, y_pred)
            size_score[str(sigma)] = score

        seed_result[str(size)] = size_score

    results_gaussian.append(seed_result)


# -----------------------------
# Random Occlusion
# -----------------------------


def apply_occlusion(images, patch_size=7):
    """
    Applies a random occlusion patch (black square) to a batch of images.
    """
    occluded_images = np.copy(images)
    h, w = images.shape[1], images.shape[2]

    for i in range(len(occluded_images)):
        x_start = np.random.randint(0, w - patch_size + 1)
        y_start = np.random.randint(0, h - patch_size + 1)
        occluded_images[
            i, y_start : y_start + patch_size, x_start : x_start + patch_size
        ] = 0

    return occluded_images


results_occlusion = []

for seed in seeds:
    _, _, x_val, y_val = load_data(pair, seed)
    y_val = remap_labels(y_val)

    # Reshape to 28x28, apply occlusion, flatten back
    x_val_reshaped = x_val.reshape(-1, 28, 28)
    x_val_occluded_reshaped = apply_occlusion(x_val_reshaped)
    x_val_occluded = x_val_occluded_reshaped.reshape(-1, 784)

    seed_result = {"seed": seed}

    for size in sizes:
        model = BinaryClassifier.load_model(
            f"./models/training/4_and_9_lr0.03_hs{size}_seed{seed}.pkl"
        )
        y_pred = model.predict(x_val_occluded)
        score = f1_score(y_val, y_pred)
        seed_result[str(size)] = score

    results_occlusion.append(seed_result)


# -----------------------------
# Serialize Results
# -----------------------------

os.makedirs("./results", exist_ok=True)

with open("./results/gaussian_noise_robustness.json", "w") as f:
    json.dump(results_gaussian, f, indent=2)

with open("./results/occlusion_robustness.json", "w") as f:
    json.dump(results_occlusion, f, indent=2)

print("✅ Saved robustness results to ./results/")
