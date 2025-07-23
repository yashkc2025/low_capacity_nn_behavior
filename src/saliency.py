import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import BinaryClassifier
from helper import load_data


def remap_labels(y):
    """
    Explicitly remaps original digit labels:
        - 4 → 0 (negative class)
        - 9 → 1 (positive class)
    """
    y = np.array(y)
    label_set = set(np.unique(y))
    assert label_set == {4, 9}, f"Unexpected labels found: {label_set}"
    return np.where(y == 4, 0, 1)


def create_saliency_map(model, image_vector, epsilon=0.1):
    """
    Creates a saliency map by computing change in model output
    probability as each pixel is perturbed independently.

    Args:
        model: Trained BinaryClassifier.
        image_vector: Flattened image (784,)
        epsilon: Small perturbation added to each pixel.

    Returns:
        A 28x28 saliency heatmap.
    """
    baseline_score = model.predict_proba(image_vector.reshape(1, -1))[0]
    saliency_values = np.zeros_like(image_vector)

    for i in range(image_vector.shape[0]):
        perturbed_image = np.copy(image_vector)
        perturbed_image[i] += epsilon
        new_score = model.predict_proba(perturbed_image.reshape(1, -1))[0]
        saliency_values[i] = np.abs(new_score - baseline_score)

    # Normalize to [0, 1]
    saliency_values -= saliency_values.min()
    if saliency_values.max() != 0:
        saliency_values /= saliency_values.max()

    return saliency_values.reshape(28, 28)


def find_example_images(model: BinaryClassifier, x_val, y_val):
    """
    Finds one correctly classified and one misclassified image.
    Assumes remapped labels:
        - 0 → class 4
        - 1 → class 9

    Model decision rule:
        - prob > 0.5 → class 9
        - prob < 0.5 → class 4
    """
    y_val = remap_labels(y_val)
    assert set(np.unique(y_val)).issubset({0, 1}), "Labels must be 0 and 1 only."

    predictions_prob = model.predict_proba(x_val)
    predicted_labels = (predictions_prob > 0.5).astype(int)

    correct_indices = np.where(predicted_labels == y_val)[0]
    misclassified_indices = np.where(predicted_labels != y_val)[0]

    correct_image = x_val[correct_indices[0]] if len(correct_indices) > 0 else None
    misclassified_image = (
        x_val[misclassified_indices[0]] if len(misclassified_indices) > 0 else None
    )

    if misclassified_image is None:
        print("✅ All predictions correct — no misclassified sample found.")

    return correct_image, misclassified_image


# --- Experiment Parameters ---
task = "4_and_9"
size = 48
seed = 42

# --- Load Models ---
original_model = BinaryClassifier.load_model(
    f"./models/training/{task}_lr0.03_hs{size}_seed{seed}.pkl"
)

pruned_model = copy.deepcopy(original_model)
pruned_model.prune_prob = 0.95
pruned_model._apply_magnitude_pruning()

# --- Load Data ---
_, _, x_val, y_val = load_data(task, seed)
correct_image, misclassified_image = find_example_images(original_model, x_val, y_val)

# --- Saliency Map Computation ---
saliency_correct = create_saliency_map(original_model, correct_image)

# Handle case with no misclassified sample
if misclassified_image is not None:
    saliency_misclassified = create_saliency_map(original_model, misclassified_image)
else:
    saliency_misclassified = np.zeros((28, 28))  # dummy heatmap

saliency_pruned = create_saliency_map(pruned_model, correct_image)

# --- Visualization ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# A) Original Correct Image
axes[0, 0].imshow(correct_image.reshape(28, 28), cmap="gray")
axes[0, 0].set_title("A) Original Correctly Classified Image")

# B) Saliency - Original Model
sns.heatmap(saliency_correct, ax=axes[0, 1], cmap="hot", cbar=False)
axes[0, 1].set_title("B) Saliency Map (Original Model)")

# C) Saliency - Misclassified Image (or blank)
sns.heatmap(saliency_misclassified, ax=axes[1, 0], cmap="hot", cbar=False)
axes[1, 0].set_title(
    "C) Saliency Map (Misclassified Image)"
    if misclassified_image is not None
    else "C) No Misclassified Image"
)

# D) Saliency - Pruned Model
sns.heatmap(saliency_pruned, ax=axes[1, 1], cmap="hot", cbar=False)
axes[1, 1].set_title("D) Saliency Map (95% Pruned Model)")

# Final layout
fig.suptitle(f"Saliency Map Analysis: {task} (Hidden Size = {size})", fontsize=16)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"./results/saliency_{task}_hs{size}_seed{seed}.png", dpi=300)
plt.show()
