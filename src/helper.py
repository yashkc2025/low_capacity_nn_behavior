from typing import Literal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# --- Define valid digit pairs for binary classification ---
ValidPairKey = Literal["0_and_1", "1_and_7", "4_and_9", "3_and_8", "5_and_6"]

pairs: dict[ValidPairKey, list[int]] = {
    "0_and_1": [0, 1],
    "1_and_7": [1, 7],
    "4_and_9": [4, 9],
    "3_and_8": [3, 8],
    "5_and_6": [5, 6],
}


# --- Data Loader using pre-defined label pairs ---
def load_data(p: ValidPairKey, seed=None):
    """
    Load balanced and shuffled train/test split for a given digit pair.
    """
    return split_by_labels(pairs[p], seed=seed)


# --- Activation Functions and Derivatives ---
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --- Binary Cross-Entropy Loss ---
def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.
    Adds epsilon to avoid log(0).
    """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# --- Evaluation Metrics: Precision, Recall, F1, AUC ---
def evaluate_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute precision, recall, F1, and ROC AUC from predicted probabilities.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_probs = np.asarray(y_pred_probs).flatten()
    y_pred = (y_pred_probs >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC AUC": auc,
    }


# --- Label Remapping for Binary Classification (0 or 1) ---
def remap_labels(y):
    """
    Convert labels to binary: lower label → 0, higher label → 1.
    Useful when original labels are e.g., [3, 8].
    """
    labels = np.unique(y)
    assert len(labels) == 2, f"Expected binary classification but got labels: {labels}"
    return (y == labels[1]).astype(int)


# --- Main Data Splitting Function ---
def split_by_labels(
    label_list,
    split_dir="../data",
    test_ratio=0.2,
    shuffle=True,
    balance=True,
    seed=None,
):
    """
    Load and filter image/label data by selected digits,
    optionally balances class samples, shuffles, and splits into train/test.

    Args:
        label_list (list[int]): List of two digit labels to include.
        split_dir (str): Directory containing 'images.npy' and 'labels.npy'.
        test_ratio (float): Test split ratio.
        shuffle (bool): Whether to shuffle the data.
        balance (bool): Balance the dataset by downsampling to the smallest class.
        seed (int): Seed for deterministic split/shuffle.

    Returns:
        (x_train, y_train, x_test, y_test): Tuple of split data arrays.
    """
    # --- Set RNG with optional seed ---
    rng = np.random.default_rng(seed)

    # --- Load raw dataset ---
    vectors = np.load(f"{split_dir}/images.npy")  # shape (N, D)
    labels = np.load(f"{split_dir}/labels.npy")  # shape (N,)

    # --- Filter for relevant classes ---
    mask = np.isin(labels, label_list)
    vectors = vectors[mask]
    labels = labels[mask]

    # --- Optional: Balance class distribution ---
    if balance:
        min_count = min([(labels == l).sum() for l in label_list])
        balanced_vectors = []
        balanced_labels = []
        for l in label_list:
            idx = np.where(labels == l)[0]
            selected_idx = rng.choice(idx, size=min_count, replace=False)
            balanced_vectors.append(vectors[selected_idx])
            balanced_labels.append(labels[selected_idx])
        vectors = np.concatenate(balanced_vectors)
        labels = np.concatenate(balanced_labels)

    # --- Optional: Shuffle dataset ---
    if shuffle:
        indices = rng.permutation(len(vectors))
        vectors = vectors[indices]
        labels = labels[indices]

    # --- Train/Test split ---
    x_train, x_test, y_train, y_test = train_test_split(
        vectors,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    return x_train, y_train, x_test, y_test
