# --- Standard Libraries ---
import numpy as np
import json

# --- Custom Helper Functions ---
from helper import (
    load_data,
    pairs,
    remap_labels,
)
from model import BinaryClassifier

# --- Training Loop over Seeds, Hidden Sizes, and Binary Classification Pairs ---

training_results = []

seeds = [0, 42, 78]
hidden_sizes = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
lr = 3e-2

# Grid search over seeds and hidden sizes for all digit pairs
for seed in seeds:
    np.random.seed(seed)  # Fix random state for reproducibility

    for size in hidden_sizes:
        for combination in pairs:
            # --- Load and preprocess dataset ---
            x_train, y_train, x_test, y_test = load_data(combination, seed)
            y_train = remap_labels(y_train)
            y_test = remap_labels(y_test)

            # --- Initialize model ---
            model = BinaryClassifier(hidden_size=size, n_features=784)
            print(f"\nTraining on lr: {lr} with size: {size} and pair: {combination}")

            # --- Train model ---
            model.train(x_train, y_train, epochs=100, lr=lr)

            # --- Save trained model ---
            model.save(
                f"./models/training/{combination}_lr{lr}_hs{size}_seed{seed}.pkl"
            )

            # --- Collect training results ---
            score = model.get_training_score() | {
                "size": size,
                "pair": combination,
                "seed": seed,
            }
            training_results.append(score)

# --- Save training results to JSON ---
with open("./results/training.json", "w") as f:
    json.dump(training_results, f)
