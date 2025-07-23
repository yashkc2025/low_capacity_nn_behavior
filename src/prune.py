from helper import pairs, remap_labels, load_data, evaluate_metrics
from model import BinaryClassifier
import copy
import numpy as np
import os
import json


lr = 0.03
hidden_sizes = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
prune_sizes = [10, 15, 20, 30, 40, 50, 80, 90, 95, 99]
fine_tune_percentages = [1, 5, 10, 15, 20, 25]
seeds = [0, 42, 78]


def magnitude_prune():
    metrics = []
    for pair in pairs:
        pair_metric = {"pair": pair}

        for size in hidden_sizes:
            size_metric = {"size": size}

            for seed in seeds:
                seed_metrics = {"seed": seed}
                model_og = BinaryClassifier.load_model(
                    f"./models/training/{pair}_lr{0.03}_hs{size}_seed{seed}.pkl"
                )

                # Load test set
                _, _, X_test, y_test = load_data(pair, seed)
                y_test = remap_labels(y_test)

                for p_size in prune_sizes:
                    model = copy.deepcopy(model_og)
                    model.prune_prob = p_size / 100.0  # FIXED
                    model._apply_magnitude_pruning()
                    print(f"Pruned {p_size}% | Sparsity: {model.sparsity()}")

                    probs = model.predict_proba(X_test)

                    metric = evaluate_metrics(y_test, probs)

                    seed_metrics[f"pruned_{p_size}%"] = {
                        "F1": metric["F1"],
                        "AUC": metric["ROC AUC"],
                    }

                # Original performance (before pruning)
                probs_og = model_og.predict_proba(X_test)
                metric_og = evaluate_metrics(y_test, probs_og)

                seed_metrics["original"] = {
                    "F1": metric_og["F1"],
                    "AUC": metric_og["ROC AUC"],
                }

                size_metric[f"seed_{seed}"] = seed_metrics

            pair_metric[f"hidden_{size}"] = size_metric

        metrics.append(pair_metric)
    return metrics


magnitude_pruning_metrics = magnitude_prune()


def count_dead_neurons(model: BinaryClassifier, x_validation):
    hidden_activations = model.get_hidden_activations(x_validation)
    is_dead = np.all(hidden_activations == 0, 0)
    return np.sum(is_dead)


dn_pairs = ["0_and_1", "4_and_9"]
dn_size = [32, 24]
dn_level = 99
dn_results = []

for p, s in zip(dn_pairs, dn_size):
    seeds_res = []
    for seed in seeds:
        _, _, x_val, _ = load_data(p, seed)
        model = BinaryClassifier.load_model(
            f"./models/training/{p}_lr0.03_hs{s}_seed{seed}.pkl"
        )
        dead_og = count_dead_neurons(model, x_val)

        pruned_model = copy.deepcopy(model)
        pruned_model.prune_prob = dn_level / 100.0
        pruned_model._apply_magnitude_pruning()

        dead_after = count_dead_neurons(pruned_model, x_val)

        seeds_res.append({"seed": seed, "og": dead_og, "prune": dead_after})
    dn_results.append({"pair": p, "size": s, "results": seeds_res})

# Ensure the results directory exists
os.makedirs("./results", exist_ok=True)

# Save Magnitude Pruning Results
with open("./results/magnitude_pruning_metrics.json", "w") as f:
    json.dump(magnitude_pruning_metrics, f, indent=2)

# Save Dead Neuron Analysis Results
with open("./results/dead_neuron_analysis.json", "w") as f:
    json.dump(dn_results, f, indent=2)

print("Saved magnitude pruning results to ./results/magnitude_pruning_metrics.json")
print("Saved dead neuron analysis results to ./results/dead_neuron_analysis.json")
