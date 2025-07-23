import numpy as np
from helper import (
    relu,
    relu_derivative,
    sigmoid,
    binary_cross_entropy,
    evaluate_metrics,
)
import pickle


# A simple 2-layer fully connected neural network for binary classification.
class BinaryClassifier:
    def __init__(self, hidden_size: int, n_features: int, prune_prob: float = 0.0):
        # Initialize weights and biases with small random values and zeros
        self.w_1 = np.random.randn(n_features, hidden_size) * 0.1  # Input to hidden
        self.b_1 = np.zeros((1, hidden_size))
        self.w_2 = np.random.randn(hidden_size, 1) * 0.1  # Hidden to output
        self.b_2 = np.zeros((1,))

        self.prune_prob = prune_prob  # Probability for pruning
        self.history = {  # Metrics tracked during training
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

    def forward(self, X):
        # Forward pass: compute intermediate and final activations
        z_1 = X @ self.w_1 + self.b_1  # Linear transformation for hidden layer
        a_1 = relu(z_1)  # Activation: ReLU
        z_2 = a_1 @ self.w_2 + self.b_2  # Linear transformation for output
        a_2 = sigmoid(z_2)  # Activation: Sigmoid for binary output
        return z_1, a_1, z_2, a_2

    def _apply_random_pruning(self):
        # Randomly zero out a percentage of weights (unstructured pruning)
        if self.prune_prob <= 0.0:
            return
        w1_mask = np.random.rand(*self.w_1.shape) >= self.prune_prob
        w2_mask = np.random.rand(*self.w_2.shape) >= self.prune_prob
        self.w_1 *= w1_mask
        self.w_2 *= w2_mask

    def _apply_magnitude_pruning(self):
        # Prune weights with smallest absolute values (more meaningful than random)
        if self.prune_prob <= 0.0:
            return

        # Prune w_1
        flat_w1 = self.w_1.flatten()
        k1 = int(self.prune_prob / 100 * flat_w1.size)  # number of weights to prune
        if k1 > 0:
            threshold1 = np.partition(np.abs(flat_w1), k1 - 1)[k1 - 1]
            self.w_1[np.abs(self.w_1) <= threshold1] = 0

        # Prune w_2
        flat_w2 = self.w_2.flatten()
        k2 = int(self.prune_prob / 100 * flat_w2.size)
        if k2 > 0:
            threshold2 = np.partition(np.abs(flat_w2), k2 - 1)[k2 - 1]
            self.w_2[np.abs(self.w_2) <= threshold2] = 0

    def backward(self, X, y, z_1, a_1, z_2, a_2, lr):
        # Backpropagation: compute gradients and update weights/biases
        m = X.shape[0]

        # Output layer gradients
        delta_2 = a_2 - y.reshape(-1, 1)
        d_w2 = a_1.T @ delta_2 / m
        d_b2 = np.mean(delta_2, axis=0)

        # Hidden layer gradients
        delta_1 = (delta_2 @ self.w_2.T) * relu_derivative(z_1)
        d_w1 = X.T @ delta_1 / m
        d_b1 = np.mean(delta_1, axis=0, keepdims=True)

        # Update parameters
        self.w_2 -= lr * d_w2
        self.b_2 -= lr * d_b2
        self.w_1 -= lr * d_w1
        self.b_1 -= lr * d_b1

    def train(self, X, y, epochs=100, lr=0.01):
        # Training loop
        for epoch in range(epochs):
            z_1, a_1, z_2, a_2 = self.forward(X)
            loss = binary_cross_entropy(y.reshape(-1, 1), a_2)
            preds = (a_2 >= 0.5).astype(int).flatten()
            acc = np.mean(preds == y)

            # Compute other metrics (Precision, Recall, F1, AUC)
            metrics = evaluate_metrics(y, a_2)

            # Log history
            self.history["loss"].append(loss)
            self.history["accuracy"].append(acc)
            self.history["precision"].append(metrics["Precision"])
            self.history["recall"].append(metrics["Recall"])
            self.history["f1"].append(metrics["F1"])
            self.history["auc"].append(metrics["ROC AUC"])

            # Print progress every 10 epochs
            if not epoch % 10:
                print(
                    f"Epoch {epoch+1:3} | Loss: {loss:.4f} | Acc: {acc:.2f} "
                    f"| F1: {metrics['F1']:.2f} | AUC: {metrics['ROC AUC']:.2f}"
                )

            # Update weights
            self.backward(X, y, z_1, a_1, z_2, a_2, lr)

    def predict(self, X):
        # Predict class labels
        _, _, _, a_2 = self.forward(X)
        return (a_2 >= 0.5).astype(int).flatten()

    def predict_proba(self, X):
        # Predict probabilities instead of class labels
        _, _, _, a_2 = self.forward(X)
        return a_2.flatten()

    def get_training_score(self):
        # Return training history
        return self.history.copy()

    def save(self, filepath: str):
        # Save model parameters and history to file
        data = {
            "w_1": self.w_1,
            "b_1": self.b_1,
            "w_2": self.w_2,
            "b_2": self.b_2,
            "history": self.history,
            "prune_prob": self.prune_prob,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_model(cls, filepath: str):
        # Load model from file
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(
            hidden_size=data["w_1"].shape[1],
            n_features=data["w_1"].shape[0],
            prune_prob=data.get("prune_prob", 0.0),
        )
        # Restore parameters
        model.w_1 = data["w_1"]
        model.b_1 = data["b_1"]
        model.w_2 = data["w_2"]
        model.b_2 = data["b_2"]
        model.history = data["history"]
        return model

    def sparsity(self):
        # Compute and return % of zero weights in each layer
        w1_sparsity = 100.0 * np.sum(self.w_1 == 0) / self.w_1.size
        w2_sparsity = 100.0 * np.sum(self.w_2 == 0) / self.w_2.size
        return {"w1": w1_sparsity, "w2": w2_sparsity}

    def get_hidden_activations(self, X):
        # Return hidden layer activations for input X
        _, a1, _, _ = self.forward(X)
        return a1
