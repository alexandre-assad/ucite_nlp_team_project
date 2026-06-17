"""PyTorch MLP classifier wrapper for M2."""

from pathlib import Path

import torch
import torch.nn as nn


class TransitionMLP(nn.Module):
    """Simple one-hidden-layer MLP for transition classification."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.network(features)


class DenseMLPTransitionClassifier:
    """Model wrapper compatible with the existing greedy decoder API."""

    def __init__(
        self,
        transition_classes,
        input_dim,
        hidden_dim=512,
        dropout=0.2,
        device=None,
    ):
        self.transition_classes = list(transition_classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.transition_classes)}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.module = TransitionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=len(self.transition_classes),
            dropout=dropout,
        ).to(self.device)

    def logits(self, features_batch):
        return self.module(features_batch)

    def predict(self, features, valid_classes=None):
        self.module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.module(tensor)[0]

            if valid_classes is not None:
                mask = torch.full_like(logits, float("-inf"))
                valid_indices = [self.class_to_idx[label] for label in valid_classes]
                mask[valid_indices] = 0.0
                logits = logits + mask

            best_idx = int(torch.argmax(logits).item())
            return self.transition_classes[best_idx], float(logits[best_idx].item())

    def save(self, filepath, extra_state=None):
        target_path = Path(filepath)
        payload = {
            "state_dict": self.module.state_dict(),
            "transition_classes": self.transition_classes,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "extra_state": extra_state or {},
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target_path)
        print(f"Model saved to {_display_path(target_path)}")

    @classmethod
    def load(cls, filepath, device=None):
        source_path = Path(filepath)
        payload = torch.load(source_path, map_location=device or "cpu", weights_only=False)
        model = cls(
            transition_classes=payload["transition_classes"],
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
            device=device,
        )
        model.module.load_state_dict(payload["state_dict"])
        model.module.to(model.device)
        model.module.eval()
        print(f"Model loaded from {_display_path(source_path)}")
        return model

    def train_mode(self):
        self.module.train()

    def eval_mode(self):
        self.module.eval()


def _display_path(path):
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)
