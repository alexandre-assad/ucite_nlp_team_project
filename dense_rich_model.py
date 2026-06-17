"""Rich dense MLP transition classifier for M2b."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RichTransitionMLP(nn.Module):
    """MLP over static word vectors, learned POS/label embeddings, and structure."""

    def __init__(
        self,
        word_embedding_dim,
        pos_vocab_size,
        label_vocab_size,
        pos_embedding_dim,
        label_embedding_dim,
        hidden_dim,
        dropout,
        token_position_count=10,
        child_label_count=4,
        bucket_size=6,
    ):
        super().__init__()
        self.token_position_count = token_position_count
        self.child_label_count = child_label_count
        self.bucket_size = bucket_size
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        self.label_embedding = nn.Embedding(label_vocab_size, label_embedding_dim)
        self.input_dim = (
            token_position_count * word_embedding_dim
            + token_position_count * pos_embedding_dim
            + child_label_count * label_embedding_dim
            + (3 * bucket_size)
            + (2 * token_position_count)
        )
        self.hidden_layer = nn.Linear(self.input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, features):
        batch_size = features["word_vectors"].shape[0]
        word_part = features["word_vectors"].reshape(batch_size, -1)
        pos_part = self.pos_embedding(features["pos_ids"]).reshape(batch_size, -1)
        label_part = self.label_embedding(features["label_ids"]).reshape(batch_size, -1)
        distance_part = F.one_hot(features["distance_bucket"], num_classes=self.bucket_size).float()
        stack_part = F.one_hot(features["stack_size_bucket"], num_classes=self.bucket_size).float()
        buffer_part = F.one_hot(features["buffer_size_bucket"], num_classes=self.bucket_size).float()
        flag_part = torch.cat([features["null_flags"], features["root_flags"]], dim=1)
        return torch.cat(
            [word_part, pos_part, label_part, distance_part, stack_part, buffer_part, flag_part],
            dim=1,
        )


class DenseRichTransitionClassifier:
    """Prediction wrapper compatible with the greedy decoder."""

    def __init__(
        self,
        transition_classes,
        word_embedding_dim,
        pos_vocab_size,
        label_vocab_size,
        pos_embedding_dim=32,
        label_embedding_dim=32,
        hidden_dim=512,
        dropout=0.2,
        device=None,
    ):
        self.transition_classes = list(transition_classes)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.transition_classes)}
        self.word_embedding_dim = word_embedding_dim
        self.pos_vocab_size = pos_vocab_size
        self.label_vocab_size = label_vocab_size
        self.pos_embedding_dim = pos_embedding_dim
        self.label_embedding_dim = label_embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_net = RichTransitionMLP(
            word_embedding_dim=word_embedding_dim,
            pos_vocab_size=pos_vocab_size,
            label_vocab_size=label_vocab_size,
            pos_embedding_dim=pos_embedding_dim,
            label_embedding_dim=label_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.feature_dim = self.feature_net.input_dim
        self.hidden_layer = self.feature_net.hidden_layer
        self.activation = self.feature_net.activation
        self.dropout_layer = self.feature_net.dropout
        self.output_layer = nn.Linear(hidden_dim, len(self.transition_classes))
        self.feature_net.to(self.device)
        self.output_layer.to(self.device)

    def logits(self, features_batch):
        hidden_input = self.feature_net.forward_features(features_batch).to(self.device)
        hidden = self.hidden_layer(hidden_input)
        hidden = self.activation(hidden)
        hidden = self.dropout_layer(hidden)
        return self.output_layer(hidden)

    def predict(self, features, valid_classes=None):
        self.eval_mode()
        with torch.no_grad():
            batch = self.prepare_batch([features])
            logits = self.logits(batch)[0]
            if valid_classes is not None:
                mask = torch.full_like(logits, float("-inf"))
                valid_indices = [self.class_to_idx[label] for label in valid_classes]
                mask[valid_indices] = 0.0
                logits = logits + mask
            best_idx = int(torch.argmax(logits).item())
            return self.transition_classes[best_idx], float(logits[best_idx].item())

    def prepare_batch(self, feature_dicts):
        return {
            "word_vectors": torch.as_tensor(
                np.stack([item["word_vectors"] for item in feature_dicts]),
                dtype=torch.float32,
                device=self.device,
            ),
            "pos_ids": torch.as_tensor(
                np.stack([item["pos_ids"] for item in feature_dicts]),
                dtype=torch.long,
                device=self.device,
            ),
            "label_ids": torch.as_tensor(
                np.stack([item["label_ids"] for item in feature_dicts]),
                dtype=torch.long,
                device=self.device,
            ),
            "distance_bucket": torch.as_tensor(
                [item["distance_bucket"] for item in feature_dicts],
                dtype=torch.long,
                device=self.device,
            ),
            "stack_size_bucket": torch.as_tensor(
                [item["stack_size_bucket"] for item in feature_dicts],
                dtype=torch.long,
                device=self.device,
            ),
            "buffer_size_bucket": torch.as_tensor(
                [item["buffer_size_bucket"] for item in feature_dicts],
                dtype=torch.long,
                device=self.device,
            ),
            "null_flags": torch.as_tensor(
                np.stack([item["null_flags"] for item in feature_dicts]),
                dtype=torch.float32,
                device=self.device,
            ),
            "root_flags": torch.as_tensor(
                np.stack([item["root_flags"] for item in feature_dicts]),
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def save(self, filepath, extra_state=None):
        target_path = Path(filepath)
        payload = {
            "feature_net": self.feature_net.state_dict(),
            "output_layer": self.output_layer.state_dict(),
            "transition_classes": self.transition_classes,
            "word_embedding_dim": self.word_embedding_dim,
            "pos_vocab_size": self.pos_vocab_size,
            "label_vocab_size": self.label_vocab_size,
            "pos_embedding_dim": self.pos_embedding_dim,
            "label_embedding_dim": self.label_embedding_dim,
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
            word_embedding_dim=payload["word_embedding_dim"],
            pos_vocab_size=payload["pos_vocab_size"],
            label_vocab_size=payload["label_vocab_size"],
            pos_embedding_dim=payload["pos_embedding_dim"],
            label_embedding_dim=payload["label_embedding_dim"],
            hidden_dim=payload["hidden_dim"],
            dropout=payload["dropout"],
            device=device,
        )
        model.feature_net.load_state_dict(payload["feature_net"])
        model.output_layer.load_state_dict(payload["output_layer"])
        model.feature_net.to(model.device)
        model.output_layer.to(model.device)
        model.eval_mode()
        print(f"Model loaded from {_display_path(source_path)}")
        return model

    def train_mode(self):
        self.feature_net.train()
        self.output_layer.train()

    def eval_mode(self):
        self.feature_net.eval()
        self.output_layer.eval()

    def parameters(self):
        return list(self.feature_net.parameters()) + list(self.output_layer.parameters())


def _display_path(path):
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)
