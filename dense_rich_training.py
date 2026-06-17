"""Training helpers for M2b rich dense features."""

import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from evaluator import evaluate
from parser import parse_all
from transitions import make_left_arc, make_right_arc

PROJECT_ROOT = Path(__file__).resolve().parent


def build_transition_inventory(labels):
    transitions = ["SHIFT"]
    for label in labels:
        transitions.append(make_left_arc(label))
    for label in labels:
        transitions.append(make_right_arc(label))
    return transitions


def generate_rich_arc_standard_examples(sentences, transition_system, oracle, feature_extractor):
    examples = []
    transition_counts = Counter()
    for sentence_id, sentence in enumerate(sentences):
        config = transition_system.initial_config(sentence)
        while not transition_system.is_terminal(config):
            gold_transition = oracle.choose(config)
            examples.append(
                {
                    "sentence_id": sentence_id,
                    "snapshot": feature_extractor.extract_snapshot(config),
                    "gold_transition": gold_transition,
                }
            )
            transition_counts[_transition_family(gold_transition)] += 1
            transition_system.apply(config, gold_transition)
    return examples, transition_counts


def train_dense_rich_model(
    train_sentences,
    dev_sentences,
    model,
    labels,
    feature_extractor,
    transition_system,
    oracle,
    batch_size,
    n_epochs,
    seed,
    learning_rate,
    weight_decay,
    checkpoint_dir,
):
    transition_inventory = build_transition_inventory(labels)
    train_examples, transition_counts = generate_rich_arc_standard_examples(
        train_sentences,
        transition_system,
        oracle,
        feature_extractor,
    )
    label_to_idx = {label: idx for idx, label in enumerate(transition_inventory)}

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = random.Random(seed)

    training_summary = {
        "seed": seed,
        "epochs": [],
        "oracle_transition_counts": dict(transition_counts),
        "generated_transition_examples": len(train_examples),
        "best_dev_checkpoint": None,
        "training_time_seconds": 0.0,
    }

    best_dev_las = float("-inf")
    best_dev_uas = float("-inf")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        shuffled_examples = list(train_examples)
        rng.shuffle(shuffled_examples)
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        model.train_mode()

        for start_idx in range(0, len(shuffled_examples), batch_size):
            batch = shuffled_examples[start_idx : start_idx + batch_size]
            batch_features = [
                feature_extractor.materialize_from_snapshot(
                    train_sentences[item["sentence_id"]],
                    item["snapshot"],
                )
                for item in batch
            ]
            targets = torch.as_tensor(
                [label_to_idx[item["gold_transition"]] for item in batch],
                dtype=torch.long,
                device=model.device,
            )

            optimizer.zero_grad()
            logits = model.logits(model.prepare_batch(batch_features))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch)
            correct_predictions += int((logits.argmax(dim=1) == targets).sum().item())
            total_predictions += len(batch)

            if (start_idx // batch_size + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch}: processed "
                    f"{min(start_idx + len(batch), len(shuffled_examples))}/{len(shuffled_examples)} examples",
                    end="\r",
                )

        elapsed = time.time() - start_time
        train_loss = total_loss / max(total_predictions, 1)
        train_accuracy = 100.0 * correct_predictions / max(total_predictions, 1)

        model.eval_mode()
        dev_predictions = parse_all(
            dev_sentences,
            model,
            labels,
            transition_system=transition_system,
            feature_extractor=feature_extractor,
        )
        dev_uas, dev_las = evaluate(dev_sentences, dev_predictions)

        print(
            f"  Epoch {epoch}: train loss = {train_loss:.4f}, "
            f"train accuracy = {train_accuracy:.2f}% [{elapsed:.1f}s]"
        )
        print(f"          dev UAS = {dev_uas:.2f}%, LAS = {dev_las:.2f}%")

        is_better_checkpoint = (
            dev_las > best_dev_las
            or (dev_las == best_dev_las and dev_uas > best_dev_uas)
        )
        if is_better_checkpoint:
            best_dev_las = dev_las
            best_dev_uas = dev_uas
            checkpoint_info = _save_dense_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                dev_uas=dev_uas,
                dev_las=dev_las,
            )
            training_summary["best_dev_checkpoint"] = checkpoint_info

        training_summary["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
                "elapsed_seconds": elapsed,
                "dev_uas": dev_uas,
                "dev_las": dev_las,
            }
        )

    training_summary["training_time_seconds"] = sum(
        epoch_info["elapsed_seconds"] for epoch_info in training_summary["epochs"]
    )
    return training_summary


def _save_dense_checkpoint(checkpoint_dir, epoch, model, dev_uas, dev_las):
    model_path = Path(checkpoint_dir) / "best_model.pt"
    metadata_path = Path(checkpoint_dir) / "best_checkpoint.json"
    model.save(str(model_path), extra_state={"epoch": epoch, "dev_uas": dev_uas, "dev_las": dev_las})
    metadata = {
        "epoch": epoch,
        "dev_uas": dev_uas,
        "dev_las": dev_las,
        "model_path": _relative_path(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _transition_family(transition):
    if transition.startswith("LEFT-ARC"):
        return "LEFT-ARC"
    if transition.startswith("RIGHT-ARC"):
        return "RIGHT-ARC"
    return transition


def _relative_path(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
