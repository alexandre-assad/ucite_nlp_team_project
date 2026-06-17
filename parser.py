"""
parser.py — Training and Greedy Parsing

This module ties everything together:

  Training:
    For each sentence, we start from the initial configuration and
    repeatedly:
      1. Extract features from the current configuration
      2. Ask the oracle for the correct transition
      3. Ask the perceptron to predict a transition
      4. Update the perceptron if the prediction is wrong
      5. Apply the correct transition (oracle's choice) to move forward
    
    After all epochs, we average the weights.

  Greedy Parsing:
    Starting from the initial configuration, we:
      1. Extract features
      2. Get the list of valid transitions
      3. Use the perceptron to pick the best valid transition
      4. Apply it
    Until we reach a terminal configuration.
"""

import copy
import json
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from evaluator import evaluate
from features import SparseFeatureExtractor
from framework import Decoder
from oracle import StaticArcEagerOracle
from transitions import ArcEagerSystem


def train(
    train_sentences,
    dev_sentences,
    model,
    labels,
    n_epochs=10,
    seed=0,
    checkpoint_dir=None,
    transition_system=None,
    oracle=None,
    feature_extractor=None,
):
    """
    Train the perceptron model on the training data.
    
    Parameters:
        train_sentences: list of sentences (from read_conllu), with gold trees
        dev_sentences:   list of sentences for development evaluation
        model:           an AveragedPerceptron instance
        labels:          list of all dependency relation labels
        n_epochs:        number of training passes over the data
    """
    transition_system = transition_system or ArcEagerSystem()
    oracle = oracle or StaticArcEagerOracle()
    feature_extractor = feature_extractor or SparseFeatureExtractor()

    print(f"Training on {len(train_sentences)} sentences for {n_epochs} epochs")
    print(f"Using {len(labels)} labels: {labels[:5]}{'...' if len(labels) > 5 else ''}")
    print("-" * 60)

    shuffled_sentences = list(train_sentences)
    rng = random.Random(seed)
    training_summary = {
        "seed": seed,
        "epochs": [],
        "oracle_transition_counts": Counter(),
        "training_time_seconds": 0.0,
        "best_dev_checkpoint": None,
    }
    best_dev_las = float("-inf")
    best_dev_uas = float("-inf")

    for epoch in range(1, n_epochs + 1):
        # count correct and total predictions for this epoch
        n_correct = 0
        n_total = 0
        start_time = time.time()
        rng.shuffle(shuffled_sentences)
        epoch_transition_counts = Counter()

        for i, sentence in enumerate(shuffled_sentences):
            # start from the initial configuration
            config = transition_system.initial_config(sentence)

            # process until we reach a terminal configuration
            while not transition_system.is_terminal(config):
                # step 1: extract features
                feats = feature_extractor.extract(config)

                # step 2: ask the oracle for the gold transition
                gold_transition = oracle.choose(config)
                epoch_transition_counts[_transition_family(gold_transition)] += 1
                training_summary["oracle_transition_counts"][_transition_family(gold_transition)] += 1

                # step 3: ask the perceptron to predict
                valid_transitions = transition_system.valid_transitions(config, labels)
                predicted, _ = model.predict(feats, valid_transitions)

                # step 4: update weights if the prediction is wrong
                model.update(gold_transition, predicted, feats)

                # count for accuracy
                if predicted == gold_transition:
                    n_correct += 1
                n_total += 1

                # step 5: apply the GOLD transition to move forward
                # (during training, we always follow the oracle)
                transition_system.apply(config, gold_transition)

            # print progress every 1000 sentences
            if (i + 1) % 2000 == 0:
                print(f"  Epoch {epoch}: processed {i + 1}/{len(train_sentences)} sentences", end="\r")

        elapsed = time.time() - start_time
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0
        print(f"  Epoch {epoch}: train accuracy = {accuracy:.2f}% "
              f"({n_correct}/{n_total}) [{elapsed:.1f}s]")

        # evaluate on the dev set after each epoch
        dev_uas = None
        dev_las = None
        if dev_sentences:
            eval_model = model.copy()
            eval_model.average_weights()
            predicted_sents = parse_all(
                dev_sentences,
                eval_model,
                labels,
                transition_system=transition_system,
                feature_extractor=feature_extractor,
            )
            dev_uas, dev_las = evaluate(dev_sentences, predicted_sents)
            print(f"          dev UAS = {dev_uas:.2f}%, LAS = {dev_las:.2f}%")
            is_better_checkpoint = (
                dev_las > best_dev_las
                or (dev_las == best_dev_las and dev_uas > best_dev_uas)
            )
            if checkpoint_dir is not None and is_better_checkpoint:
                best_dev_las = dev_las
                best_dev_uas = dev_uas
                checkpoint_info = _save_best_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    model=eval_model,
                    labels=labels,
                    dev_uas=dev_uas,
                    dev_las=dev_las,
                )
                training_summary["best_dev_checkpoint"] = checkpoint_info

        training_summary["epochs"].append({
            "epoch": epoch,
            "train_accuracy": accuracy,
            "correct_predictions": n_correct,
            "total_predictions": n_total,
            "elapsed_seconds": elapsed,
            "dev_uas": dev_uas,
            "dev_las": dev_las,
            "oracle_transition_counts": dict(epoch_transition_counts),
        })

    # after all epochs, average the weights
    print("-" * 60)
    print("Averaging weights...")
    model.average_weights()
    print("Training complete!")
    training_summary["training_time_seconds"] = sum(
        epoch_info["elapsed_seconds"] for epoch_info in training_summary["epochs"]
    )
    training_summary["oracle_transition_counts"] = dict(training_summary["oracle_transition_counts"])
    return training_summary


def parse_sentence(
    sentence,
    model,
    labels,
    return_details=False,
    repair=True,
    transition_system=None,
    feature_extractor=None,
    decoder=None,
):
    """
    Parse a single sentence using greedy decoding.
    
    Starting from the initial configuration, at each step we:
      1. Extract features
      2. Get valid transitions
      3. Pick the best one according to the model
      4. Apply it
    Until we reach a terminal configuration.
    
    Returns:
        A new sentence (list of token dicts) with predicted 'head' and 'deprel'
    """
    transition_system = transition_system or ArcEagerSystem()
    feature_extractor = feature_extractor or SparseFeatureExtractor()
    decoder = decoder or GreedyDecoder()
    return decoder.parse(
        sentence,
        model,
        transition_system,
        feature_extractor,
        labels,
        return_details=return_details,
        repair=repair,
    )


def parse_all(
    sentences,
    model,
    labels,
    return_details=False,
    repair=True,
    transition_system=None,
    feature_extractor=None,
    decoder=None,
):
    """
    Parse all sentences and return the list of predicted sentences.
    """
    predicted = []
    details = []
    for sentence in sentences:
        if return_details:
            parsed_sentence, parse_details = parse_sentence(
                sentence,
                model,
                labels,
                return_details=True,
                repair=repair,
                transition_system=transition_system,
                feature_extractor=feature_extractor,
                decoder=decoder,
            )
            predicted.append(parsed_sentence)
            details.append(parse_details)
        else:
            predicted.append(
                parse_sentence(
                    sentence,
                    model,
                    labels,
                    repair=repair,
                    transition_system=transition_system,
                    feature_extractor=feature_extractor,
                    decoder=decoder,
                )
            )
    if return_details:
        return predicted, details
    return predicted


def _collect_parse_details(predicted_sentence, config, num_steps):
    """Collect parse diagnostics before any root repair is applied."""
    pre_repair_root_ids = [
        token["id"] for token in predicted_sentence[1:] if token["head"] == 0
    ]
    unattached_ids = [
        token["id"] for token in predicted_sentence[1:] if token["head"] is None
    ]
    return {
        "terminal_reached": config.is_terminal(),
        "num_steps": num_steps,
        "num_arcs": len(config.arcs),
        "pre_repair_root_ids": pre_repair_root_ids,
        "pre_repair_root_count": len(pre_repair_root_ids),
        "pre_repair_unattached_ids": unattached_ids,
        "pre_repair_unattached_count": len(unattached_ids),
    }


def _transition_family(transition):
    """Collapse labeled transitions into high-level transition families."""
    if transition.startswith("LEFT-ARC"):
        return "LEFT-ARC"
    if transition.startswith("RIGHT-ARC"):
        return "RIGHT-ARC"
    return transition


class GreedyDecoder(Decoder):
    """Greedy decoder wrapped in the generic framework API."""

    def parse(
        self,
        sentence,
        model,
        transition_system,
        feature_extractor,
        labels,
        return_details=False,
        repair=True,
    ):
        config = transition_system.initial_config(sentence)
        num_steps = 0

        while not transition_system.is_terminal(config):
            feats = feature_extractor.extract(config)
            valid = transition_system.valid_transitions(config, labels)

            if not valid:
                break

            best_transition, _ = model.predict(feats, valid)
            transition_system.apply(config, best_transition)
            num_steps += 1

        predicted_sentence = copy.deepcopy(sentence)
        for token in predicted_sentence:
            if token["id"] != 0:
                token["head"] = None
                token["deprel"] = "_"

        for head, dep, label in config.arcs:
            predicted_sentence[dep]["head"] = head
            predicted_sentence[dep]["deprel"] = label

        details = _collect_parse_details(predicted_sentence, config, num_steps)

        if repair:
            _ensure_single_root(predicted_sentence)
            details["repair_applied"] = (
                details["pre_repair_unattached_count"] > 0
                or details["pre_repair_root_count"] != 1
            )
            details["post_repair_root_ids"] = [
                token["id"] for token in predicted_sentence[1:] if token["head"] == 0
            ]
            details["post_repair_root_count"] = len(details["post_repair_root_ids"])
        else:
            details["repair_applied"] = False
            details["post_repair_root_ids"] = details["pre_repair_root_ids"]
            details["post_repair_root_count"] = details["pre_repair_root_count"]

        if return_details:
            return predicted_sentence, details
        return predicted_sentence


def _save_best_checkpoint(checkpoint_dir, epoch, model, labels, dev_uas, dev_las):
    """Save the current best dev checkpoint for future comparative experiments."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_path / "best_model.pkl"
    labels_path = checkpoint_path / "best_model_labels.pkl"
    metadata_path = checkpoint_path / "best_checkpoint.json"

    model.save(str(model_path))
    with labels_path.open("wb") as handle:
        pickle.dump(labels, handle)

    metadata = {
        "epoch": epoch,
        "dev_uas": dev_uas,
        "dev_las": dev_las,
        "model_path": str(model_path),
        "labels_path": str(labels_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _ensure_single_root(sentence):
    """
    Repair partially formed outputs so every sentence has exactly one root.

    This keeps evaluation and downstream analysis focused on parser quality
    rather than on malformed outputs caused by greedy decoding mistakes.
    """
    root_idx = _choose_root(sentence)

    for token in sentence[1:]:
        if token["head"] is None:
            if token["id"] == root_idx:
                token["head"] = 0
                token["deprel"] = "root"
            else:
                token["head"] = root_idx
                if token["deprel"] == "_":
                    token["deprel"] = "dep"

    root_candidates = [tok for tok in sentence[1:] if tok["head"] == 0]
    for token in root_candidates:
        if token["id"] != root_idx:
            token["head"] = root_idx
            if token["deprel"] == "root":
                token["deprel"] = "dep"

    sentence[root_idx]["head"] = 0
    sentence[root_idx]["deprel"] = "root"


def _choose_root(sentence):
    """Pick a single sentence root using simple syntax-aware heuristics."""
    explicit_roots = [tok for tok in sentence[1:] if tok["head"] == 0]
    if explicit_roots:
        return _prefer_root_candidate(explicit_roots)["id"]

    unattached = [tok for tok in sentence[1:] if tok["head"] is None]
    if unattached:
        return _prefer_root_candidate(unattached)["id"]

    return _prefer_root_candidate(sentence[1:])["id"]


def _prefer_root_candidate(tokens):
    """Prefer verbal heads, then keep the leftmost candidate."""
    preferred_upos = {"VERB", "AUX"}
    verbal = [tok for tok in tokens if tok["upos"] in preferred_upos]
    if verbal:
        return min(verbal, key=lambda tok: tok["id"])
    return min(tokens, key=lambda tok: tok["id"])
