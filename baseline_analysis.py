"""Utilities for analyzing and plotting baseline parsing experiments."""

from collections import Counter, defaultdict


def count_tokens(sentences):
    """Count non-ROOT tokens across a list of sentences."""
    return sum(len(sentence) - 1 for sentence in sentences)


def get_sentence_root_id(sentence):
    """Return the token id attached to ROOT, or None if absent/malformed."""
    root_ids = [token["id"] for token in sentence[1:] if token["head"] == 0]
    if len(root_ids) != 1:
        return None
    return root_ids[0]


def sentence_length_bucket(length):
    """Bucket sentence lengths into broad ranges for analysis plots."""
    if length <= 10:
        return "1-10"
    if length <= 20:
        return "11-20"
    if length <= 30:
        return "21-30"
    if length <= 40:
        return "31-40"
    return "41+"


def compute_label_metrics(gold_sentences, predicted_sentences):
    """Compute LAS by dependency label together with label support counts."""
    total_by_label = Counter()
    correct_by_label = Counter()

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        for gold_token, predicted_token in zip(gold_sentence[1:], predicted_sentence[1:]):
            label = gold_token["deprel"]
            total_by_label[label] += 1
            if (
                predicted_token["head"] == gold_token["head"]
                and predicted_token["deprel"] == gold_token["deprel"]
            ):
                correct_by_label[label] += 1

    metrics = []
    for label, support in total_by_label.most_common():
        las = 100.0 * correct_by_label[label] / support if support else 0.0
        metrics.append({
            "label": label,
            "support": support,
            "correct": correct_by_label[label],
            "las": las,
        })
    return metrics


def compute_length_bucket_metrics(gold_sentences, predicted_sentences):
    """Compute UAS/LAS grouped by sentence-length bucket."""
    bucket_stats = defaultdict(lambda: {"correct_head": 0, "correct_label": 0, "total": 0, "sentences": 0})

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        bucket = sentence_length_bucket(len(gold_sentence) - 1)
        bucket_stats[bucket]["sentences"] += 1
        for gold_token, predicted_token in zip(gold_sentence[1:], predicted_sentence[1:]):
            bucket_stats[bucket]["total"] += 1
            if predicted_token["head"] == gold_token["head"]:
                bucket_stats[bucket]["correct_head"] += 1
                if predicted_token["deprel"] == gold_token["deprel"]:
                    bucket_stats[bucket]["correct_label"] += 1

    ordered_buckets = ["1-10", "11-20", "21-30", "31-40", "41+"]
    results = []
    for bucket in ordered_buckets:
        stats = bucket_stats[bucket]
        total = stats["total"]
        results.append({
            "bucket": bucket,
            "sentences": stats["sentences"],
            "tokens": total,
            "uas": 100.0 * stats["correct_head"] / total if total else 0.0,
            "las": 100.0 * stats["correct_label"] / total if total else 0.0,
        })
    return results


def compute_root_accuracy(gold_sentences, predicted_sentences):
    """Compute sentence-level root accuracy after parsing."""
    correct = 0
    total = len(gold_sentences)
    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        if get_sentence_root_id(gold_sentence) == get_sentence_root_id(predicted_sentence):
            correct += 1
    accuracy = 100.0 * correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def compute_label_confusion(gold_sentences, predicted_sentences, top_n=12):
    """Build a confusion matrix for the most frequent gold dependency labels."""
    label_support = Counter()
    confusion = defaultdict(Counter)

    for gold_sentence, predicted_sentence in zip(gold_sentences, predicted_sentences):
        for gold_token, predicted_token in zip(gold_sentence[1:], predicted_sentence[1:]):
            gold_label = gold_token["deprel"]
            pred_label = predicted_token["deprel"]
            label_support[gold_label] += 1
            confusion[gold_label][pred_label] += 1

    labels = [label for label, _ in label_support.most_common(top_n)]
    matrix = []
    for gold_label in labels:
        row = []
        for predicted_label in labels:
            row.append(confusion[gold_label][predicted_label])
        matrix.append(row)
    return {
        "labels": labels,
        "matrix": matrix,
        "support": {label: label_support[label] for label in labels},
    }


def compute_malformed_parse_stats(gold_sentences, parse_details, repaired_predictions):
    """Summarize malformed parse behavior before repair and root behavior after repair."""
    zero_root = 0
    multi_root = 0
    unattached = 0
    malformed = 0
    repaired = 0
    correct_root_before_repair = 0
    correct_root_after_repair = 0

    for gold_sentence, details, repaired_sentence in zip(gold_sentences, parse_details, repaired_predictions):
        root_count = details["pre_repair_root_count"]
        unattached_count = details["pre_repair_unattached_count"]
        malformed_here = root_count != 1 or unattached_count > 0

        if root_count == 0:
            zero_root += 1
        if root_count > 1:
            multi_root += 1
        if unattached_count > 0:
            unattached += 1
        if malformed_here:
            malformed += 1
        if details["repair_applied"]:
            repaired += 1

        gold_root = get_sentence_root_id(gold_sentence)
        if root_count == 1 and details["pre_repair_root_ids"][0] == gold_root:
            correct_root_before_repair += 1
        if get_sentence_root_id(repaired_sentence) == gold_root:
            correct_root_after_repair += 1

    total = len(parse_details)
    return {
        "total_sentences": total,
        "zero_root_sentences_before_repair": zero_root,
        "multi_root_sentences_before_repair": multi_root,
        "unattached_sentences_before_repair": unattached,
        "malformed_sentences_before_repair": malformed,
        "repaired_sentences": repaired,
        "correct_root_sentences_before_repair": correct_root_before_repair,
        "correct_root_sentences_after_repair": correct_root_after_repair,
    }

