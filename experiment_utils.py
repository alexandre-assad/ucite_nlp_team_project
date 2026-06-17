"""Shared utilities for experiment runners."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from baseline_analysis import (
    compute_label_confusion,
    compute_label_metrics,
    compute_length_bucket_metrics,
    compute_malformed_parse_stats,
    compute_root_accuracy,
    count_tokens,
)


class Tee:
    """Mirror stdout to both console and log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_results(
    train_sentences,
    dev_sentences,
    test_sentences,
    dev_predictions,
    test_predictions,
    malformed_predictions,
    malformed_parse_details,
    repaired_parse_details,
    training_summary,
    training_time,
    parsing_time,
    dev_uas,
    dev_las,
    test_uas,
    test_las,
):
    """Assemble experiment metrics and analysis into a JSON-friendly dict."""
    label_metrics = compute_label_metrics(test_sentences, test_predictions)
    length_bucket_metrics = compute_length_bucket_metrics(test_sentences, test_predictions)
    root_accuracy = compute_root_accuracy(test_sentences, test_predictions)
    malformed_stats = compute_malformed_parse_stats(
        test_sentences, malformed_parse_details, test_predictions
    )
    label_confusion = compute_label_confusion(test_sentences, test_predictions)

    test_tokens = count_tokens(test_sentences)
    results = {
        "dev": {
            "uas": dev_uas,
            "las": dev_las,
            "sentences": len(dev_sentences),
            "tokens": count_tokens(dev_sentences),
        },
        "test": {
            "uas": test_uas,
            "las": test_las,
            "sentences": len(test_sentences),
            "tokens": test_tokens,
        },
        "train": {
            "sentences": len(train_sentences),
            "tokens": count_tokens(train_sentences),
        },
        "efficiency": {
            "training_time_seconds": training_time,
            "parsing_time_seconds": parsing_time,
            "test_sentences_per_second": len(test_sentences) / parsing_time if parsing_time else 0.0,
            "test_tokens_per_second": test_tokens / parsing_time if parsing_time else 0.0,
        },
        "epoch_history": training_summary["epochs"],
        "transition_distribution": training_summary["oracle_transition_counts"],
        "analysis": {
            "las_by_label": label_metrics,
            "scores_by_sentence_length": length_bucket_metrics,
            "root_accuracy": root_accuracy,
            "label_confusion_matrix": label_confusion,
            "malformed_parse_stats_before_repair": malformed_stats,
            "parse_detail_sample": malformed_parse_details[:5],
        },
    }
    return results


def generate_plots(results, training_summary, figures_dir, title_prefix):
    """Generate experiment result figures."""
    plot_training_curves(training_summary["epochs"], figures_dir / "training_curves.png", title_prefix)
    plot_final_scores(results, figures_dir / "final_scores_dev_test.png", title_prefix)
    plot_las_by_label(results["analysis"]["las_by_label"], figures_dir / "las_by_label.png", title_prefix)
    plot_scores_by_length(
        results["analysis"]["scores_by_sentence_length"],
        figures_dir / "scores_by_sentence_length.png",
        title_prefix,
    )
    plot_label_confusion(
        results["analysis"]["label_confusion_matrix"],
        figures_dir / "label_confusion_matrix.png",
        title_prefix,
    )
    plot_root_analysis(
        results["analysis"]["malformed_parse_stats_before_repair"],
        figures_dir / "root_analysis.png",
        title_prefix,
    )
    plot_transition_distribution(
        results["transition_distribution"],
        figures_dir / "transition_distribution.png",
        title_prefix,
    )
    plot_efficiency_metrics(results["efficiency"], figures_dir / "efficiency_metrics.png", title_prefix)


def plot_training_curves(epoch_history, output_path, title_prefix):
    epochs = [entry["epoch"] for entry in epoch_history]
    train_acc = [entry["train_accuracy"] for entry in epoch_history]
    dev_uas = [entry["dev_uas"] for entry in epoch_history]
    dev_las = [entry["dev_las"] for entry in epoch_history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_acc, marker="o", label="Train transition accuracy")
    plt.plot(epochs, dev_uas, marker="s", label="Dev UAS")
    plt.plot(epochs, dev_las, marker="^", label="Dev LAS")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title(f"{title_prefix} Training Curves")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_final_scores(results, output_path, title_prefix):
    labels = ["UAS", "LAS"]
    dev_scores = [results["dev"]["uas"], results["dev"]["las"]]
    test_scores = [results["test"]["uas"], results["test"]["las"]]
    x = range(len(labels))

    plt.figure(figsize=(7, 5))
    plt.bar([i - 0.18 for i in x], dev_scores, width=0.36, label="Dev")
    plt.bar([i + 0.18 for i in x], test_scores, width=0.36, label="Test")
    plt.xticks(list(x), labels)
    plt.ylim(0, 100)
    plt.ylabel("Score (%)")
    plt.title(f"Final Dev/Test Scores for {title_prefix}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_las_by_label(label_metrics, output_path, title_prefix, top_n=15):
    top = label_metrics[:top_n]
    labels = [entry["label"] for entry in top]
    scores = [entry["las"] for entry in top]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LAS (%)")
    plt.ylim(0, 100)
    plt.title(f"LAS by Dependency Label ({title_prefix})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_scores_by_length(bucket_metrics, output_path, title_prefix):
    buckets = [entry["bucket"] for entry in bucket_metrics]
    uas = [entry["uas"] for entry in bucket_metrics]
    las = [entry["las"] for entry in bucket_metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(buckets, uas, marker="o", label="UAS")
    plt.plot(buckets, las, marker="s", label="LAS")
    plt.ylabel("Score (%)")
    plt.title(f"Scores by Sentence Length Bucket ({title_prefix})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_label_confusion(confusion_data, output_path, title_prefix):
    labels = confusion_data["labels"]
    matrix = confusion_data["matrix"]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Gold label")
    plt.title(f"Label Confusion Matrix ({title_prefix})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_root_analysis(root_stats, output_path, title_prefix):
    labels = [
        "Correct root\nbefore repair",
        "Malformed root\nbefore repair",
        "Repaired\nsentences",
    ]
    values = [
        root_stats["correct_root_sentences_before_repair"],
        root_stats["malformed_sentences_before_repair"],
        root_stats["repaired_sentences"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Sentences")
    plt.title(f"Root and Repair Analysis ({title_prefix})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_transition_distribution(transition_counts, output_path, title_prefix):
    labels = list(transition_counts.keys())
    values = [transition_counts[label] for label in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title(f"Oracle Transition Distribution ({title_prefix})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_efficiency_metrics(efficiency, output_path, title_prefix):
    labels = [
        "Training time (s)",
        "Parsing time (s)",
        "Sent/s",
        "Tok/s",
    ]
    values = [
        efficiency["training_time_seconds"],
        efficiency["parsing_time_seconds"],
        efficiency["test_sentences_per_second"],
        efficiency["test_tokens_per_second"],
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(f"Efficiency Metrics ({title_prefix})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
