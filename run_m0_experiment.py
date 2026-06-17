"""Freeze, evaluate, and document the M0 baseline experiment."""

import json
import pickle
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

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
from configuration import Configuration
from data_utils import get_labels, read_conllu, write_conllu
from evaluator import evaluate
from oracle import get_oracle_transition
from parser import parse_all, train
from perceptron import AveragedPerceptron
from transitions import apply_transition


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "m0_arc_eager_static_sparse"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
TRAIN_PATH = Path("data/fr_gsd-ud-train.conllu")
DEV_PATH = Path("data/fr_gsd-ud-dev.conllu")
TEST_PATH = Path("data/fr_gsd-ud-test.conllu")
EPOCHS = 10
SEED = 0


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


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": "M0",
        "description": "Arc-Eager + Static Oracle + Sparse Features + Averaged Perceptron + Greedy Decoding",
        "transition_system": "arc-eager",
        "oracle": "static",
        "features": "sparse handcrafted features",
        "classifier": "averaged perceptron",
        "decoder": "greedy",
        "epochs": EPOCHS,
        "seed": SEED,
        "train_path": str(TRAIN_PATH).replace("\\", "/"),
        "dev_path": str(DEV_PATH).replace("\\", "/"),
        "test_path": str(TEST_PATH).replace("\\", "/"),
    }
    write_json(EXPERIMENT_DIR / "config.json", config)

    log_path = EXPERIMENT_DIR / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            run_experiment(config)


def run_experiment(config):
    print("=" * 80)
    print("Freezing baseline experiment M0")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print()

    train_sentences = read_conllu(str(PROJECT_ROOT / TRAIN_PATH))
    dev_sentences = read_conllu(str(PROJECT_ROOT / DEV_PATH))
    test_sentences = read_conllu(str(PROJECT_ROOT / TEST_PATH))
    labels = get_labels(train_sentences)

    print(f"Train sentences: {len(train_sentences)}")
    print(f"Dev sentences:   {len(dev_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Train tokens:    {count_tokens(train_sentences)}")
    print(f"Dev tokens:      {count_tokens(dev_sentences)}")
    print(f"Test tokens:     {count_tokens(test_sentences)}")
    print(f"Labels:          {len(labels)}")
    print()

    model = AveragedPerceptron()

    training_start = time.time()
    training_summary = train(
        train_sentences,
        dev_sentences,
        model,
        labels,
        n_epochs=EPOCHS,
        seed=SEED,
    )
    training_time = time.time() - training_start

    model_path = EXPERIMENT_DIR / "model.pkl"
    model.save(str(model_path))
    with (EXPERIMENT_DIR / "model_labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)

    dev_predictions, _ = parse_all(dev_sentences, model, labels, return_details=True, repair=True)
    dev_uas, dev_las = evaluate(dev_sentences, dev_predictions)

    parse_start = time.time()
    test_predictions, repaired_parse_details = parse_all(
        test_sentences, model, labels, return_details=True, repair=True
    )
    parsing_time = time.time() - parse_start
    test_uas, test_las = evaluate(test_sentences, test_predictions)
    write_conllu(str(EXPERIMENT_DIR / "predictions.conllu"), test_predictions)

    malformed_predictions, malformed_parse_details = parse_all(
        test_sentences, model, labels, return_details=True, repair=False
    )

    results = build_results(
        train_sentences=train_sentences,
        dev_sentences=dev_sentences,
        test_sentences=test_sentences,
        dev_predictions=dev_predictions,
        test_predictions=test_predictions,
        malformed_predictions=malformed_predictions,
        malformed_parse_details=malformed_parse_details,
        repaired_parse_details=repaired_parse_details,
        training_summary=training_summary,
        training_time=training_time,
        parsing_time=parsing_time,
        dev_uas=dev_uas,
        dev_las=dev_las,
        test_uas=test_uas,
        test_las=test_las,
    )
    write_json(EXPERIMENT_DIR / "results.json", results)

    generate_plots(results, training_summary)
    generate_explanatory_figures(train_sentences)
    write_experiment_readme(results, training_summary)

    print()
    print("Final M0 results")
    print(f"  Dev UAS/LAS:  {dev_uas:.2f} / {dev_las:.2f}")
    print(f"  Test UAS/LAS: {test_uas:.2f} / {test_las:.2f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Parsing time:  {parsing_time:.2f}s")
    print(f"  Test throughput: {results['efficiency']['test_sentences_per_second']:.2f} sent/s, "
          f"{results['efficiency']['test_tokens_per_second']:.2f} tok/s")


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
    """Assemble the experiment results and analysis into a JSON-friendly dict."""
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


def generate_plots(results, training_summary):
    """Generate the required M0 result figures."""
    plot_training_curves(training_summary["epochs"], FIGURES_DIR / "training_curves.png")
    plot_final_scores(results, FIGURES_DIR / "final_scores_dev_test.png")
    plot_las_by_label(results["analysis"]["las_by_label"], FIGURES_DIR / "las_by_label.png")
    plot_scores_by_length(
        results["analysis"]["scores_by_sentence_length"],
        FIGURES_DIR / "scores_by_sentence_length.png",
    )
    plot_label_confusion(
        results["analysis"]["label_confusion_matrix"],
        FIGURES_DIR / "label_confusion_matrix.png",
    )
    plot_root_analysis(
        results["analysis"]["malformed_parse_stats_before_repair"],
        FIGURES_DIR / "root_analysis.png",
    )
    plot_transition_distribution(
        results["transition_distribution"],
        FIGURES_DIR / "transition_distribution.png",
    )
    plot_efficiency_metrics(results["efficiency"], FIGURES_DIR / "efficiency_metrics.png")


def generate_explanatory_figures(train_sentences):
    """Create explanatory diagrams for Arc-Eager and one full parse sequence."""
    plot_arc_eager_schema(FIGURES_DIR / "arc_eager_transitions_schema.png")
    plot_example_parse_sequence(FIGURES_DIR / "example_parse_sequence.png")


def plot_training_curves(epoch_history, output_path):
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
    plt.title("M0 Training Curves")
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_final_scores(results, output_path):
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
    plt.title("Final Dev/Test Scores for M0")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_las_by_label(label_metrics, output_path, top_n=15):
    top = label_metrics[:top_n]
    labels = [entry["label"] for entry in top]
    scores = [entry["las"] for entry in top]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LAS (%)")
    plt.ylim(0, 100)
    plt.title("LAS by Dependency Label (Most Frequent Labels)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_scores_by_length(bucket_metrics, output_path):
    buckets = [entry["bucket"] for entry in bucket_metrics]
    uas = [entry["uas"] for entry in bucket_metrics]
    las = [entry["las"] for entry in bucket_metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(buckets, uas, marker="o", label="UAS")
    plt.plot(buckets, las, marker="s", label="LAS")
    plt.ylabel("Score (%)")
    plt.title("Scores by Sentence Length Bucket")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_label_confusion(confusion_data, output_path):
    labels = confusion_data["labels"]
    matrix = confusion_data["matrix"]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Gold label")
    plt.title("Label Confusion Matrix (Most Frequent Labels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_root_analysis(root_stats, output_path):
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
    plt.title("Root and Repair Analysis")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_transition_distribution(transition_counts, output_path):
    labels = list(transition_counts.keys())
    values = [transition_counts[label] for label in labels]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title("Oracle Transition Distribution During Training")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_efficiency_metrics(efficiency, output_path):
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
    plt.title("Efficiency Metrics for M0")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_arc_eager_schema(output_path):
    plt.figure(figsize=(11, 6))
    plt.axis("off")
    plt.title("Arc-Eager Transition System", fontsize=16, pad=12)
    plt.text(0.05, 0.84, "Configuration", fontsize=13, weight="bold")
    plt.text(0.05, 0.72, "Stack:   [ROOT, s1, s0]", fontsize=12)
    plt.text(0.05, 0.64, "Buffer:  [b0, b1, ...]", fontsize=12)
    plt.text(0.05, 0.56, "Arcs:    built dependencies so far", fontsize=12)

    x = 0.45
    plt.text(x, 0.82, "SHIFT", fontsize=13, weight="bold")
    plt.text(x, 0.73, "Move b0 from buffer to stack.", fontsize=11)
    plt.text(x, 0.60, "LEFT-ARC(label)", fontsize=13, weight="bold")
    plt.text(x, 0.51, "Add b0 -> s0 and pop s0.", fontsize=11)
    plt.text(x, 0.38, "RIGHT-ARC(label)", fontsize=13, weight="bold")
    plt.text(x, 0.29, "Add s0 -> b0, pop b0 from buffer,\npush it onto stack.", fontsize=11)
    plt.text(x, 0.12, "REDUCE", fontsize=13, weight="bold")
    plt.text(x, 0.05, "Pop s0 once it already has a head.", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_example_parse_sequence(output_path):
    sentence = [
        {"id": 0, "form": "ROOT", "lemma": "ROOT", "upos": "ROOT", "head": -1, "deprel": "ROOT"},
        {"id": 1, "form": "Je", "lemma": "je", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"id": 2, "form": "mange", "lemma": "manger", "upos": "VERB", "head": 0, "deprel": "root"},
    ]
    config = Configuration(sentence)
    rows = []
    step = 0
    while True:
        rows.append([
            str(step),
            format_stack(config, sentence),
            format_buffer(config, sentence),
            format_arcs(config, sentence),
            get_oracle_transition(config) if not config.is_terminal() else "STOP",
        ])
        if config.is_terminal():
            break
        transition = get_oracle_transition(config)
        apply_transition(config, transition)
        step += 1

    fig, ax = plt.subplots(figsize=(12, 2 + 0.55 * len(rows)))
    ax.axis("off")
    ax.set_title("Example Arc-Eager Parse Sequence for 'Je mange'", fontsize=15, pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=["Step", "Stack", "Buffer", "Arcs", "Oracle transition"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def format_stack(config, sentence):
    return "[" + ", ".join(sentence[idx]["form"] for idx in config.stack) + "]"


def format_buffer(config, sentence):
    return "[" + ", ".join(sentence[idx]["form"] for idx in config.buffer) + "]"


def format_arcs(config, sentence):
    if not config.arcs:
        return "{}"
    parts = []
    for head, dep, label in config.arcs:
        parts.append(f"{sentence[head]['form']}->{sentence[dep]['form']} ({label})")
    return "; ".join(parts)


def write_experiment_readme(results, training_summary):
    label_metrics = results["analysis"]["las_by_label"][:5]
    root_stats = results["analysis"]["malformed_parse_stats_before_repair"]
    lines = [
        "# M0 Baseline Experiment",
        "",
        "## Goal",
        "",
        "Freeze and document the baseline parser before comparing it with more advanced projective parsing strategies.",
        "",
        "Model definition:",
        "",
        "- Arc-Eager transition system",
        "- Static oracle",
        "- Sparse handcrafted features",
        "- Averaged perceptron classifier",
        "- Greedy decoding",
        "",
        "## Architecture",
        "",
        "The parser processes each sentence with a stack, a buffer, and a set of dependency arcs.",
        "At training time, a static oracle provides the gold transition for each configuration.",
        "The averaged perceptron scores valid transitions from sparse local features.",
        "At test time, decoding is greedy: the highest-scoring valid transition is applied at each step.",
        "",
        "## Arc-Eager Transition System",
        "",
        "- `SHIFT`: move the buffer front onto the stack",
        "- `LEFT-ARC(label)`: add `b0 -> s0` and pop `s0`",
        "- `RIGHT-ARC(label)`: add `s0 -> b0` and push `b0` onto the stack",
        "- `REDUCE`: pop the stack top once it already has a head",
        "",
        "## Static Oracle",
        "",
        "The static oracle chooses:",
        "",
        "- `LEFT-ARC` when the buffer front is the gold head of the stack top",
        "- `RIGHT-ARC` when the stack top is the gold head of the buffer front",
        "- `REDUCE` when the stack top already has a head and all of its gold dependents are attached",
        "- `SHIFT` otherwise",
        "",
        "## Sparse Feature Templates",
        "",
        "- lexical features for `s0`, `s1`, `b0`, `b1`",
        "- UPOS features and POS n-grams",
        "- word + POS combinations",
        "- leftmost/rightmost child features",
        "- dependent count of `s0`",
        "- distance, stack size, and buffer size",
        "- bias feature",
        "",
        "## Averaged Perceptron",
        "",
        "The classifier is an averaged perceptron over sparse feature vectors.",
        "Weights are updated online from oracle supervision, then averaged at the end of training.",
        "",
        "## Greedy Decoding",
        "",
        "At each parser step, the model scores the valid transitions and applies the best one.",
        "A lightweight post-processing step repairs malformed outputs so each final tree has exactly one root.",
        "",
        "## Evaluation Metrics",
        "",
        "- `UAS`: head accuracy",
        "- `LAS`: head + label accuracy",
        "- root accuracy and malformed parse analysis are also reported",
        "",
        "## Final Results",
        "",
        f"- Dev UAS/LAS: `{results['dev']['uas']:.2f}` / `{results['dev']['las']:.2f}`",
        f"- Test UAS/LAS: `{results['test']['uas']:.2f}` / `{results['test']['las']:.2f}`",
        f"- Training time: `{results['efficiency']['training_time_seconds']:.2f}s`",
        f"- Parsing time: `{results['efficiency']['parsing_time_seconds']:.2f}s`",
        f"- Test throughput: `{results['efficiency']['test_sentences_per_second']:.2f}` sent/s, `{results['efficiency']['test_tokens_per_second']:.2f}` tok/s",
        "",
        "## Key Observations From Error Analysis",
        "",
        f"- Root accuracy after repair: `{results['analysis']['root_accuracy']['accuracy']:.2f}%`",
        f"- Malformed sentences before repair: `{root_stats['malformed_sentences_before_repair']}` / `{root_stats['total_sentences']}`",
        f"- Repaired sentences: `{root_stats['repaired_sentences']}`",
        "- Most frequent labels and their LAS:",
    ]
    for entry in label_metrics:
        lines.append(f"  - `{entry['label']}`: {entry['las']:.2f}% over {entry['support']} tokens")

    lines.extend([
        "",
        "## Files",
        "",
        "- `model.pkl`: trained perceptron weights",
        "- `model_labels.pkl`: dependency label vocabulary",
        "- `predictions.conllu`: predicted test trees",
        "- `config.json`: experiment configuration",
        "- `results.json`: metrics and analysis summary",
        "- `training_log.txt`: full console log",
        "- `figures/`: plots and explanatory diagrams",
    ])

    (EXPERIMENT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
matplotlib.use("Agg")
