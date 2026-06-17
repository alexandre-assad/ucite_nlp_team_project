"""Train and document the M2 Arc-Standard dense baseline experiment."""

import argparse
import json
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from baseline_analysis import count_tokens
from data_utils import get_labels, read_conllu, write_conllu
from dense_embeddings import build_embedding_provider
from dense_features import DenseFeatureExtractor
from dense_model import DenseMLPTransitionClassifier
from dense_training import train_dense_model
from evaluator import evaluate
from experiment_utils import Tee, build_results, write_json
from oracle import StaticArcStandardOracle
from parser import parse_all
from seed_utils import set_global_seed
from transitions import ArcStandardSystem


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "m2_arc_standard_static_dense"
DEFAULT_CONFIG_PATH = DEFAULT_EXPERIMENT_DIR / "config.json"


def main():
    arg_parser = argparse.ArgumentParser(description="Run the M2 Arc-Standard dense experiment")
    arg_parser.parse_args()

    experiment_dir = DEFAULT_EXPERIMENT_DIR
    figures_dir = experiment_dir / "figures"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    write_json(experiment_dir / "config.json", config)

    log_path = experiment_dir / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            run_experiment(config, experiment_dir, figures_dir)


def run_experiment(config, experiment_dir, figures_dir):
    print("=" * 80)
    print(f"Running {config['model_id']} Arc-Standard dense experiment")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print()

    set_global_seed(config["seed"], use_cuda=True)

    raw_train_sentences = read_conllu(str(PROJECT_ROOT / config["train_path"]))
    dev_sentences = read_conllu(str(PROJECT_ROOT / config["dev_path"]))
    test_sentences = read_conllu(str(PROJECT_ROOT / config["test_path"]))

    labels = get_labels(raw_train_sentences)
    train_sentences = [sentence for sentence in raw_train_sentences if is_projective(sentence)]

    embedding_provider = build_embedding_provider(config["embedding"], PROJECT_ROOT)
    feature_extractor = DenseFeatureExtractor(embedding_provider)
    transition_system = ArcStandardSystem()
    oracle = StaticArcStandardOracle()
    device = "cuda" if config["training"]["use_gpu_if_available"] and _torch_cuda_available() else "cpu"

    print(f"Train sentences kept for M2: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"Dev sentences:   {len(dev_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Train tokens:    {count_tokens(train_sentences)}")
    print(f"Dev tokens:      {count_tokens(dev_sentences)}")
    print(f"Test tokens:     {count_tokens(test_sentences)}")
    print(f"Labels:          {len(labels)}")
    print(f"Embedding mode:  {config['embedding']['source']}")
    print(f"Embedding dim:   {config['embedding']['embedding_dim']}")
    print(f"Device:          {device}")
    print()

    model = DenseMLPTransitionClassifier(
        transition_classes=build_transition_classes(labels),
        input_dim=feature_extractor.output_dim,
        hidden_dim=config["classifier"]["hidden_dim"],
        dropout=config["classifier"]["dropout"],
        device=device,
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    batch_size = config["training"]["batch_size"]

    training_start = time.time()
    training_summary = train_dense_model(
        train_sentences=train_sentences,
        dev_sentences=dev_sentences,
        model=model,
        labels=labels,
        feature_extractor=feature_extractor,
        transition_system=transition_system,
        oracle=oracle,
        batch_size=batch_size,
        n_epochs=config["epochs"],
        seed=config["seed"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_dir=checkpoint_dir,
    )
    training_time = time.time() - training_start

    model_final_path = experiment_dir / "model_final.pt"
    model.save(str(model_final_path), extra_state={"selected_epoch": config["epochs"]})

    best_checkpoint = training_summary.get("best_dev_checkpoint")
    if best_checkpoint is None:
        raise RuntimeError("No best-dev checkpoint was saved for M2")
    shutil.copyfile(PROJECT_ROOT / best_checkpoint["model_path"], experiment_dir / "model_best_dev.pt")

    best_model = DenseMLPTransitionClassifier.load(str(experiment_dir / "model_best_dev.pt"), device=device)

    dev_predictions, _ = parse_all(
        dev_sentences,
        best_model,
        labels,
        return_details=True,
        transition_system=transition_system,
        feature_extractor=feature_extractor,
    )
    dev_uas, dev_las = evaluate(dev_sentences, dev_predictions)

    parse_start = time.time()
    test_predictions, repaired_parse_details = parse_all(
        test_sentences,
        best_model,
        labels,
        return_details=True,
        transition_system=transition_system,
        feature_extractor=feature_extractor,
    )
    parsing_time = time.time() - parse_start
    test_uas, test_las = evaluate(test_sentences, test_predictions)
    write_conllu(str(experiment_dir / "predictions.conllu"), test_predictions)

    malformed_predictions, malformed_parse_details = parse_all(
        test_sentences,
        best_model,
        labels,
        return_details=True,
        repair=False,
        transition_system=transition_system,
        feature_extractor=feature_extractor,
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
    results.update(
        {
            "model_id": config["model_id"],
            "description": config["description"],
            "transition_system": config["transition_system"],
            "oracle": config["oracle"],
            "features": config["features"],
            "classifier": config["classifier"]["type"],
            "decoder": config["decoder"],
            "epochs": config["epochs"],
            "selected_epoch": best_checkpoint["epoch"],
            "seed": config["seed"],
            "train_sentences_original": len(raw_train_sentences),
            "train_sentences_kept": len(train_sentences),
            "train_sentences_filtered": len(raw_train_sentences) - len(train_sentences),
            "train_tokens": count_tokens(train_sentences),
            "dev_sentences": len(dev_sentences),
            "test_sentences": len(test_sentences),
            "dev_tokens": count_tokens(dev_sentences),
            "test_tokens": count_tokens(test_sentences),
            "dev_uas": round(dev_uas, 2),
            "dev_las": round(dev_las, 2),
            "test_uas": round(test_uas, 2),
            "test_las": round(test_las, 2),
            "embedding": config["embedding"],
            "paths": {
                "train_path": config["train_path"],
                "dev_path": config["dev_path"],
                "test_path": config["test_path"],
                "output_dir": config["output_dir"],
            },
            "selection_metric": {
                "primary": "dev_las",
                "tie_breaker": "dev_uas",
            },
        }
    )
    write_json(experiment_dir / "results.json", results)

    generate_m2_plots(results, training_summary, figures_dir, config["model_id"])
    write_m2_readme(results, config, experiment_dir)

    print()
    print("Final M2 results (best-dev checkpoint)")
    print(f"  Projective training sentences kept: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"  Generated transition examples: {training_summary['generated_transition_examples']}")
    print(f"  Selected epoch: {best_checkpoint['epoch']}")
    print(f"  Dev UAS/LAS:  {dev_uas:.2f} / {dev_las:.2f}")
    print(f"  Test UAS/LAS: {test_uas:.2f} / {test_las:.2f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Parsing time:  {parsing_time:.2f}s")


def build_transition_classes(labels):
    classes = ["SHIFT"]
    classes.extend([f"LEFT-ARC:{label}" for label in labels])
    classes.extend([f"RIGHT-ARC:{label}" for label in labels])
    return classes


def write_m2_readme(results, config, experiment_dir):
    embedding_path = config["embedding"].get("path")
    lines = [
        "# M2 Arc-Standard Dense Baseline Experiment",
        "",
        "## Goal",
        "",
        "Compare M1 sparse handcrafted features against static dense embeddings while keeping",
        "Arc-Standard, the static oracle, and greedy decoding fixed.",
        "",
        "## M2 definition",
        "",
        "- Arc-Standard transition system",
        "- Static Arc-Standard oracle",
        "- Static dense embeddings over s1, s0, b0, and b1",
        "- MLP transition classifier",
        "- Greedy decoding",
        "",
        "## Embeddings",
        "",
        f"- Embedding source: `{config['embedding']['source']}`",
        f"- Embedding dimension: `{config['embedding']['embedding_dim']}`",
        f"- Embedding path: `{embedding_path}`",
        "- Lookup key: lemma.lower() if available and not `_`, otherwise form.lower()",
        f"- OOV strategy: `{config['embedding']['oov_strategy']}`",
        "",
        "Place the FastText file at `data/embeddings/cc.fr.300.bin`.",
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{config['seed']}`",
        "- Paths are stored relative to the project root",
        "- Best-dev checkpointing uses dev LAS with dev UAS as tie-breaker",
        "",
        "## Final results",
        "",
        f"- Selected epoch: `{results['selected_epoch']}`",
        f"- Dev UAS/LAS: `{results['dev']['uas']:.2f}` / `{results['dev']['las']:.2f}`",
        f"- Test UAS/LAS: `{results['test']['uas']:.2f}` / `{results['test']['las']:.2f}`",
        "",
        "## Outputs",
        "",
        "- `config.json`",
        "- `results.json`",
        "- `training_log.txt`",
        "- `README.md`",
        "- `model_best_dev.pt`",
        "- `model_final.pt`",
        "- `predictions.conllu`",
        "- `figures/`",
    ]
    (experiment_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_m2_plots(results, training_summary, figures_dir, title_prefix):
    plot_training_loss(training_summary["epochs"], figures_dir / "training_loss_curve.png", title_prefix)
    plot_dev_curves(training_summary["epochs"], figures_dir / "dev_uas_las_curve.png", title_prefix)
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
    plot_efficiency(results["efficiency"], figures_dir / "efficiency_metrics.png", title_prefix)


def plot_training_loss(epoch_history, output_path, title_prefix):
    epochs = [entry["epoch"] for entry in epoch_history]
    losses = [entry["train_loss"] for entry in epoch_history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{title_prefix} Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_dev_curves(epoch_history, output_path, title_prefix):
    epochs = [entry["epoch"] for entry in epoch_history]
    dev_uas = [entry["dev_uas"] for entry in epoch_history]
    dev_las = [entry["dev_las"] for entry in epoch_history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, dev_uas, marker="o", label="Dev UAS")
    plt.plot(epochs, dev_las, marker="s", label="Dev LAS")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title(f"{title_prefix} Dev UAS/LAS")
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


def plot_efficiency(efficiency, output_path, title_prefix):
    labels = ["Training time (s)", "Parsing time (s)", "Sent/s", "Tok/s"]
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


def is_projective(sentence):
    arcs = []
    for token in sentence[1:]:
        head = token["head"]
        dep = token["id"]
        left, right = sorted((head, dep))
        arcs.append((left, right))

    for i, (a_left, a_right) in enumerate(arcs):
        for b_left, b_right in arcs[i + 1 :]:
            if a_left < b_left < a_right < b_right:
                return False
            if b_left < a_left < b_right < a_right:
                return False
    return True


def _torch_cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
