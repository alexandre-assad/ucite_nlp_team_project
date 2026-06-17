"""Train and document the M2b Arc-Standard dense-rich experiment."""

import argparse
import json
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from baseline_analysis import count_tokens
from data_utils import get_labels, read_conllu, write_conllu
from dense_embeddings import build_embedding_provider
from dense_rich_features import DenseRichFeatureExtractor, build_label_vocab, build_pos_vocab
from dense_rich_model import DenseRichTransitionClassifier
from dense_rich_training import train_dense_rich_model
from evaluator import evaluate
from experiment_utils import Tee, build_results, write_json
from oracle import StaticArcStandardOracle
from parser import parse_all
from run_m2_dense_experiment import build_transition_classes, generate_m2_plots, is_projective
from seed_utils import set_global_seed
from transitions import ArcStandardSystem


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "m2b_arc_standard_dense_rich"
DEFAULT_CONFIG_PATH = DEFAULT_EXPERIMENT_DIR / "config.json"


def main():
    arg_parser = argparse.ArgumentParser(description="Run the M2b Arc-Standard dense-rich experiment")
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
    print(f"Running {config['model_id']} Arc-Standard dense-rich experiment")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print()

    set_global_seed(config["seed"], use_cuda=True)

    raw_train_sentences = read_conllu(str(PROJECT_ROOT / config["train_path"]))
    dev_sentences = read_conllu(str(PROJECT_ROOT / config["dev_path"]))
    test_sentences = read_conllu(str(PROJECT_ROOT / config["test_path"]))

    labels = get_labels(raw_train_sentences)
    train_sentences = [sentence for sentence in raw_train_sentences if is_projective(sentence)]

    pos_vocab = build_pos_vocab(train_sentences)
    label_vocab = build_label_vocab(train_sentences)
    embedding_provider = build_embedding_provider(config["embedding"], PROJECT_ROOT)
    feature_extractor = DenseRichFeatureExtractor(
        embedding_provider=embedding_provider,
        pos_vocab=pos_vocab,
        label_vocab=label_vocab,
    )
    transition_system = ArcStandardSystem()
    oracle = StaticArcStandardOracle()
    device = "cuda" if config["training"]["use_gpu_if_available"] and _torch_cuda_available() else "cpu"

    print(f"Train sentences kept for M2b: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"Dev sentences:   {len(dev_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Train tokens:    {count_tokens(train_sentences)}")
    print(f"Dev tokens:      {count_tokens(dev_sentences)}")
    print(f"Test tokens:     {count_tokens(test_sentences)}")
    print(f"Labels:          {len(labels)}")
    print(f"POS vocab:       {len(pos_vocab)}")
    print(f"Label vocab:     {len(label_vocab)}")
    print(f"Embedding mode:  {config['embedding']['source']}")
    print(f"Embedding dim:   {config['embedding']['embedding_dim']}")
    print(f"Device:          {device}")
    print()

    model = DenseRichTransitionClassifier(
        transition_classes=build_transition_classes(labels),
        word_embedding_dim=config["embedding"]["embedding_dim"],
        pos_vocab_size=len(pos_vocab),
        label_vocab_size=len(label_vocab),
        pos_embedding_dim=config["classifier"]["pos_embedding_dim"],
        label_embedding_dim=config["classifier"]["label_embedding_dim"],
        hidden_dim=config["classifier"]["hidden_dim"],
        dropout=config["classifier"]["dropout"],
        device=device,
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    batch_size = config["training"]["batch_size"]

    training_start = time.time()
    training_summary = train_dense_rich_model(
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
        raise RuntimeError("No best-dev checkpoint was saved for M2b")
    shutil.copyfile(PROJECT_ROOT / best_checkpoint["model_path"], experiment_dir / "model_best_dev.pt")

    best_model = DenseRichTransitionClassifier.load(str(experiment_dir / "model_best_dev.pt"), device=device)

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
            "feature_set_description": (
                "Word embeddings for s0/s1/s2/b0/b1/b2 and selected children, "
                "POS embeddings for the same token positions, dependency-label "
                "embeddings for child arcs, plus distance/stack/buffer structural buckets "
                "and NULL/ROOT indicators"
            ),
            "classifier": config["classifier"]["type"],
            "decoder": config["decoder"],
            "epochs": config["epochs"],
            "selected_epoch": best_checkpoint["epoch"],
            "seed": config["seed"],
            "train_sentences_original": len(raw_train_sentences),
            "train_sentences_kept": len(train_sentences),
            "train_sentences_filtered": len(raw_train_sentences) - len(train_sentences),
            "generated_transition_examples": training_summary["generated_transition_examples"],
            "train_tokens": count_tokens(train_sentences),
            "dev_sentences": len(dev_sentences),
            "test_sentences": len(test_sentences),
            "dev_tokens": count_tokens(dev_sentences),
            "test_tokens": count_tokens(test_sentences),
            "dev_uas": round(dev_uas, 2),
            "dev_las": round(dev_las, 2),
            "test_uas": round(test_uas, 2),
            "test_las": round(test_las, 2),
            "training_time_seconds": training_time,
            "parsing_time_seconds": parsing_time,
            "embedding": config["embedding"],
            "pos_vocab_size": len(pos_vocab),
            "label_vocab_size": len(label_vocab),
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
    write_m2b_readme(results, config, experiment_dir)

    print()
    print("Final M2b results (best-dev checkpoint)")
    print(f"  Projective training sentences kept: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"  Generated transition examples: {training_summary['generated_transition_examples']}")
    print(f"  Selected epoch: {best_checkpoint['epoch']}")
    print(f"  Dev UAS/LAS:  {dev_uas:.2f} / {dev_las:.2f}")
    print(f"  Test UAS/LAS: {test_uas:.2f} / {test_las:.2f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Parsing time:  {parsing_time:.2f}s")


def write_m2b_readme(results, config, experiment_dir):
    embedding_path = config["embedding"].get("path")
    lines = [
        "# M2b Arc-Standard Dense-Rich Experiment",
        "",
        "## Goal",
        "",
        "Add richer dense syntactic features inspired by Chen and Manning (2014) while keeping",
        "Arc-Standard, the static oracle, greedy decoding, and the projective training subset fixed.",
        "",
        "## Feature template",
        "",
        "- Word embeddings for s0, s1, s2, b0, b1, b2, s0-left, s0-right, s1-left, s1-right",
        "- POS embeddings for the same token positions",
        "- Dependency label embeddings for the four child relations",
        "- Distance bucket between s0 and b0",
        "- Stack-size bucket",
        "- Buffer-size bucket",
        "- NULL and ROOT indicator flags",
        "",
        f"- Embedding source: `{config['embedding']['source']}`",
        f"- Embedding dimension: `{config['embedding']['embedding_dim']}`",
        f"- Embedding path: `{embedding_path}`",
        f"- POS embedding dimension: `{config['classifier']['pos_embedding_dim']}`",
        f"- Label embedding dimension: `{config['classifier']['label_embedding_dim']}`",
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
    ]
    (experiment_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _torch_cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
