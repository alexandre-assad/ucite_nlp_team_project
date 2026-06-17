"""Train and document the M0p Arc-Eager projective sparse baseline experiment."""

import argparse
import json
import pickle
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from baseline_analysis import count_tokens
from component_registry import (
    resolve_decoder,
    resolve_feature_extractor,
    resolve_oracle,
    resolve_transition_system,
)
from data_utils import get_labels, read_conllu, write_conllu
from evaluator import evaluate
from experiment_utils import Tee, build_results, generate_plots, write_json
from parser import parse_all, train
from perceptron import AveragedPerceptron
from run_m1_experiment import is_projective


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "m0p_arc_eager_static_sparse_projective"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run the M0p Arc-Eager projective sparse experiment"
    )
    arg_parser.parse_args()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": "M0p",
        "description": (
            "Arc-Eager + Static Oracle + Sparse Handcrafted Features + Averaged Perceptron + "
            "Greedy Decoding + Projective Training Filter"
        ),
        "transition_system": "arc-eager",
        "oracle": "static",
        "features": "sparse handcrafted",
        "classifier": "averaged perceptron",
        "decoder": "greedy",
        "epochs": 10,
        "seed": 0,
        "train_path": "data/fr_gsd-ud-train.conllu",
        "dev_path": "data/fr_gsd-ud-dev.conllu",
        "test_path": "data/fr_gsd-ud-test.conllu",
        "projective_filter": True,
        "output_dir": "experiments/m0p_arc_eager_static_sparse_projective",
    }
    write_json(CONFIG_PATH, config)

    log_path = EXPERIMENT_DIR / "training_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            run_experiment(config)


def run_experiment(config):
    print("=" * 80)
    print("Running M0p Arc-Eager projective sparse baseline experiment")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print()

    raw_train_sentences = read_conllu(str(PROJECT_ROOT / config["train_path"]))
    dev_sentences = read_conllu(str(PROJECT_ROOT / config["dev_path"]))
    test_sentences = read_conllu(str(PROJECT_ROOT / config["test_path"]))

    labels = get_labels(raw_train_sentences)
    train_sentences = [sentence for sentence in raw_train_sentences if is_projective(sentence)]

    transition_system = resolve_transition_system(config["transition_system"])
    oracle = resolve_oracle(config["oracle"], config["transition_system"])
    feature_extractor = resolve_feature_extractor(config["features"])
    decoder = resolve_decoder(config["decoder"])

    print(f"Train sentences kept for M0p: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"Dev sentences:   {len(dev_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Train tokens:    {count_tokens(train_sentences)}")
    print(f"Dev tokens:      {count_tokens(dev_sentences)}")
    print(f"Test tokens:     {count_tokens(test_sentences)}")
    print(f"Labels:          {len(labels)}")
    print()

    model = AveragedPerceptron()
    checkpoint_dir = EXPERIMENT_DIR / "checkpoints"

    training_start = time.time()
    training_summary = train(
        train_sentences,
        dev_sentences,
        model,
        labels,
        n_epochs=config["epochs"],
        seed=config["seed"],
        checkpoint_dir=str(checkpoint_dir),
        transition_system=transition_system,
        oracle=oracle,
        feature_extractor=feature_extractor,
    )
    training_time = time.time() - training_start

    model_final_path = EXPERIMENT_DIR / "model_final.pkl"
    model.save(str(model_final_path))
    with (EXPERIMENT_DIR / "model_labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)

    best_checkpoint = training_summary.get("best_dev_checkpoint")
    if best_checkpoint is None:
        raise RuntimeError("No best-dev checkpoint was saved for M0p")
    shutil.copyfile(best_checkpoint["model_path"], EXPERIMENT_DIR / "model_best_dev.pkl")

    best_model = AveragedPerceptron()
    best_model.load(str(EXPERIMENT_DIR / "model_best_dev.pkl"))

    dev_predictions, _ = parse_all(
        dev_sentences,
        best_model,
        labels,
        return_details=True,
        transition_system=transition_system,
        feature_extractor=feature_extractor,
        decoder=decoder,
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
        decoder=decoder,
    )
    parsing_time = time.time() - parse_start
    test_uas, test_las = evaluate(test_sentences, test_predictions)
    write_conllu(str(EXPERIMENT_DIR / "predictions.conllu"), test_predictions)

    malformed_predictions, malformed_parse_details = parse_all(
        test_sentences,
        best_model,
        labels,
        return_details=True,
        repair=False,
        transition_system=transition_system,
        feature_extractor=feature_extractor,
        decoder=decoder,
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
            "model_id": "M0p",
            "description": config["description"],
            "transition_system": config["transition_system"],
            "oracle": config["oracle"],
            "features": config["features"],
            "classifier": config["classifier"],
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
            "projective_filter": True,
            "selection_metric": {
                "primary": "dev_las",
                "tie_breaker": "dev_uas",
            },
        }
    )
    write_json(EXPERIMENT_DIR / "results.json", results)

    generate_plots(results, training_summary, FIGURES_DIR, config["model_id"])
    write_readme(results, config, best_checkpoint)

    print()
    print("Final M0p results (best-dev checkpoint)")
    print(f"  Selected epoch: {best_checkpoint['epoch']}")
    print(f"  Dev UAS/LAS:  {dev_uas:.2f} / {dev_las:.2f}")
    print(f"  Test UAS/LAS: {test_uas:.2f} / {test_las:.2f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Parsing time:  {parsing_time:.2f}s")


def write_readme(results, config, best_checkpoint):
    lines = [
        "# M0p Arc-Eager Projective Sparse Experiment",
        "",
        "## Goal",
        "",
        "Train Arc-Eager under the same projective training condition used by M1/M2/M2b/M3,",
        "while keeping the sparse perceptron parser architecture unchanged.",
        "",
        "## M0p definition",
        "",
        "- Arc-Eager transition system",
        "- Static Arc-Eager oracle",
        "- Sparse handcrafted features",
        "- Averaged perceptron classifier",
        "- Greedy decoding",
        "- Projective training filter",
        "",
        "## Projective filtering",
        "",
        f"- Training sentences kept: `{results['train_sentences_kept']} / {results['train_sentences_original']}`",
        "- Dev and test sets are not filtered",
        "",
        "## Checkpoint selection",
        "",
        "Best-dev checkpointing is enabled for M0p.",
        f"The selected checkpoint is epoch `{best_checkpoint['epoch']}`, chosen by highest dev LAS with dev UAS as the tie-breaker.",
        "",
        "## Final results",
        "",
        f"- Dev UAS/LAS: `{results['dev']['uas']:.2f}` / `{results['dev']['las']:.2f}`",
        f"- Test UAS/LAS: `{results['test']['uas']:.2f}` / `{results['test']['las']:.2f}`",
        "",
        "## Outputs",
        "",
        "- `config.json`",
        "- `results.json`",
        "- `training_log.txt`",
        "- `README.md`",
        "- `model_best_dev.pkl`",
        "- `model_final.pkl`",
        "- `model_labels.pkl`",
        "- `predictions.conllu`",
        "- `figures/`",
    ]
    (EXPERIMENT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
