"""Train and document the M1 Arc-Standard baseline experiment."""

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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "m1_arc_standard_static_sparse"
DEFAULT_CONFIG_PATH = DEFAULT_EXPERIMENT_DIR / "config.json"


def main():
    arg_parser = argparse.ArgumentParser(description="Run the M1 Arc-Standard baseline experiment")
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
    run_name = "M1 Arc-Standard baseline experiment"
    print("=" * 80)
    print(f"Running {run_name}")
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

    print(f"Train sentences kept for M1: {len(train_sentences)} / {len(raw_train_sentences)}")
    print(f"Dev sentences:   {len(dev_sentences)}")
    print(f"Test sentences:  {len(test_sentences)}")
    print(f"Train tokens:    {count_tokens(train_sentences)}")
    print(f"Dev tokens:      {count_tokens(dev_sentences)}")
    print(f"Test tokens:     {count_tokens(test_sentences)}")
    print(f"Labels:          {len(labels)}")
    print()

    model = AveragedPerceptron()
    checkpoint_dir = experiment_dir / "checkpoints"

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

    model_final_path = experiment_dir / "model_final.pkl"
    model.save(str(model_final_path))
    with (experiment_dir / "model_labels.pkl").open("wb") as handle:
        pickle.dump(labels, handle)

    best_checkpoint = training_summary.get("best_dev_checkpoint")
    if best_checkpoint is None:
        raise RuntimeError("No best-dev checkpoint was saved for M1")
    shutil.copyfile(best_checkpoint["model_path"], experiment_dir / "model_best_dev.pkl")

    best_model = AveragedPerceptron()
    best_model.load(str(experiment_dir / "model_best_dev.pkl"))

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
    write_conllu(str(experiment_dir / "predictions.conllu"), test_predictions)

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
    results["selected_epoch"] = best_checkpoint["epoch"]
    results["selection_metric"] = {
        "primary": "dev_las",
        "tie_breaker": "dev_uas",
    }
    write_json(experiment_dir / "results.json", results)

    generate_plots(results, training_summary, figures_dir, config["model_id"])
    write_m1_readme(results, config, best_checkpoint, experiment_dir)

    print()
    print("Final M1 results (best-dev checkpoint)")
    print(f"  Selected epoch: {best_checkpoint['epoch']}")
    print(f"  Dev UAS/LAS:  {dev_uas:.2f} / {dev_las:.2f}")
    print(f"  Test UAS/LAS: {test_uas:.2f} / {test_las:.2f}")


def write_m1_readme(results, config, best_checkpoint, experiment_dir):
    run_cmd = "python run_m1_experiment.py"
    model_final_name = "model_final.pkl"
    model_best_name = "model_best_dev.pkl"

    lines = [
        "# M1 Arc-Standard Baseline Experiment",
        "",
        "## Goal",
        "",
        "Provide a clean comparison against M0 by changing only the transition system and oracle.",
        "",
        "Because Arc-Standard with a static oracle is projective, training uses the projective subset",
        "of the selected training data drawn from the same train file.",
        "",
        "M1 definition:",
        "",
        "- Arc-Standard transition system",
        "- Static oracle",
        "- Sparse handcrafted features",
        "- Averaged perceptron classifier",
        "- Greedy decoding",
    ]
    lines.extend([
        "",
        "## Run",
        "",
        "```bash",
        run_cmd,
        "```",
        "",
        "## Checkpoint selection",
        "",
        "Best-dev checkpointing is enabled from the beginning for M1.",
        f"The selected checkpoint is epoch `{best_checkpoint['epoch']}`, chosen by highest dev LAS with dev UAS as the tie-breaker.",
        "",
        "## Outputs",
        "",
        f"- `{model_final_name}`: final averaged model for this run",
        f"- `{model_best_name}`: checkpoint selected by dev LAS, with dev UAS tie-break",
        "- `model_labels.pkl`: dependency label vocabulary",
        "- `predictions.conllu`: predicted test trees from the best-dev model",
        "- `results.json`: evaluation and analysis summary",
        "- `training_log.txt`: full training/evaluation log",
        "- `figures/`: plots and analysis figures",
        "",
        "## Latest available results",
        "",
        f"- Dev UAS/LAS: `{results['dev']['uas']:.2f}` / `{results['dev']['las']:.2f}`",
        f"- Test UAS/LAS: `{results['test']['uas']:.2f}` / `{results['test']['las']:.2f}`",
    ])
    (experiment_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def is_projective(sentence):
    """Return True if the gold dependency tree is projective."""
    arcs = []
    for token in sentence[1:]:
        head = token["head"]
        dep = token["id"]
        left, right = sorted((head, dep))
        arcs.append((left, right))

    for i, (a_left, a_right) in enumerate(arcs):
        for b_left, b_right in arcs[i + 1:]:
            if a_left < b_left < a_right < b_right:
                return False
            if b_left < a_left < b_right < a_right:
                return False
    return True


if __name__ == "__main__":
    main()
