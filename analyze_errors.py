"""Cross-model error analysis over frozen parser outputs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from data_utils import read_conllu
from evaluator import evaluate


PROJECT_ROOT = Path(__file__).resolve().parent
GOLD_PATH = PROJECT_ROOT / "data" / "fr_gsd-ud-test.conllu"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "error_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODELS = {
    "M0": {
        "label": "M0",
        "predictions": PROJECT_ROOT / "experiments" / "m0_arc_eager_static_sparse" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m0_arc_eager_static_sparse" / "results.json",
        "description": "Arc-Eager, full training set, sparse perceptron",
    },
    "M0p": {
        "label": "M0p",
        "predictions": PROJECT_ROOT / "experiments" / "m0p_arc_eager_static_sparse_projective" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m0p_arc_eager_static_sparse_projective" / "results.json",
        "description": "Arc-Eager, projective subset, sparse perceptron",
    },
    "M1": {
        "label": "M1",
        "predictions": PROJECT_ROOT / "experiments" / "m1_arc_standard_static_sparse" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m1_arc_standard_static_sparse" / "results.json",
        "description": "Arc-Standard, projective subset, sparse perceptron",
    },
    "M2": {
        "label": "M2",
        "predictions": PROJECT_ROOT / "experiments" / "m2_arc_standard_static_dense" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m2_arc_standard_static_dense" / "results.json",
        "description": "Arc-Standard, minimal FastText dense embeddings + MLP",
    },
    "M2b": {
        "label": "M2b",
        "predictions": PROJECT_ROOT / "experiments" / "m2b_arc_standard_dense_rich" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m2b_arc_standard_dense_rich" / "results.json",
        "description": "Arc-Standard, FastText rich dense features + MLP",
    },
    "M3": {
        "label": "M3",
        "predictions": PROJECT_ROOT / "experiments" / "m3_arc_standard_contextual_rich" / "predictions.conllu",
        "results": PROJECT_ROOT / "experiments" / "m3_arc_standard_contextual_rich" / "results.json",
        "description": "Arc-Standard, CamemBERT contextual rich features + MLP",
    },
}

MODEL_ORDER = ["M0", "M0p", "M1", "M2", "M2b", "M3"]
LENGTH_BUCKETS = ["1-10", "11-20", "21-30", "31-40", "41+"]
DISTANCE_BUCKETS = ["ROOT", "1", "2-3", "4-7", "8+"]


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run cross-model error analysis over frozen parser outputs"
    )
    arg_parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gold_sentences = read_conllu(str(GOLD_PATH))
    model_data = load_all_models(gold_sentences)

    overall_rows = build_overall_scores(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "overall_scores.csv", overall_rows)
    write_overall_markdown(OUTPUT_DIR / "overall_scores.md", overall_rows)

    label_rows, label_metrics_by_model = build_label_error_rows(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "label_error_table.csv", label_rows)
    write_label_error_markdown(OUTPUT_DIR / "label_error_table.md", label_rows)

    write_csv(
        OUTPUT_DIR / "label_gains_m1_vs_m0p.csv",
        build_label_gain_rows(label_metrics_by_model, "M0p", "M1"),
    )
    write_csv(
        OUTPUT_DIR / "label_gains_m2b_vs_m2.csv",
        build_label_gain_rows(label_metrics_by_model, "M2", "M2b"),
    )
    write_csv(
        OUTPUT_DIR / "label_gains_m3_vs_m2b.csv",
        build_label_gain_rows(label_metrics_by_model, "M2b", "M3"),
    )

    length_rows = build_length_bucket_rows(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "length_bucket_table.csv", length_rows)

    distance_rows = build_distance_bucket_rows(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "distance_bucket_table.csv", distance_rows)

    root_rows, root_examples = build_root_analysis_rows(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "root_analysis.csv", root_rows)
    write_root_markdown(OUTPUT_DIR / "root_analysis.md", root_rows, root_examples)

    confusion_rows = build_label_confusion_rows(model_data, gold_sentences)
    write_csv(OUTPUT_DIR / "label_confusions.csv", confusion_rows)
    write_confusion_markdown(OUTPUT_DIR / "label_confusions.md", confusion_rows)

    write_pair_examples(
        OUTPUT_DIR / "m0p_vs_m1_examples.md",
        collect_pair_examples(gold_sentences, model_data["M0p"]["sentences"], model_data["M1"]["sentences"], "M0p", "M1"),
        "M0p",
        "M1",
    )
    write_pair_examples(
        OUTPUT_DIR / "m2_vs_m2b_examples.md",
        collect_pair_examples(gold_sentences, model_data["M2"]["sentences"], model_data["M2b"]["sentences"], "M2", "M2b"),
        "M2",
        "M2b",
    )
    write_pair_examples(
        OUTPUT_DIR / "m2b_vs_m3_examples.md",
        collect_pair_examples(gold_sentences, model_data["M2b"]["sentences"], model_data["M3"]["sentences"], "M2b", "M3"),
        "M2b",
        "M3",
    )

    write_transition_system_analysis(OUTPUT_DIR / "transition_system_error_analysis.md", overall_rows)
    generate_figures(overall_rows, label_metrics_by_model, length_rows, distance_rows, root_rows)
    write_summary(
        OUTPUT_DIR / "error_analysis_summary.md",
        overall_rows,
        label_metrics_by_model,
        length_rows,
        distance_rows,
        root_rows,
    )
    write_readme(OUTPUT_DIR / "README.md")

    print("Validation passed for all prediction files.")
    mismatch_rows = [row for row in overall_rows if row["warning"]]
    if mismatch_rows:
        print("Score mismatches detected between recomputed values and stored results.")
    else:
        print("Recomputed scores match stored results.json values within tolerance.")
    print("Generated files in experiments/error_analysis/:")
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    print("Most important findings:")
    print("  - Projective filtering improves Arc-Eager, but Arc-Standard still leads under matched conditions.")
    print("  - Minimal dense embeddings underperform strongly relative to sparse and richer dense models.")
    print("  - Rich dense syntax restores large gains, and contextual embeddings provide the strongest overall improvement.")


def load_all_models(gold_sentences):
    model_data = {}
    for model_id in MODEL_ORDER:
        spec = MODELS[model_id]
        predicted_sentences = read_conllu(str(spec["predictions"]))
        validate_predictions(gold_sentences, predicted_sentences, model_id)
        results = json.loads(spec["results"].read_text(encoding="utf-8"))
        model_data[model_id] = {
            "sentences": predicted_sentences,
            "results": results,
            "description": spec["description"],
        }
    return model_data


def validate_predictions(gold_sentences, predicted_sentences, model_id):
    if len(gold_sentences) != len(predicted_sentences):
        raise ValueError(f"{model_id}: sentence count mismatch ({len(predicted_sentences)} vs {len(gold_sentences)})")
    for sent_idx, (gold_sentence, pred_sentence) in enumerate(zip(gold_sentences, predicted_sentences), start=1):
        if len(gold_sentence) != len(pred_sentence):
            raise ValueError(
                f"{model_id}: token count mismatch in sentence {sent_idx} "
                f"({len(pred_sentence) - 1} vs {len(gold_sentence) - 1})"
            )
        for tok_idx, (gold_tok, pred_tok) in enumerate(zip(gold_sentence[1:], pred_sentence[1:]), start=1):
            if gold_tok["form"] != pred_tok["form"]:
                raise ValueError(
                    f"{model_id}: token form mismatch in sentence {sent_idx}, token {tok_idx}: "
                    f"{pred_tok['form']} vs {gold_tok['form']}"
                )


def build_overall_scores(model_data, gold_sentences):
    rows = []
    for model_id in MODEL_ORDER:
        predicted = model_data[model_id]["sentences"]
        results = model_data[model_id]["results"]
        recomputed_uas, recomputed_las = evaluate(gold_sentences, predicted)
        stored_uas = float(results["test"]["uas"])
        stored_las = float(results["test"]["las"])
        warning = ""
        if abs(recomputed_uas - stored_uas) > 0.01 or abs(recomputed_las - stored_las) > 0.01:
            warning = "stored results.json differs from recomputed score"
        rows.append(
            {
                "model_id": model_id,
                "description": model_data[model_id]["description"],
                "stored_test_uas": round(stored_uas, 2),
                "stored_test_las": round(stored_las, 2),
                "recomputed_test_uas": round(recomputed_uas, 2),
                "recomputed_test_las": round(recomputed_las, 2),
                "warning": warning,
            }
        )
    return rows


def build_label_error_rows(model_data, gold_sentences):
    rows = []
    metrics_by_model = {}
    for model_id in MODEL_ORDER:
        totals = Counter()
        correct_head = Counter()
        correct_both = Counter()
        for gold_sentence, pred_sentence in zip(gold_sentences, model_data[model_id]["sentences"]):
            for gold_tok, pred_tok in zip(gold_sentence[1:], pred_sentence[1:]):
                label = gold_tok["deprel"]
                totals[label] += 1
                if pred_tok["head"] == gold_tok["head"]:
                    correct_head[label] += 1
                    if pred_tok["deprel"] == gold_tok["deprel"]:
                        correct_both[label] += 1

        metrics_by_model[model_id] = {}
        for label, support in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
            uas = pct(correct_head[label], support)
            las = pct(correct_both[label], support)
            head_errors = support - correct_head[label]
            label_errors = correct_head[label] - correct_both[label]
            row = {
                "model_id": model_id,
                "label": label,
                "gold_count": support,
                "uas": round(uas, 2),
                "las": round(las, 2),
                "head_errors": head_errors,
                "label_errors": label_errors,
            }
            rows.append(row)
            metrics_by_model[model_id][label] = row
    return rows, metrics_by_model


def build_label_gain_rows(metrics_by_model, base_model, improved_model, min_support=20):
    rows = []
    labels = set(metrics_by_model[base_model]) & set(metrics_by_model[improved_model])
    for label in labels:
        base = metrics_by_model[base_model][label]
        improved = metrics_by_model[improved_model][label]
        if base["gold_count"] < min_support:
            continue
        rows.append(
            {
                "label": label,
                "gold_count": base["gold_count"],
                f"{base_model}_las": base["las"],
                f"{improved_model}_las": improved["las"],
                "las_gain": round(improved["las"] - base["las"], 2),
                f"{base_model}_uas": base["uas"],
                f"{improved_model}_uas": improved["uas"],
                "uas_gain": round(improved["uas"] - base["uas"], 2),
            }
        )
    return sorted(rows, key=lambda row: (-row["las_gain"], -row["gold_count"], row["label"]))


def build_length_bucket_rows(model_data, gold_sentences):
    rows = []
    for model_id in MODEL_ORDER:
        stats = defaultdict(lambda: {"correct_head": 0, "correct_both": 0, "tokens": 0, "sentences": 0})
        for gold_sentence, pred_sentence in zip(gold_sentences, model_data[model_id]["sentences"]):
            bucket = sentence_length_bucket(len(gold_sentence) - 1)
            stats[bucket]["sentences"] += 1
            for gold_tok, pred_tok in zip(gold_sentence[1:], pred_sentence[1:]):
                stats[bucket]["tokens"] += 1
                if pred_tok["head"] == gold_tok["head"]:
                    stats[bucket]["correct_head"] += 1
                    if pred_tok["deprel"] == gold_tok["deprel"]:
                        stats[bucket]["correct_both"] += 1
        for bucket in LENGTH_BUCKETS:
            bucket_stats = stats[bucket]
            rows.append(
                {
                    "model_id": model_id,
                    "bucket": bucket,
                    "sentences": bucket_stats["sentences"],
                    "tokens": bucket_stats["tokens"],
                    "uas": round(pct(bucket_stats["correct_head"], bucket_stats["tokens"]), 2),
                    "las": round(pct(bucket_stats["correct_both"], bucket_stats["tokens"]), 2),
                }
            )
    return rows


def build_distance_bucket_rows(model_data, gold_sentences):
    rows = []
    for model_id in MODEL_ORDER:
        stats = defaultdict(lambda: {"correct_head": 0, "correct_both": 0, "tokens": 0})
        for gold_sentence, pred_sentence in zip(gold_sentences, model_data[model_id]["sentences"]):
            for gold_tok, pred_tok in zip(gold_sentence[1:], pred_sentence[1:]):
                bucket = dependency_distance_bucket(gold_tok["head"], gold_tok["id"])
                stats[bucket]["tokens"] += 1
                if pred_tok["head"] == gold_tok["head"]:
                    stats[bucket]["correct_head"] += 1
                    if pred_tok["deprel"] == gold_tok["deprel"]:
                        stats[bucket]["correct_both"] += 1
        for bucket in DISTANCE_BUCKETS:
            bucket_stats = stats[bucket]
            rows.append(
                {
                    "model_id": model_id,
                    "bucket": bucket,
                    "tokens": bucket_stats["tokens"],
                    "uas": round(pct(bucket_stats["correct_head"], bucket_stats["tokens"]), 2),
                    "las": round(pct(bucket_stats["correct_both"], bucket_stats["tokens"]), 2),
                }
            )
    return rows


def build_root_analysis_rows(model_data, gold_sentences):
    rows = []
    examples = {}
    for model_id in MODEL_ORDER:
        predicted = model_data[model_id]["sentences"]
        gold_root_count = len(gold_sentences)
        predicted_root_count = 0
        correct_root_count = 0
        wrong_examples = []

        for sent_idx, (gold_sentence, pred_sentence) in enumerate(zip(gold_sentences, predicted), start=1):
            gold_root_id = get_sentence_root_id(gold_sentence)
            predicted_root_ids = [tok["id"] for tok in pred_sentence[1:] if tok["head"] == 0]
            predicted_root_count += len(predicted_root_ids)
            if gold_root_id in predicted_root_ids:
                correct_root_count += 1
            if predicted_root_ids != [gold_root_id]:
                wrong_examples.append(
                    {
                        "sentence_id": sent_idx,
                        "sentence_text": sentence_text(gold_sentence),
                        "gold_root": gold_root_id,
                        "predicted_roots": predicted_root_ids,
                    }
                )

        precision = safe_div(correct_root_count, predicted_root_count)
        recall = safe_div(correct_root_count, gold_root_count)
        f1 = f1_score(precision, recall)
        rows.append(
            {
                "model_id": model_id,
                "gold_root_count": gold_root_count,
                "predicted_root_count": predicted_root_count,
                "correct_root_count": correct_root_count,
                "root_precision": round(precision * 100, 2),
                "root_recall": round(recall * 100, 2),
                "root_f1": round(f1 * 100, 2),
                "wrong_root_attachments": predicted_root_count - correct_root_count,
            }
        )
        examples[model_id] = wrong_examples[:5]
    return rows, examples


def build_label_confusion_rows(model_data, gold_sentences):
    rows = []
    for model_id in MODEL_ORDER:
        confusion = Counter()
        for gold_sentence, pred_sentence in zip(gold_sentences, model_data[model_id]["sentences"]):
            for gold_tok, pred_tok in zip(gold_sentence[1:], pred_sentence[1:]):
                if pred_tok["head"] == gold_tok["head"] and pred_tok["deprel"] != gold_tok["deprel"]:
                    confusion[(gold_tok["deprel"], pred_tok["deprel"])] += 1
        for (gold_label, pred_label), count in confusion.most_common(30):
            rows.append(
                {
                    "model_id": model_id,
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "count": count,
                }
            )
    return rows


def collect_pair_examples(gold_sentences, model_a_sentences, model_b_sentences, model_a, model_b, limit=20):
    groups = {
        "b_correct_a_wrong": [],
        "a_correct_b_wrong": [],
        "both_wrong_different": [],
    }
    for sent_idx, (gold_sentence, a_sentence, b_sentence) in enumerate(
        zip(gold_sentences, model_a_sentences, model_b_sentences), start=1
    ):
        for gold_tok, a_tok, b_tok in zip(gold_sentence[1:], a_sentence[1:], b_sentence[1:]):
            a_correct = a_tok["head"] == gold_tok["head"] and a_tok["deprel"] == gold_tok["deprel"]
            b_correct = b_tok["head"] == gold_tok["head"] and b_tok["deprel"] == gold_tok["deprel"]
            if a_correct == b_correct and a_correct:
                continue
            example = {
                "sentence_id": sent_idx,
                "sentence_text": sentence_text(gold_sentence),
                "token_id": gold_tok["id"],
                "token_form": gold_tok["form"],
                "gold_head": gold_tok["head"],
                "gold_label": gold_tok["deprel"],
                f"{model_a}_head": a_tok["head"],
                f"{model_a}_label": a_tok["deprel"],
                f"{model_b}_head": b_tok["head"],
                f"{model_b}_label": b_tok["deprel"],
                "note": classify_error(gold_sentence, gold_tok, a_tok, b_tok),
            }
            if b_correct and not a_correct:
                groups["b_correct_a_wrong"].append(example)
            elif a_correct and not b_correct:
                groups["a_correct_b_wrong"].append(example)
            elif not a_correct and not b_correct and (a_tok["head"], a_tok["deprel"]) != (b_tok["head"], b_tok["deprel"]):
                groups["both_wrong_different"].append(example)

    return {
        "b_correct_a_wrong": groups["b_correct_a_wrong"][:7],
        "a_correct_b_wrong": groups["a_correct_b_wrong"][:7],
        "both_wrong_different": groups["both_wrong_different"][:6],
    }


def classify_error(gold_sentence, gold_tok, a_tok, b_tok):
    notes = []
    distance = dependency_distance_bucket(gold_tok["head"], gold_tok["id"])
    if gold_tok["deprel"] == "root" or a_tok["head"] == 0 or b_tok["head"] == 0:
        notes.append("wrong ROOT")
    if distance in {"4-7", "8+"}:
        notes.append("long-distance")
    if gold_tok["deprel"] in {"conj", "cc"}:
        notes.append("coordination")
    if gold_tok["deprel"] in {"advcl", "ccomp", "xcomp", "acl", "acl:relcl"}:
        notes.append("clausal dependency")
    if gold_tok["upos"] in {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PART", "PRON"}:
        notes.append("function word")
    if a_tok["head"] != gold_tok["head"] or b_tok["head"] != gold_tok["head"]:
        notes.append("wrong head")
    if a_tok["deprel"] != gold_tok["deprel"] or b_tok["deprel"] != gold_tok["deprel"]:
        notes.append("wrong label")
    deduped = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return ", ".join(deduped) if deduped else "mixed error"


def write_overall_markdown(path, rows):
    lines = [
        "# Overall Scores",
        "",
        "| Model | Description | Stored Test UAS | Stored Test LAS | Recomputed Test UAS | Recomputed Test LAS | Warning |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['description']} | {row['stored_test_uas']:.2f} | "
            f"{row['stored_test_las']:.2f} | {row['recomputed_test_uas']:.2f} | "
            f"{row['recomputed_test_las']:.2f} | {row['warning'] or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_label_error_markdown(path, rows):
    lines = [
        "# Label Error Table",
        "",
        "For each model and gold dependency label, `head_errors` counts tokens with a wrong head,",
        "while `label_errors` counts tokens where the head is correct but the dependency label is wrong.",
        "",
        "| Model | Label | Gold count | UAS | LAS | Head errors | Label errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['label']} | {row['gold_count']} | {row['uas']:.2f} | "
            f"{row['las']:.2f} | {row['head_errors']} | {row['label_errors']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_markdown(path, rows, examples):
    lines = [
        "# Root Analysis",
        "",
        "| Model | Gold roots | Predicted roots | Correct roots | Precision | Recall | F1 | Wrong root attachments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['gold_root_count']} | {row['predicted_root_count']} | "
            f"{row['correct_root_count']} | {row['root_precision']:.2f} | {row['root_recall']:.2f} | "
            f"{row['root_f1']:.2f} | {row['wrong_root_attachments']} |"
        )
    lines.extend(["", "## Wrong ROOT examples", ""])
    for model_id in MODEL_ORDER:
        lines.append(f"### {model_id}")
        model_examples = examples[model_id]
        if not model_examples:
            lines.append("")
            lines.append("No wrong ROOT examples found.")
            lines.append("")
            continue
        for example in model_examples:
            lines.append(
                f"- Sentence {example['sentence_id']}: {example['sentence_text']} | "
                f"gold ROOT={example['gold_root']}, predicted ROOTs={example['predicted_roots']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_markdown(path, rows):
    lines = [
        "# Label Confusions",
        "",
        "These counts only consider cases where the predicted head is correct but the dependency label is wrong.",
        "",
        "| Model | Gold label | Predicted label | Count |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['gold_label']} | {row['predicted_label']} | {row['count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pair_examples(path, grouped_examples, model_a, model_b):
    lines = [
        f"# {model_a} vs {model_b} Examples",
        "",
        f"This file highlights representative tokens where {model_b} improves on {model_a},",
        f"where {model_a} is correct and {model_b} is wrong, and where both are wrong differently.",
        "",
    ]
    sections = [
        ("b_correct_a_wrong", f"{model_b} correct, {model_a} wrong"),
        ("a_correct_b_wrong", f"{model_a} correct, {model_b} wrong"),
        ("both_wrong_different", "Both wrong differently"),
    ]
    for key, title in sections:
        lines.append(f"## {title}")
        lines.append("")
        examples = grouped_examples[key]
        if not examples:
            lines.append("No examples found.")
            lines.append("")
            continue
        for example in examples:
            lines.append(f"### Sentence {example['sentence_id']} — {example['sentence_text']}")
            lines.append("")
            lines.append(f"- Token: `{example['token_form']}` (id={example['token_id']})")
            lines.append(f"- Gold: head={example['gold_head']}, label={example['gold_label']}")
            lines.append(
                f"- {model_a}: head={example[f'{model_a}_head']}, label={example[f'{model_a}_label']}"
            )
            lines.append(
                f"- {model_b}: head={example[f'{model_b}_head']}, label={example[f'{model_b}_label']}"
            )
            lines.append(f"- Automatic note: {example['note']}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transition_system_analysis(path, overall_rows):
    row_by_model = {row["model_id"]: row for row in overall_rows}
    m0 = row_by_model["M0"]
    m0p = row_by_model["M0p"]
    m1 = row_by_model["M1"]

    def gain(model_hi, model_lo, metric):
        return row_by_model[model_hi][metric] - row_by_model[model_lo][metric]

    lines = [
        "# Transition System Error Analysis",
        "",
        "## Controlled comparison motivation",
        "",
        "- M0 vs M1 was not perfectly controlled because M0 used the full training set and M1 used projective filtering.",
        "- M0p makes the comparison fairer by training Arc-Eager on the same projective subset used by M1.",
        "- The results suggest that projective filtering improves Arc-Eager.",
        "- The results also suggest that Arc-Standard still outperforms Arc-Eager under the same projective training condition.",
        "",
        "## Test-score differences",
        "",
        f"- M0p vs M0: `{gain('M0p', 'M0', 'recomputed_test_uas'):+.2f}` UAS, `{gain('M0p', 'M0', 'recomputed_test_las'):+.2f}` LAS",
        f"- M1 vs M0p: `{gain('M1', 'M0p', 'recomputed_test_uas'):+.2f}` UAS, `{gain('M1', 'M0p', 'recomputed_test_las'):+.2f}` LAS",
        f"- M1 vs M0: `{gain('M1', 'M0', 'recomputed_test_uas'):+.2f}` UAS, `{gain('M1', 'M0', 'recomputed_test_las'):+.2f}` LAS",
        "",
        "## Interpretation",
        "",
        "The results suggest that projective filtering alone helps Arc-Eager, but it does not close the gap to Arc-Standard.",
        "This supports using M0p vs M1 as the cleaner transition-system comparison for the report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_figures(overall_rows, label_metrics_by_model, length_rows, distance_rows, root_rows):
    plot_overall_metric(overall_rows, "recomputed_test_uas", "Test UAS", FIGURES_DIR / "test_uas_all_models.png")
    plot_overall_metric(overall_rows, "recomputed_test_las", "Test LAS", FIGURES_DIR / "test_las_all_models.png")
    plot_top_label_las(label_metrics_by_model, FIGURES_DIR / "las_by_label_top_labels.png")
    plot_bucket_metric(length_rows, LENGTH_BUCKETS, "uas", "UAS (%)", FIGURES_DIR / "uas_by_sentence_length.png")
    plot_bucket_metric(length_rows, LENGTH_BUCKETS, "las", "LAS (%)", FIGURES_DIR / "las_by_sentence_length.png")
    plot_bucket_metric(distance_rows, DISTANCE_BUCKETS, "uas", "UAS (%)", FIGURES_DIR / "uas_by_dependency_distance.png")
    plot_bucket_metric(distance_rows, DISTANCE_BUCKETS, "las", "LAS (%)", FIGURES_DIR / "las_by_dependency_distance.png")
    plot_root_f1(root_rows, FIGURES_DIR / "root_f1_by_model.png")
    plot_label_gain(build_label_gain_rows(label_metrics_by_model, "M0p", "M1"), "M1 vs M0p LAS gains", FIGURES_DIR / "m0p_vs_m1_label_las_gains.png")
    plot_label_gain(build_label_gain_rows(label_metrics_by_model, "M2b", "M3"), "M3 vs M2b LAS gains", FIGURES_DIR / "m3_vs_m2b_label_las_gains.png")


def plot_overall_metric(rows, metric_key, ylabel, output_path):
    labels = [row["model_id"] for row in rows]
    values = [row[metric_key] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel(ylabel)
    plt.title(ylabel + " Across Frozen Models")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_top_label_las(label_metrics_by_model, output_path, top_n=10):
    top_labels = [
        row["label"]
        for row in sorted(
            label_metrics_by_model["M3"].values(),
            key=lambda row: (-row["gold_count"], row["label"]),
        )[:top_n]
    ]
    x = range(len(top_labels))
    width = 0.13
    plt.figure(figsize=(12, 6))
    for idx, model_id in enumerate(MODEL_ORDER):
        values = [label_metrics_by_model[model_id][label]["las"] for label in top_labels]
        plt.bar([pos + (idx - 2.5) * width for pos in x], values, width=width, label=model_id)
    plt.xticks(list(x), top_labels, rotation=45, ha="right")
    plt.ylabel("LAS (%)")
    plt.title("LAS by Label for Top-Support Dependency Labels")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_bucket_metric(rows, ordered_buckets, metric_key, ylabel, output_path):
    plt.figure(figsize=(9, 5))
    for model_id in MODEL_ORDER:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        mapping = {row["bucket"]: row[metric_key] for row in model_rows}
        plt.plot(ordered_buckets, [mapping[bucket] for bucket in ordered_buckets], marker="o", label=model_id)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by Bucket")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_root_f1(rows, output_path):
    labels = [row["model_id"] for row in rows]
    values = [row["root_f1"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("ROOT F1 (%)")
    plt.title("ROOT F1 by Model")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_label_gain(rows, title, output_path, top_n=12):
    top_rows = rows[:top_n]
    labels = [row["label"] for row in top_rows]
    gains = [row["las_gain"] for row in top_rows]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, gains)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("LAS gain")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(path, overall_rows, label_metrics_by_model, length_rows, distance_rows, root_rows):
    def find_row(model_id):
        return next(row for row in overall_rows if row["model_id"] == model_id)

    m0 = find_row("M0")
    m0p = find_row("M0p")
    m1 = find_row("M1")
    m2 = find_row("M2")
    m2b = find_row("M2b")
    m3 = find_row("M3")
    top_m1_vs_m0p = build_label_gain_rows(label_metrics_by_model, "M0p", "M1")[:5]
    top_m3_vs_m2b = build_label_gain_rows(label_metrics_by_model, "M2b", "M3")[:5]

    lines = [
        "# Error Analysis Summary",
        "",
        "## Purpose",
        "",
        "This analysis compares the frozen parsing outputs of M0, M0p, M1, M2, M2b, and M3 to understand transition-system effects and representation effects without rerunning any model.",
        "",
        "## Overall comparison",
        "",
        f"- Test LAS rises from `{m0['recomputed_test_las']:.2f}` in M0 to `{m0p['recomputed_test_las']:.2f}` in M0p, suggesting that projective filtering helps Arc-Eager.",
        f"- Under the same projective training condition, M1 still improves over M0p by `{m1['recomputed_test_uas'] - m0p['recomputed_test_uas']:+.2f}` UAS and `{m1['recomputed_test_las'] - m0p['recomputed_test_las']:+.2f}` LAS.",
        f"- M2 drops sharply relative to M1, while M2b and M3 recover and then exceed the sparse baseline.",
        "",
        "## Arc-Eager vs Arc-Standard",
        "",
        "The results suggest that Arc-Standard remains stronger than Arc-Eager even after controlling for projective filtering with M0p.",
        "This is consistent with the hypothesis that Arc-Standard's bottom-up attachment order may avoid some premature Arc-Eager attachment or reduction mistakes.",
        "",
        "## Sparse vs dense vs contextual representations",
        "",
        "The results suggest that minimal dense lexical embeddings alone are not enough for this transition-based parser setting.",
        "M2 underperforms badly, which is consistent with the idea that replacing sparse syntactic cues with only word vectors removes too much explicit structure.",
        "M2b restores much of that structure with POS and dependency-label information and improves strongly over M2.",
        "M3 then adds a further large improvement, suggesting that contextualized embeddings inject useful sentence-level information into local parsing decisions.",
        "",
        "## Label-level patterns",
        "",
        "The largest M1-over-M0p label gains among labels with enough support are:",
    ]
    for row in top_m1_vs_m0p:
        lines.append(f"- `{row['label']}`: {row['las_gain']:+.2f} LAS")
    lines.extend(["", "The largest M3-over-M2b label gains among labels with enough support are:"])
    for row in top_m3_vs_m2b:
        lines.append(f"- `{row['label']}`: {row['las_gain']:+.2f} LAS")
    lines.extend([
        "",
        "These gains should be interpreted cautiously, especially for labels with lower support.",
        "",
        "## Sentence length and dependency distance",
        "",
        "The bucket tables show whether gains are uniform or concentrated in longer sentences and longer dependencies.",
        "The results suggest that contextual and richer dense representations help across buckets, but the exact pattern is mixed in some ranges and should not be overinterpreted without the full tables.",
        "",
        "## ROOT errors",
        "",
        "ROOT precision, recall, and F1 improve steadily across the model sequence, with M3 achieving the strongest ROOT behavior.",
        "",
        "## Concrete examples",
        "",
        "Representative examples are saved separately for M0p vs M1, M2 vs M2b, and M2b vs M3.",
        "They are intended to support the transition-system and representation comparisons with concrete token-level cases.",
        "",
        "## Conclusions",
        "",
        "The results suggest three main conclusions:",
        "- Projective filtering alone does not explain the M1 advantage; Arc-Standard still outperforms Arc-Eager under matched projective training conditions.",
        "- Minimal dense lexical embeddings underperform because they remove too many explicit syntactic cues.",
        "- Rich dense features and then contextual embeddings produce progressively stronger results, with M3 as the strongest frozen model.",
        "",
        "## Limitations",
        "",
        "- Error categories are automatically assigned and should be treated as heuristic labels.",
        "- Some dependency labels have low support and should not be overinterpreted.",
        "- The analysis is based on the frozen test predictions only and does not include retraining or resampling.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path):
    lines = [
        "# Error Analysis",
        "",
        "## How to run",
        "",
        "```bash",
        "python analyze_errors.py",
        "```",
        "",
        "## Input files",
        "",
        "- `data/fr_gsd-ud-test.conllu`",
        "- `experiments/m0_arc_eager_static_sparse/predictions.conllu`",
        "- `experiments/m0p_arc_eager_static_sparse_projective/predictions.conllu`",
        "- `experiments/m1_arc_standard_static_sparse/predictions.conllu`",
        "- `experiments/m2_arc_standard_static_dense/predictions.conllu`",
        "- `experiments/m2b_arc_standard_dense_rich/predictions.conllu`",
        "- `experiments/m3_arc_standard_contextual_rich/predictions.conllu`",
        "",
        "## Validation",
        "",
        "Before analysis, the script checks that every prediction file matches the gold test file in:",
        "- number of sentences",
        "- number of tokens per sentence",
        "- token forms",
        "",
        "The loader ignores CoNLL-U comments, multiword tokens, and empty nodes.",
        "",
        "## Recomputed metrics",
        "",
        "UAS and LAS are recomputed directly from the prediction files by comparing each non-ROOT token",
        "against the gold test file. The script also checks whether the recomputed test scores match the",
        "stored values in each `results.json` file.",
        "",
        "## Generated outputs",
        "",
        "- `overall_scores.csv` and `overall_scores.md`",
        "- `transition_system_error_analysis.md`",
        "- `label_error_table.csv` and `label_error_table.md`",
        "- `label_gains_m1_vs_m0p.csv`",
        "- `label_gains_m2b_vs_m2.csv`",
        "- `label_gains_m3_vs_m2b.csv`",
        "- `length_bucket_table.csv`",
        "- `distance_bucket_table.csv`",
        "- `root_analysis.csv` and `root_analysis.md`",
        "- `m0p_vs_m1_examples.md`",
        "- `m2_vs_m2b_examples.md`",
        "- `m2b_vs_m3_examples.md`",
        "- `label_confusions.csv` and `label_confusions.md`",
        "- `figures/`",
        "- `error_analysis_summary.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sentence_length_bucket(length):
    if length <= 10:
        return "1-10"
    if length <= 20:
        return "11-20"
    if length <= 30:
        return "21-30"
    if length <= 40:
        return "31-40"
    return "41+"


def dependency_distance_bucket(head, dep):
    if head == 0:
        return "ROOT"
    distance = abs(dep - head)
    if distance == 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 7:
        return "4-7"
    return "8+"


def get_sentence_root_id(sentence):
    roots = [token["id"] for token in sentence[1:] if token["head"] == 0]
    if len(roots) != 1:
        return None
    return roots[0]


def sentence_text(sentence):
    return " ".join(token["form"] for token in sentence[1:])


def pct(numerator, denominator):
    return (100.0 * numerator / denominator) if denominator else 0.0


def safe_div(numerator, denominator):
    return (numerator / denominator) if denominator else 0.0


def f1_score(precision, recall):
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main()
