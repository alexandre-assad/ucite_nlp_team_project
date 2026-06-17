# Error Analysis

## How to run

```bash
python analyze_errors.py
```

## Input files

- `data/fr_gsd-ud-test.conllu`
- `experiments/m0_arc_eager_static_sparse/predictions.conllu`
- `experiments/m0p_arc_eager_static_sparse_projective/predictions.conllu`
- `experiments/m1_arc_standard_static_sparse/predictions.conllu`
- `experiments/m2_arc_standard_static_dense/predictions.conllu`
- `experiments/m2b_arc_standard_dense_rich/predictions.conllu`
- `experiments/m3_arc_standard_contextual_rich/predictions.conllu`

## Validation

Before analysis, the script checks that every prediction file matches the gold test file in:
- number of sentences
- number of tokens per sentence
- token forms

The loader ignores CoNLL-U comments, multiword tokens, and empty nodes.

## Recomputed metrics

UAS and LAS are recomputed directly from the prediction files by comparing each non-ROOT token
against the gold test file. The script also checks whether the recomputed test scores match the
stored values in each `results.json` file.

## Generated outputs

- `overall_scores.csv` and `overall_scores.md`
- `transition_system_error_analysis.md`
- `label_error_table.csv` and `label_error_table.md`
- `label_gains_m1_vs_m0p.csv`
- `label_gains_m2b_vs_m2.csv`
- `label_gains_m3_vs_m2b.csv`
- `length_bucket_table.csv`
- `distance_bucket_table.csv`
- `root_analysis.csv` and `root_analysis.md`
- `m0p_vs_m1_examples.md`
- `m2_vs_m2b_examples.md`
- `m2b_vs_m3_examples.md`
- `label_confusions.csv` and `label_confusions.md`
- `figures/`
- `error_analysis_summary.md`
