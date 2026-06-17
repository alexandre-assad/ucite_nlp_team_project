# M1 Arc-Standard Baseline Experiment

## Goal

M1 is the Arc-Standard baseline used to compare against M0 while keeping the rest of the
pipeline fixed.

M1 uses:

- Arc-Standard transition system
- Static oracle
- Sparse handcrafted features
- Averaged perceptron classifier
- Greedy decoding
- Seed `0`
- `10` training epochs

Relative to M0, M1 differs only in the transition system and static oracle. The feature
representation, classifier, decoding strategy, random seed and number of epochs are the same.

## Training data and scope

Because Arc-Standard with a static oracle assumes projective derivations, training was filtered
to the projective subset of the original training corpus.

- Training sentences kept: `13,816 / 14,450`
- Training sentences filtered out: `634`
- Dev/test were **not** filtered

## Checkpoint selection

Best-dev checkpointing was used from the beginning for M1.

- Selection criterion: highest dev LAS
- Tie-breaker: dev UAS
- Selected checkpoint: epoch `10`

Both `model_final.pkl` and `model_best_dev.pkl` are saved. The frozen M1 baseline corresponds to
the selected best-dev checkpoint from epoch 10.

## Run

```bash
python run_m1_experiment.py
```

## Outputs

- `model_final.pkl`: final averaged model after epoch 10
- `model_best_dev.pkl`: checkpoint selected by dev LAS, with dev UAS tie-break
- `model_labels.pkl`: dependency label vocabulary
- `predictions.conllu`: predicted test trees from the selected checkpoint
- `results.json`: evaluation and analysis summary
- `training_log.txt`: full training/evaluation log
- `figures/`: plots and analysis figures

## Final results

- Dev UAS/LAS: `89.76` / `86.49`
- Test UAS/LAS: `87.69` / `83.69`

Final test score for the frozen M1 baseline: `87.69` UAS / `83.69` LAS.

## Figures

The experiment folder includes the core analysis figures:

- `training_curves.png`
- `final_scores_dev_test.png`
- `las_by_label.png`
- `scores_by_sentence_length.png`
- `label_confusion_matrix.png`
- `root_analysis.png`
- `transition_distribution.png`
- `efficiency_metrics.png`

Optional explanatory figures such as `arc_standard_transitions_schema.png` and
`example_parse_sequence.png` are not part of the frozen output set at this stage.
