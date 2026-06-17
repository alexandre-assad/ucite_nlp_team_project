# A Comparative Study of Transition-Based Dependency Parsing for French

This repository contains the code, frozen experiment outputs, error analysis,
and final report for our project on transition-based dependency parsing for
French.

Central research question:

How do transition systems and feature representations affect the performance of
a transition-based dependency parser for French?

## Archive Organization

The submission is organized into clearly separated parts:

- Code: parser modules and experiment runners in the project root
- Report: `report.txt`
- Experiments: frozen outputs in `experiments/`
- Tests: `tests/`
- Data placement: `data/`

Useful top-level files and folders:

- `main.py`: generic command-line interface for the classical parser
- `run_m0_experiment.py`: frozen M0 baseline runner
- `run_m0p_projective_experiment.py`: Arc-Eager projective control experiment
- `run_m1_experiment.py`: Arc-Standard sparse baseline runner
- `run_m2_dense_experiment.py`: minimal FastText dense experiment
- `run_m2b_dense_rich_experiment.py`: rich dense FastText experiment
- `run_m3_contextual_experiment.py`: contextual Camembert experiment
- `analyze_errors.py`: cross-model error analysis
- `report.txt`: final report
- `experiments/summary.csv`: compact summary of frozen model scores

## Final Frozen Models

| Model | Main idea | Test UAS | Test LAS |
| --- | --- | ---: | ---: |
| M0 | Arc-Eager + sparse perceptron, full train set | 83.76 | 77.92 |
| M0p | Arc-Eager + sparse perceptron, projective train subset | 85.17 | 79.50 |
| M1 | Arc-Standard + sparse perceptron, projective train subset | 87.69 | 83.69 |
| M2 | Arc-Standard + minimal FastText word embeddings + MLP | 78.47 | 69.74 |
| M2b | Arc-Standard + rich dense FastText features + MLP | 88.90 | 85.12 |
| M3 | Arc-Standard + frozen Camembert contextual features + MLP | 93.41 | 90.51 |

Main findings:

- Projective filtering improves Arc-Eager: M0p vs M0 = +1.41 UAS / +1.58 LAS
- Arc-Standard still beats Arc-Eager under matched projective training:
  M1 vs M0p = +2.52 UAS / +4.19 LAS
- Minimal word embeddings alone are not enough: M2 is much weaker than M1
- Rich dense syntactic features recover and surpass the sparse baseline: M2b
  beats both M2 and M1
- Contextual embeddings are best: M3 is the strongest model

## Requirements

Python 3.10+ is recommended.

Packages needed for the full project:

- `numpy`
- `matplotlib`
- `torch`
- `transformers`
- `fasttext`

Notes:

- The sparse baselines do not require neural packages.
- Full M2 and M2b require the FastText French embedding file.
- Full M3 requires `torch`, `transformers`, and a locally available
  `camembert-base` model.
- The frozen experiment outputs are already included, so rerunning everything
  is optional.

## Data Placement

Place the treebank files in `data/`:

- `data/fr_gsd-ud-train.conllu`
- `data/fr_gsd-ud-dev.conllu`
- `data/fr_gsd-ud-test.conllu`

For M2 and M2b, place the FastText embedding file at:

- `data/embeddings/cc.fr.300.bin`

The expected embedding path is documented in:

- `data/embeddings/README.md`

## How To Run

Generic parser interface:

```bash
python main.py --help
python main.py --train data/fr_gsd-ud-train.conllu --dev data/fr_gsd-ud-dev.conllu --model model.pkl --epochs 10 --seed 0
python main.py --test data/fr_gsd-ud-test.conllu --model model.pkl --output predictions.conllu
```

Experiment runners:

```bash
python run_m0_experiment.py
python run_m0p_projective_experiment.py
python run_m1_experiment.py
python run_m2_dense_experiment.py
python run_m2b_dense_rich_experiment.py
python run_m3_contextual_experiment.py
python analyze_errors.py
```

Practical note:

- `run_m3_contextual_experiment.py` is the heaviest script.
- The repository already includes the frozen results for M0, M0p, M1, M2,
  M2b, M3, and the final error analysis.

## Online Help

Main parser help:

```text
usage: main.py [-h] [--train TRAIN] [--dev DEV] [--test TEST] [--model MODEL]
               [--epochs EPOCHS] [--seed SEED] [--output OUTPUT]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--transition-system {arc-eager,arc-standard}]
               [--oracle {static}] [--features {sparse}] [--decoder {greedy}]

Transition-based dependency parser for French
```

Runner help:

```text
python run_m0p_projective_experiment.py --help
usage: run_m0p_projective_experiment.py [-h]

Run the M0p Arc-Eager projective sparse experiment
```

```text
python run_m2_dense_experiment.py --help
usage: run_m2_dense_experiment.py [-h]

Run the M2 Arc-Standard dense experiment
```

```text
python run_m2b_dense_rich_experiment.py --help
usage: run_m2b_dense_rich_experiment.py [-h]

Run the M2b Arc-Standard dense-rich experiment
```

```text
python run_m3_contextual_experiment.py --help
usage: run_m3_contextual_experiment.py [-h]

Run the M3 Arc-Standard contextual-rich experiment
```

```text
python analyze_errors.py --help
usage: analyze_errors.py [-h]

Run cross-model error analysis over frozen parser outputs
```

## Where Results Are Stored

Frozen outputs are stored in:

- `experiments/m0_arc_eager_static_sparse/`
- `experiments/m0p_arc_eager_static_sparse_projective/`
- `experiments/m1_arc_standard_static_sparse/`
- `experiments/m2_arc_standard_static_dense/`
- `experiments/m2b_arc_standard_dense_rich/`
- `experiments/m3_arc_standard_contextual_rich/`
- `experiments/error_analysis/`

Each experiment folder contains the final documentation for that run, typically:

- `config.json`
- `results.json`
- `training_log.txt`
- `README.md`
- `predictions.conllu`
- `figures/`

Large model binaries and checkpoints are not kept in this cleaned submission
archive.

## Reproducibility Notes

- All main experiments use fixed random seed `0`.
- M0 is frozen as the epoch-10 averaged model because intermediate checkpoints
  were not saved during the first baseline run.
- M0p, M1, M2, M2b, and M3 use best-dev checkpointing by dev LAS, with dev UAS
  as tie-breaker.
- M0p, M1, M2, M2b, and M3 use the same projective training subset:
  13,816 / 14,450 training sentences.
- Dev and test sets remain unfiltered.

## Error Analysis

The final cross-model analysis is stored in `experiments/error_analysis/`.
It includes:

- overall score tables
- label-level analysis
- sentence-length analysis
- dependency-distance analysis
- root analysis
- label confusions
- pairwise qualitative examples
- summary figures

The main synthesis file is:

- `experiments/error_analysis/error_analysis_summary.md`
