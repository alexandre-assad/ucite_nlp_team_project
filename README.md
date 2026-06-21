# A Comparative Study of Transition-Based Dependency Parsing for French

This repository contains the code, frozen experiment outputs, error analysis,
and report sources for our project on transition-based dependency parsing for
French.

Central research question:

How do transition systems and feature representations affect the performance of
a transition-based dependency parser for French?

## Repository Structure

- `data/`: treebank files and embedding placement instructions
- `experiments/`: frozen experiment outputs and error analysis
- `tests/`: unit tests
- project root: parser modules, experiment runners, utilities, and report files

Useful top-level scripts:

- `main.py`: generic CLI for the classical sparse parser pipeline
- `run_m0_experiment.py`
- `run_m0p_projective_experiment.py`
- `run_m1_experiment.py`
- `run_m2_dense_experiment.py`
- `run_m2b_dense_rich_experiment.py`
- `run_m3_contextual_experiment.py`
- `analyze_errors.py`

Useful report/result files:

- `report.tex`
- `main.tex`
- `requirements.txt`
- `experiments/summary.csv`

## Frozen Models

| Model | Main idea | Test UAS | Test LAS |
| --- | --- | ---: | ---: |
| M0 | Arc-Eager + sparse perceptron, full train set | 83.76 | 77.92 |
| M0p | Arc-Eager + sparse perceptron, projective train subset | 85.17 | 79.50 |
| M1 | Arc-Standard + sparse perceptron, projective train subset | 87.69 | 83.69 |
| M2 | Arc-Standard + minimal FastText word embeddings + MLP | 78.47 | 69.74 |
| M2b | Arc-Standard + rich dense FastText features + MLP | 88.90 | 85.12 |
| M3 | Arc-Standard + frozen CamemBERT contextual features + MLP | 93.41 | 90.51 |

Main findings:

- Projective filtering improves Arc-Eager.
- Arc-Standard still beats Arc-Eager under matched projective training.
- Minimal word embeddings alone are not enough.
- Rich dense syntactic features recover and surpass the sparse baseline.
- Contextual embeddings give the strongest final performance.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The main external packages are:

- `numpy`
- `matplotlib`
- `torch`
- `transformers`
- `fasttext`
- `sentencepiece`

Notes:

- The sparse baselines do not require the neural dependencies.
- M2 and M2b require the FastText French embedding file.
- M3 requires a locally available `camembert-base` model.

## Data Placement

Place the UD French-GSD files in `data/`:

- `data/fr_gsd-ud-train.conllu`
- `data/fr_gsd-ud-dev.conllu`
- `data/fr_gsd-ud-test.conllu`

For M2 and M2b, place the FastText embedding file at:

- `data/embeddings/cc.fr.300.bin`

See:

- `data/embeddings/README.md`

For M3, `camembert-base` must already be available in the local Hugging Face
cache because the code uses `local_files_only=True`.

## Running

Frozen experiment runners:

```bash
python run_m0_experiment.py
python run_m0p_projective_experiment.py
python run_m1_experiment.py
python run_m2_dense_experiment.py
python run_m2b_dense_rich_experiment.py
python run_m3_contextual_experiment.py
python analyze_errors.py
```

Generic sparse parser CLI:

```bash
python main.py --help
```

## Results

The repository already includes the frozen outputs. The main summary file is:

```text
experiments/summary.csv
```

The cross-model error analysis is in:

```text
experiments/error_analysis/
```
