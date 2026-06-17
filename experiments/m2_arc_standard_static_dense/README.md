# M2 Arc-Standard Dense Baseline Experiment

## Goal

Compare M1 sparse handcrafted features against static dense embeddings while keeping
Arc-Standard, the static oracle, and greedy decoding fixed.

## M2 definition

- Arc-Standard transition system
- Static Arc-Standard oracle
- Static dense embeddings over s1, s0, b0, and b1
- MLP transition classifier
- Greedy decoding

## Embeddings

- Embedding source: `fasttext_bin`
- Embedding dimension: `300`
- Embedding path: `data/embeddings/cc.fr.300.bin`
- Lookup key: lemma.lower() if available and not `_`, otherwise form.lower()
- OOV strategy: `fasttext_subword_vector`

Place the FastText file at `data/embeddings/cc.fr.300.bin`.

## Reproducibility

- Seed: `0`
- Paths are stored relative to the project root
- Best-dev checkpointing uses dev LAS with dev UAS as tie-breaker

## Final results

- Selected epoch: `9`
- Dev UAS/LAS: `80.87` / `72.23`
- Test UAS/LAS: `78.47` / `69.74`

## Outputs

- `config.json`
- `results.json`
- `training_log.txt`
- `README.md`
- `model_best_dev.pt`
- `model_final.pt`
- `predictions.conllu`
- `figures/`
