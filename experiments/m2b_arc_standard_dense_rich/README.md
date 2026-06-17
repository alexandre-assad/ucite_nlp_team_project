# M2b Arc-Standard Dense-Rich Experiment

## Goal

Add richer dense syntactic features inspired by Chen and Manning (2014) while keeping
Arc-Standard, the static oracle, greedy decoding, and the projective training subset fixed.

## Feature template

- Word embeddings for s0, s1, s2, b0, b1, b2, s0-left, s0-right, s1-left, s1-right
- POS embeddings for the same token positions
- Dependency label embeddings for the four child relations
- Distance bucket between s0 and b0
- Stack-size bucket
- Buffer-size bucket
- NULL and ROOT indicator flags

- Embedding source: `fasttext_bin`
- Embedding dimension: `300`
- Embedding path: `data/embeddings/cc.fr.300.bin`
- POS embedding dimension: `32`
- Label embedding dimension: `32`

## Reproducibility

- Seed: `0`
- Paths are stored relative to the project root
- Best-dev checkpointing uses dev LAS with dev UAS as tie-breaker

## Final results

- Projective training sentences kept: `13,816 / 14,450`
- Generated transition examples: `665,488`
- Selected epoch: `10`
- Dev UAS/LAS: `91.56` / `88.31`
- Test UAS/LAS: `88.90` / `85.12`
- Training time: `2275.03s`
- Parsing time: `11.55s`

## Outputs

- `config.json`
- `results.json`
- `training_log.txt`
- `README.md`
- `model_best_dev.pt`
- `model_final.pt`
- `predictions.conllu`
- `figures/`
