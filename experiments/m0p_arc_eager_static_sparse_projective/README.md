# M0p Arc-Eager Projective Sparse Experiment

## Goal

Train Arc-Eager under the same projective training condition used by M1/M2/M2b/M3,
while keeping the sparse perceptron parser architecture unchanged.

## M0p definition

- Arc-Eager transition system
- Static Arc-Eager oracle
- Sparse handcrafted features
- Averaged perceptron classifier
- Greedy decoding
- Projective training filter

## Projective filtering

- Training sentences kept: `13816 / 14450`
- Dev and test sets are not filtered

## Checkpoint selection

Best-dev checkpointing is enabled for M0p.
The selected checkpoint is epoch `8`, chosen by highest dev LAS with dev UAS as the tie-breaker.

## Final results

- Dev UAS/LAS: `88.04` / `82.96`
- Test UAS/LAS: `85.17` / `79.50`

## Outputs

- `config.json`
- `results.json`
- `training_log.txt`
- `README.md`
- `model_best_dev.pkl`
- `model_final.pkl`
- `model_labels.pkl`
- `predictions.conllu`
- `figures/`
