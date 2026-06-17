# M3 Arc-Standard Contextual-Rich Experiment

## Goal

Compare M2b against a contextual variant that keeps the same parser structure
but replaces static FastText word embeddings with frozen CamemBERT representations.

## Contextual setup

- Contextual model: `camembert-base`
- Pooling strategy: `average`
- Token embeddings are cached per sentence
- CamemBERT remains frozen with `torch.no_grad()`
- CamemBERT is not fine-tuned in this version

## Final results

- Projective training sentences kept: `13816 / 14450`
- Generated transition examples: `665488`
- Selected epoch: `8`
- Dev UAS/LAS: `95.09` / `92.76`
- Test UAS/LAS: `93.41` / `90.51`
- Training time: `9759.36s`
- Parsing time: `21.82s`

## Outputs

- `config.json`
- `results.json`
- `training_log.txt`
- `README.md`
- `model_best_dev.pt`
- `model_final.pt`
- `predictions.conllu`
- `figures/`
