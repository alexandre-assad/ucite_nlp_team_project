# Embedding File Placement

Expected file name: `cc.fr.300.bin`

Expected path inside the project:

`data/embeddings/cc.fr.300.bin`

This large embedding file is not committed with the project if it is too large to share.

To reproduce the full M2 run, each teammate should download the French FastText binary file
and place it at the path above while keeping the same internal project structure.

```bash
python run_m2_dense_experiment.py
```
