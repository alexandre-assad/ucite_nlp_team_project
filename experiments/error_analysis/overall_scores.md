# Overall Scores

| Model | Description | Stored Test UAS | Stored Test LAS | Recomputed Test UAS | Recomputed Test LAS | Warning |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M0 | Arc-Eager, full training set, sparse perceptron | 83.76 | 77.92 | 83.76 | 77.92 | - |
| M0p | Arc-Eager, projective subset, sparse perceptron | 85.17 | 79.50 | 85.17 | 79.50 | - |
| M1 | Arc-Standard, projective subset, sparse perceptron | 87.69 | 83.69 | 87.69 | 83.69 | - |
| M2 | Arc-Standard, minimal FastText dense embeddings + MLP | 78.47 | 69.74 | 78.47 | 69.74 | - |
| M2b | Arc-Standard, FastText rich dense features + MLP | 88.90 | 85.12 | 88.90 | 85.12 | - |
| M3 | Arc-Standard, CamemBERT contextual rich features + MLP | 93.41 | 90.51 | 93.41 | 90.51 | - |
