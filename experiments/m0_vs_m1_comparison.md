# M0 vs M1 Comparison

## Summary table

| Model | Transition system | Oracle | Features | Classifier | Decoder | Train sentences | Selected epoch | Dev UAS | Dev LAS | Test UAS | Test LAS |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | Arc-Eager | Static | Sparse handcrafted | Averaged perceptron | Greedy | 14450 | 10 | 86.86 | 81.64 | 83.76 | 77.92 |
| M1 | Arc-Standard | Static | Sparse handcrafted | Averaged perceptron | Greedy | 13816 | 10 | 89.76 | 86.49 | 87.69 | 83.69 |

## Absolute improvements of M1 over M0

- Dev UAS: `+2.90`
- Dev LAS: `+4.85`
- Test UAS: `+3.93`
- Test LAS: `+5.77`

## Interpretation

Under the same sparse feature representation, averaged perceptron classifier and greedy decoding
strategy, Arc-Standard obtained substantially higher UAS and LAS than the Arc-Eager baseline.
However, the comparison must be interpreted with caution because M1 was trained on the projective
subset of the training corpus, whereas M0 was trained on the full corpus.

## Suggested control experiment

For a stricter comparison, a future M0-projective model could be trained using Arc-Eager on the
same projective subset as M1.
