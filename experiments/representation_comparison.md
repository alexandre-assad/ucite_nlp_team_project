# Representation Comparison: M1 vs M2 vs M2b vs M3

## Summary table

| Model | Transition system | Oracle | Representation | Classifier | Decoder | Train sentences | Selected epoch | Dev UAS | Dev LAS | Test UAS | Test LAS |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | Arc-Standard | Static | Sparse handcrafted | Averaged perceptron | Greedy | 13816 | 10 | 89.76 | 86.49 | 87.69 | 83.69 |
| M2 | Arc-Standard | Static | Static dense word embeddings | MLP | Greedy | 13816 | 9 | 80.87 | 72.23 | 78.47 | 69.74 |
| M2b | Arc-Standard | Static | Dense rich embeddings | MLP | Greedy | 13816 | 10 | 91.56 | 88.31 | 88.90 | 85.12 |
| M3 | Arc-Standard | Static | Contextual rich embeddings | MLP | Greedy | 13816 | 8 | 95.09 | 92.76 | 93.41 | 90.51 |

## Absolute improvements

### M2b vs M1

- Test UAS: `+1.21`
- Test LAS: `+1.43`

### M2b vs M2

- Test UAS: `+10.43`
- Test LAS: `+15.38`

### M3 vs M2b

- Test UAS: `+4.51`
- Test LAS: `+5.39`

### M3 vs M1

- Test UAS: `+5.72`
- Test LAS: `+6.82`

### M3 vs M2

- Test UAS: `+14.94`
- Test LAS: `+20.77`

## Interpretation

- M2, which used only FastText word embeddings for a few parser positions, performed much worse than the sparse M1 baseline.
- M2b, which adds POS embeddings, dependency label embeddings, and richer syntactic information, strongly improves over M2 and also outperforms M1.
- This confirms that the problem with M2 was not the use of dense representations in general, but the lack of explicit syntactic information.
- This result is consistent with neural transition-based parsing literature such as Chen & Manning, where dense parsers use embeddings for words, POS tags, dependency labels and richer configuration elements, not only word vectors.
- M3 is the strongest model so far.
- M2 showed that static word embeddings alone were insufficient.
- M2b showed that dense models need explicit syntactic information such as POS tags and dependency labels.
- M3 shows that contextualized embeddings provide a major additional improvement over static embeddings.
- This supports the motivation from Kulmizev et al.: contextual embeddings are especially useful for transition-based parsing because they inject broader sentence-level information into local parser-state decisions.
