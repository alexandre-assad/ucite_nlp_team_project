# Error Analysis Summary

## Purpose

This analysis compares the frozen parsing outputs of M0, M0p, M1, M2, M2b, and M3 to understand transition-system effects and representation effects without rerunning any model.

## Overall comparison

- Test LAS rises from `77.92` in M0 to `79.50` in M0p, suggesting that projective filtering helps Arc-Eager.
- Under the same projective training condition, M1 still improves over M0p by `+2.52` UAS and `+4.19` LAS.
- M2 drops sharply relative to M1, while M2b and M3 recover and then exceed the sparse baseline.

## Arc-Eager vs Arc-Standard

The results suggest that Arc-Standard remains stronger than Arc-Eager even after controlling for projective filtering with M0p.
This is consistent with the hypothesis that Arc-Standard's bottom-up attachment order may avoid some premature Arc-Eager attachment or reduction mistakes.

## Sparse vs dense vs contextual representations

The results suggest that minimal dense lexical embeddings alone are not enough for this transition-based parser setting.
M2 underperforms badly, which is consistent with the idea that replacing sparse syntactic cues with only word vectors removes too much explicit structure.
M2b restores much of that structure with POS and dependency-label information and improves strongly over M2.
M3 then adds a further large improvement, suggesting that contextualized embeddings inject useful sentence-level information into local parsing decisions.

## Label-level patterns

The largest M1-over-M0p label gains among labels with enough support are:
- `advcl`: +39.45 LAS
- `obl:agent`: +39.40 LAS
- `conj`: +24.77 LAS
- `acl:relcl`: +24.29 LAS
- `nsubj:pass`: +16.92 LAS

The largest M3-over-M2b label gains among labels with enough support are:
- `acl:relcl`: +20.00 LAS
- `acl`: +19.62 LAS
- `conj`: +19.43 LAS
- `advcl`: +18.35 LAS
- `fixed`: +17.14 LAS

These gains should be interpreted cautiously, especially for labels with lower support.

## Sentence length and dependency distance

The bucket tables show whether gains are uniform or concentrated in longer sentences and longer dependencies.
The results suggest that contextual and richer dense representations help across buckets, but the exact pattern is mixed in some ranges and should not be overinterpreted without the full tables.

## ROOT errors

ROOT precision, recall, and F1 improve steadily across the model sequence, with M3 achieving the strongest ROOT behavior.

## Concrete examples

Representative examples are saved separately for M0p vs M1, M2 vs M2b, and M2b vs M3.
They are intended to support the transition-system and representation comparisons with concrete token-level cases.

## Conclusions

The results suggest three main conclusions:
- Projective filtering alone does not explain the M1 advantage; Arc-Standard still outperforms Arc-Eager under matched projective training conditions.
- Minimal dense lexical embeddings underperform because they remove too many explicit syntactic cues.
- Rich dense features and then contextual embeddings produce progressively stronger results, with M3 as the strongest frozen model.

## Limitations

- Error categories are automatically assigned and should be treated as heuristic labels.
- Some dependency labels have low support and should not be overinterpreted.
- The analysis is based on the frozen test predictions only and does not include retraining or resampling.
