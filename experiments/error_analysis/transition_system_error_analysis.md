# Transition System Error Analysis

## Controlled comparison motivation

- M0 vs M1 was not perfectly controlled because M0 used the full training set and M1 used projective filtering.
- M0p makes the comparison fairer by training Arc-Eager on the same projective subset used by M1.
- The results suggest that projective filtering improves Arc-Eager.
- The results also suggest that Arc-Standard still outperforms Arc-Eager under the same projective training condition.

## Test-score differences

- M0p vs M0: `+1.41` UAS, `+1.58` LAS
- M1 vs M0p: `+2.52` UAS, `+4.19` LAS
- M1 vs M0: `+3.93` UAS, `+5.77` LAS

## Interpretation

The results suggest that projective filtering alone helps Arc-Eager, but it does not close the gap to Arc-Standard.
This supports using M0p vs M1 as the cleaner transition-system comparison for the report.
