# M0 Baseline Experiment

## Goal

Freeze and document the baseline parser before comparing it with more advanced projective parsing strategies.

Model definition:

- Arc-Eager transition system
- Static oracle
- Sparse handcrafted features
- Averaged perceptron classifier
- Greedy decoding

## Architecture

The parser processes each sentence with a stack, a buffer, and a set of dependency arcs.
At training time, a static oracle provides the gold transition for each configuration.
The averaged perceptron scores valid transitions from sparse local features.
At test time, decoding is greedy: the highest-scoring valid transition is applied at each step.

## Arc-Eager Transition System

- `SHIFT`: move the buffer front onto the stack
- `LEFT-ARC(label)`: add `b0 -> s0` and pop `s0`
- `RIGHT-ARC(label)`: add `s0 -> b0` and push `b0` onto the stack
- `REDUCE`: pop the stack top once it already has a head

## Static Oracle

The static oracle chooses:

- `LEFT-ARC` when the buffer front is the gold head of the stack top
- `RIGHT-ARC` when the stack top is the gold head of the buffer front
- `REDUCE` when the stack top already has a head and all of its gold dependents are attached
- `SHIFT` otherwise

## Sparse Feature Templates

- lexical features for `s0`, `s1`, `b0`, `b1`
- UPOS features and POS n-grams
- word + POS combinations
- leftmost/rightmost child features
- dependent count of `s0`
- distance, stack size, and buffer size
- bias feature

## Averaged Perceptron

The classifier is an averaged perceptron over sparse feature vectors.
Weights are updated online from oracle supervision, then averaged at the end of training.

## Greedy Decoding

At each parser step, the model scores the valid transitions and applies the best one.
A lightweight post-processing step repairs malformed outputs so each final tree has exactly one root.

## Evaluation Metrics

- `UAS`: head accuracy
- `LAS`: head + label accuracy
- root accuracy and malformed parse analysis are also reported

## Final Results

- Dev UAS/LAS: `86.86` / `81.64`
- Test UAS/LAS: `83.76` / `77.92`
- Training time: `11439.90s`
- Parsing time: `32.46s`
- Test throughput: `12.81` sent/s, `308.55` tok/s

## Key Observations From Error Analysis

- Root accuracy after repair: `69.71%`
- Malformed sentences before repair: `122` / `416`
- Repaired sentences: `0`
- Most frequent labels and their LAS:
  - `det`: 99.25% over 1459 tokens
  - `case`: 98.32% over 1309 tokens
  - `punct`: 68.13% over 1186 tokens
  - `nmod`: 76.87% over 856 tokens
  - `nsubj`: 72.64% over 519 tokens

## Files

- `model.pkl`: trained perceptron weights
- `model_labels.pkl`: dependency label vocabulary
- `predictions.conllu`: predicted test trees
- `config.json`: experiment configuration
- `results.json`: metrics and analysis summary
- `training_log.txt`: full console log
- `figures/`: plots and explanatory diagrams
