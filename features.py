"""
features.py — Feature Extraction from Configurations

Features are how the parser "sees" the current state. We extract a set of
indicator features (each is either present or absent) from the configuration.
These features are what the perceptron uses to decide which transition to apply.

We use a "sparse" representation: a dictionary mapping feature names (strings)
to 1. Only the features that are active are stored — everything else is
implicitly 0. This is efficient because each configuration activates only
a small number of features out of a potentially huge feature space.

Our feature templates are inspired by the classic features from
Zhang and Nivre (2011) and similar work:

  1. Unigram features: word form and POS of important positions
     (stack top, second on stack, buffer front, buffer second)
  2. Bigram features: pairs of POS tags from these positions
  3. Children features: leftmost/rightmost dependents of stack top
  4. Distance feature: how far apart are the stack top and buffer front
"""

from framework import FeatureExtractor


def extract_features(config):
    """
    Extract features from the current configuration.
    
    Parameters:
        config: a Configuration object
    
    Returns:
        features: a dictionary {feature_name: 1} for all active features
    """
    features = {}

    # ── Get the key positions ────────────────────────────────
    s0_idx = config.get_stack_top()       # top of stack
    s1_idx = config.get_stack_second()    # second on stack
    b0_idx = config.get_buffer_front()    # front of buffer
    b1_idx = config.get_buffer_second()   # second in buffer

    # ── Get the token information at each position ───────────
    s0 = config.get_token(s0_idx)
    s1 = config.get_token(s1_idx)
    b0 = config.get_token(b0_idx)
    b1 = config.get_token(b1_idx)

    # ── Get children of the stack top ────────────────────────
    s0_left = config.get_token(config.get_left_most_dep(s0_idx)) if s0_idx is not None else config.get_token(None)
    s0_right = config.get_token(config.get_right_most_dep(s0_idx)) if s0_idx is not None else config.get_token(None)

    # ── 1. UNIGRAM FEATURES ─────────────────────────────────
    # Word forms
    features[f"s0.form={s0['form']}"] = 1
    features[f"s1.form={s1['form']}"] = 1
    features[f"b0.form={b0['form']}"] = 1
    features[f"b1.form={b1['form']}"] = 1

    # POS tags
    features[f"s0.upos={s0['upos']}"] = 1
    features[f"s1.upos={s1['upos']}"] = 1
    features[f"b0.upos={b0['upos']}"] = 1
    features[f"b1.upos={b1['upos']}"] = 1

    # Lemmas
    features[f"s0.lemma={s0['lemma']}"] = 1
    features[f"b0.lemma={b0['lemma']}"] = 1

    # ── 2. BIGRAM FEATURES ──────────────────────────────────
    # Pairs of POS tags — these capture patterns like
    # "a verb on the stack and a noun in the buffer"
    features[f"s0.upos+b0.upos={s0['upos']}+{b0['upos']}"] = 1
    features[f"s0.upos+s1.upos={s0['upos']}+{s1['upos']}"] = 1
    features[f"b0.upos+b1.upos={b0['upos']}+{b1['upos']}"] = 1
    features[f"s0.form+b0.form={s0['form']}+{b0['form']}"] = 1

    # Word + POS combinations
    features[f"s0.form+s0.upos={s0['form']}+{s0['upos']}"] = 1
    features[f"b0.form+b0.upos={b0['form']}+{b0['upos']}"] = 1

    # ── 3. TRIGRAM FEATURES ──────────────────────────────────
    features[f"s0.upos+b0.upos+b1.upos={s0['upos']}+{b0['upos']}+{b1['upos']}"] = 1
    features[f"s1.upos+s0.upos+b0.upos={s1['upos']}+{s0['upos']}+{b0['upos']}"] = 1

    # ── 4. CHILDREN FEATURES ────────────────────────────────
    # The leftmost/rightmost children of the stack top tell us about
    # the partial subtree we've built so far
    features[f"s0.left.upos={s0_left['upos']}"] = 1
    features[f"s0.right.upos={s0_right['upos']}"] = 1
    features[f"s0.left.form={s0_left['form']}"] = 1
    features[f"s0.right.form={s0_right['form']}"] = 1

    # Number of dependents of the stack top (bucketized)
    if s0_idx is not None:
        n_deps = len(config.get_dependents(s0_idx))
        features[f"s0.n_deps={min(n_deps, 5)}"] = 1

    # ── 5. DISTANCE FEATURE ─────────────────────────────────
    # The distance between stack top and buffer front gives a sense
    # of how "far" we are in the sentence
    if s0_idx is not None and b0_idx is not None:
        distance = abs(b0_idx - s0_idx)
        # bucketize the distance: 1, 2, 3, 4, 5+
        features[f"distance={min(distance, 5)}"] = 1

    # ── 6. BUFFER/STACK SIZE FEATURES ────────────────────────
    features[f"stack_size={min(len(config.stack), 5)}"] = 1
    features[f"buffer_size={min(len(config.buffer), 5)}"] = 1

    # ── 7. BIAS FEATURE ─────────────────────────────────────
    # A bias feature that is always active (helps the model learn
    # a default preference for transitions)
    features["BIAS"] = 1

    return features


class SparseFeatureExtractor(FeatureExtractor):
    """Sparse handcrafted feature extractor for the baseline parser."""

    def extract(self, config):
        return extract_features(config)
