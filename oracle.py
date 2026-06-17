"""
oracle.py — Static Oracle for the Arc-Eager System

An oracle tells us what the correct transition is at each step during
training. It looks at the current configuration and the gold-standard
(correct) tree, and decides which transition to apply.

This is a "static" oracle: there is exactly one correct transition at
each step. (A "dynamic" oracle, which is an improvement we don't implement
here, would allow multiple correct transitions.)

The logic for the arc-eager static oracle is:

  1. LEFT-ARC(l) — if the front of the buffer is the head of the stack top
                    in the gold tree, and the label is l.
  2. RIGHT-ARC(l) — if the stack top is the head of the buffer front
                     in the gold tree, and the label is l.
  3. REDUCE — if the stack top already has a head (from a previous LEFT-ARC
              or RIGHT-ARC), AND all of its gold dependents have already
              been assigned. We can safely remove it.
  4. SHIFT — otherwise, we push the next buffer token onto the stack.
"""

from framework import Oracle
from transitions import make_left_arc, make_right_arc


def get_oracle_transition(config):
    """
    Determine the correct transition for the current configuration,
    based on the gold-standard dependency tree stored in the sentence.
    
    Parameters:
        config: a Configuration object. The sentence tokens contain the gold
                'head' and 'deprel' fields.
    
    Returns:
        A transition string (e.g., "SHIFT", "LEFT-ARC:nsubj", etc.)
    """
    stack_top = config.get_stack_top()
    buffer_front = config.get_buffer_front()

    # if the buffer is empty, the only option is REDUCE
    if buffer_front is None:
        return "REDUCE"

    # get the gold head and label for the stack top and buffer front
    s0_token = config.sentence[stack_top]
    b0_token = config.sentence[buffer_front]

    # ── Case 1: LEFT-ARC ─────────────────────────────────────
    # The buffer front is the gold head of the stack top
    # (and the stack top is not ROOT)
    if stack_top != 0 and s0_token["head"] == buffer_front:
        label = s0_token["deprel"]
        return make_left_arc(label)

    # ── Case 2: RIGHT-ARC ────────────────────────────────────
    # The stack top is the gold head of the buffer front
    if b0_token["head"] == stack_top:
        label = b0_token["deprel"]
        return make_right_arc(label)

    # ── Case 3: REDUCE ───────────────────────────────────────
    # The stack top already has a head, and all its gold dependents
    # that are to its left have already been collected
    if stack_top != 0 and config.has_head(stack_top):
        # check if all gold dependents of stack_top have been attached
        if _all_dependents_attached(config, stack_top):
            return "REDUCE"

    # ── Case 4: SHIFT ────────────────────────────────────────
    # Default action: push the next buffer token onto the stack
    return "SHIFT"


def _all_dependents_attached(config, token_id):
    """
    Check whether all gold-standard dependents of a token have already
    been assigned in the current configuration.
    
    We look at all tokens in the sentence and check: for each token whose
    gold head is token_id, has it already been given a head in config.arcs?
    """
    for token in config.sentence:
        if token["id"] == 0:
            continue  # skip ROOT
        # if this token's gold head is our token, check if it's been attached
        if token["head"] == token_id:
            if not config.has_head(token["id"]):
                return False
    return True


class StaticArcEagerOracle(Oracle):
    """Static arc-eager oracle wrapped in the generic framework API."""

    def choose(self, config):
        return get_oracle_transition(config)


def get_arc_standard_oracle_transition(config):
    """
    Determine the static Arc-Standard oracle transition for the configuration.
    """
    stack_top = config.get_stack_top()
    stack_second = config.get_stack_second()
    buffer_front = config.get_buffer_front()

    if stack_top is None:
        raise ValueError("Arc-Standard oracle received an empty stack")

    if stack_second is not None and stack_top != 0:
        s0_token = config.sentence[stack_top]
        s1_token = config.sentence[stack_second]

        # If s0 is the gold head of s1, attach s1 now.
        if (
            stack_second != 0
            and s1_token["head"] == stack_top
            and _all_dependents_attached(config, stack_second)
        ):
            return make_left_arc(s1_token["deprel"])

        # If s1 is the gold head of s0 and all gold dependents of s0 are attached,
        # attach s0 now.
        if s0_token["head"] == stack_second and _all_dependents_attached(config, stack_top):
            return make_right_arc(s0_token["deprel"])

    if buffer_front is not None:
        return "SHIFT"

    # If buffer is empty, a projective gold tree should still permit a reduce-like
    # Arc-Standard attachment before termination. Reaching this point means the
    # current configuration is inconsistent with the oracle assumptions.
    raise ValueError("Arc-Standard oracle reached a non-terminal state with no valid SHIFT")


class StaticArcStandardOracle(Oracle):
    """Static Arc-Standard oracle wrapped in the generic framework API."""

    def choose(self, config):
        return get_arc_standard_oracle_transition(config)
