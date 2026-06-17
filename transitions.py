"""
transitions.py — Arc-Eager Transition System

The arc-eager system (Nivre, 2003) uses four types of transitions:

  1. SHIFT       — Move the first token from the buffer onto the stack.
                   Precondition: buffer is not empty.

  2. LEFT-ARC(l) — Create an arc from the buffer front to the stack top
                   (i.e., buffer_front --l--> stack_top), then remove
                   the stack top.
                   Precondition: stack top is not ROOT (index 0).

  3. RIGHT-ARC(l) — Create an arc from the stack top to the buffer front
                    (i.e., stack_top --l--> buffer_front), then push the
                    buffer front onto the stack.
                    Precondition: buffer is not empty.

  4. REDUCE      — Pop the top of the stack.
                   Precondition: the stack top already has a head assigned.

These transitions let us build projective dependency trees in linear time.
"""

from configuration import Configuration
from framework import TransitionSystem


# ──────────────────────────────────────────────────────────────
# Transition names
# ──────────────────────────────────────────────────────────────
# We represent a transition as a string: "SHIFT", "REDUCE",
# "LEFT-ARC:label", or "RIGHT-ARC:label"

def make_left_arc(label):
    """Create a LEFT-ARC transition string with the given label."""
    return f"LEFT-ARC:{label}"


def make_right_arc(label):
    """Create a RIGHT-ARC transition string with the given label."""
    return f"RIGHT-ARC:{label}"


def parse_transition(transition):
    """
    Parse a transition string into (action, label).
    For SHIFT and REDUCE, label is None.
    For LEFT-ARC:nsubj, returns ("LEFT-ARC", "nsubj"), etc.
    """
    if ":" in transition:
        action, label = transition.split(":", 1)
        return action, label
    return transition, None


# ──────────────────────────────────────────────────────────────
# Preconditions: which transitions are valid?
# ──────────────────────────────────────────────────────────────

def get_valid_transitions(config, labels):
    """
    Return a list of all valid transitions for the current configuration.
    
    We check the preconditions for each transition type:
      - SHIFT:       buffer must not be empty
      - LEFT-ARC(l): stack top must not be ROOT, and must not already have a head
      - RIGHT-ARC(l): buffer must not be empty
      - REDUCE:      stack top must already have a head
    
    Parameters:
        config: a Configuration object
        labels: list of all possible dependency labels
    
    Returns:
        list of transition strings
    """
    valid = []
    stack_top = config.get_stack_top()
    buffer_front = config.get_buffer_front()

    # SHIFT: we can shift if the buffer is not empty
    # But we should not shift if the buffer has just one element and
    # the stack has elements that still need to be processed
    if buffer_front is not None:
        valid.append("SHIFT")

    # LEFT-ARC: stack top is not ROOT and doesn't already have a head
    if stack_top is not None and stack_top != 0 and buffer_front is not None:
        if not config.has_head(stack_top):
            for label in labels:
                valid.append(make_left_arc(label))

    # RIGHT-ARC: buffer must not be empty
    if stack_top is not None and buffer_front is not None:
        for label in labels:
            valid.append(make_right_arc(label))

    # REDUCE: stack top must already have a head
    if stack_top is not None and stack_top != 0:
        if config.has_head(stack_top):
            valid.append("REDUCE")

    return valid


# ──────────────────────────────────────────────────────────────
# Applying transitions
# ──────────────────────────────────────────────────────────────

def apply_transition(config, transition):
    """
    Apply a transition to the configuration (modifies it in place).
    
    Parameters:
        config:     a Configuration object
        transition: a transition string like "SHIFT", "LEFT-ARC:nsubj", etc.
    """
    action, label = parse_transition(transition)

    if action == "SHIFT":
        # move the front of the buffer onto the stack
        token = config.buffer.pop(0)
        config.stack.append(token)

    elif action == "LEFT-ARC":
        # create arc: buffer_front --> stack_top with the given label
        stack_top = config.stack.pop()
        buffer_front = config.buffer[0]
        config.arcs.append((buffer_front, stack_top, label))

    elif action == "RIGHT-ARC":
        # create arc: stack_top --> buffer_front with the given label
        stack_top = config.stack[-1]
        buffer_front = config.buffer.pop(0)
        config.arcs.append((stack_top, buffer_front, label))
        config.stack.append(buffer_front)

    elif action == "REDUCE":
        # simply pop the stack top
        config.stack.pop()

    else:
        raise ValueError(f"Unknown transition: {transition}")


class ArcEagerSystem(TransitionSystem):
    """Arc-eager transition system wrapped in the generic framework API."""

    def initial_config(self, sentence):
        return Configuration(sentence)

    def is_terminal(self, config):
        return config.is_terminal()

    def valid_transitions(self, config, labels):
        return get_valid_transitions(config, labels)

    def apply(self, config, transition):
        apply_transition(config, transition)


def get_valid_transitions_arc_standard(config, labels):
    """Return valid Arc-Standard transitions for the current configuration."""
    valid = []
    stack_top = config.get_stack_top()
    stack_second = config.get_stack_second()
    buffer_front = config.get_buffer_front()

    if buffer_front is not None:
        valid.append("SHIFT")

    if stack_top is not None and stack_second is not None:
        if stack_second != 0 and not config.has_head(stack_second):
            for label in labels:
                valid.append(make_left_arc(label))
        if stack_top != 0 and not config.has_head(stack_top):
            for label in labels:
                valid.append(make_right_arc(label))

    return valid


def apply_transition_arc_standard(config, transition):
    """Apply an Arc-Standard transition to the configuration."""
    action, label = parse_transition(transition)

    if action == "SHIFT":
        token = config.buffer.pop(0)
        config.stack.append(token)

    elif action == "LEFT-ARC":
        # create arc: s0 --> s1, then remove s1
        stack_top = config.stack[-1]
        stack_second = config.stack[-2]
        config.arcs.append((stack_top, stack_second, label))
        del config.stack[-2]

    elif action == "RIGHT-ARC":
        # create arc: s1 --> s0, then remove s0
        stack_top = config.stack.pop()
        stack_second = config.stack[-1]
        config.arcs.append((stack_second, stack_top, label))

    else:
        raise ValueError(f"Unknown Arc-Standard transition: {transition}")


class ArcStandardSystem(TransitionSystem):
    """Arc-standard transition system wrapped in the generic framework API."""

    def initial_config(self, sentence):
        return Configuration(sentence)

    def is_terminal(self, config):
        return len(config.buffer) == 0 and len(config.stack) == 1

    def valid_transitions(self, config, labels):
        return get_valid_transitions_arc_standard(config, labels)

    def apply(self, config, transition):
        apply_transition_arc_standard(config, transition)
