"""
configuration.py — Parser Configuration

A "configuration" is a snapshot of the parser's state at a given moment.
It consists of three things:
  1. A stack   — holds tokens being processed (starts with just ROOT)
  2. A buffer  — holds tokens waiting to be processed (starts with all tokens)
  3. A set of arcs — the dependency relations we've built so far

As parsing proceeds, tokens move from the buffer to the stack, and arcs
are created between tokens. When the buffer is empty and the stack only
has ROOT left, parsing is done.
"""

import copy


class Configuration:
    """
    Represents the current state of the parser.
    
    Attributes:
        stack:    list of token indices currently on the stack
        buffer:   list of token indices waiting to be processed
        arcs:     list of (head_id, dep_id, label) triples — the built tree
        sentence: list of token dicts (the full sentence with ROOT at index 0)
    """

    def __init__(self, sentence):
        """
        Create the initial configuration for a sentence.
        - The stack starts with ROOT (index 0)
        - The buffer contains all real tokens (indices 1 to n)
        - No arcs have been built yet
        """
        self.sentence = sentence
        self.stack = [0]                             # ROOT is on the stack
        self.buffer = list(range(1, len(sentence)))  # all real tokens
        self.arcs = []                               # no arcs yet

    def get_stack_top(self):
        """Return the index of the token on top of the stack, or None if empty."""
        if len(self.stack) > 0:
            return self.stack[-1]
        return None

    def get_stack_second(self):
        """Return the index of the second element on the stack, or None."""
        if len(self.stack) > 1:
            return self.stack[-2]
        return None

    def get_buffer_front(self):
        """Return the index of the first token in the buffer, or None if empty."""
        if len(self.buffer) > 0:
            return self.buffer[0]
        return None

    def get_buffer_second(self):
        """Return the index of the second token in the buffer, or None."""
        if len(self.buffer) > 1:
            return self.buffer[1]
        return None

    def is_terminal(self):
        """
        Check if we've reached a terminal configuration.
        Parsing is done when the buffer is empty and the stack only has ROOT.
        """
        return len(self.buffer) == 0 and len(self.stack) == 1

    def has_head(self, token_id):
        """Check if a token already has a head assigned in the current arcs."""
        for head, dep, label in self.arcs:
            if dep == token_id:
                return True
        return False

    def get_head(self, token_id):
        """Return the head of a token if it has been assigned, else None."""
        for head, dep, label in self.arcs:
            if dep == token_id:
                return head
        return None

    def get_dependents(self, token_id):
        """Return the list of dependents (children) of a given token."""
        deps = []
        for head, dep, label in self.arcs:
            if head == token_id:
                deps.append(dep)
        return sorted(deps)

    def get_left_most_dep(self, token_id):
        """Return the leftmost dependent of a token, or None."""
        deps = self.get_dependents(token_id)
        if deps:
            return min(deps)
        return None

    def get_right_most_dep(self, token_id):
        """Return the rightmost dependent of a token, or None."""
        deps = self.get_dependents(token_id)
        if deps:
            return max(deps)
        return None

    def get_token(self, index):
        """
        Safely get a token dict by its index.
        Returns a dummy token if the index is None or out of bounds.
        """
        if index is not None and 0 <= index < len(self.sentence):
            return self.sentence[index]
        # return a "null" token for positions that don't exist
        return {"id": -1, "form": "NULL", "lemma": "NULL", "upos": "NULL",
                "head": -1, "deprel": "NULL"}

    def copy(self):
        """Create a deep copy of this configuration."""
        new_config = Configuration.__new__(Configuration)
        new_config.sentence = self.sentence  # shared reference is fine, we don't modify it
        new_config.stack = list(self.stack)
        new_config.buffer = list(self.buffer)
        new_config.arcs = list(self.arcs)
        return new_config

    def __repr__(self):
        """A readable representation for inspection."""
        stack_words = [self.sentence[i]["form"] for i in self.stack]
        buffer_words = [self.sentence[i]["form"] for i in self.buffer[:3]]
        return (f"Config(stack={stack_words}, "
                f"buffer={buffer_words}{'...' if len(self.buffer) > 3 else ''}, "
                f"arcs={len(self.arcs)})")
