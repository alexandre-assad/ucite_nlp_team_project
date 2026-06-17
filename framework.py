"""Generic abstractions for transition-based dependency parsing components."""

from abc import ABC, abstractmethod


class TransitionSystem(ABC):
    """Abstract transition system interface."""

    @abstractmethod
    def initial_config(self, sentence):
        """Build the initial parser configuration for a sentence."""

    @abstractmethod
    def is_terminal(self, config):
        """Return True when the configuration is terminal."""

    @abstractmethod
    def valid_transitions(self, config, labels):
        """Return the valid transitions for the current configuration."""

    @abstractmethod
    def apply(self, config, transition):
        """Apply a transition to the configuration."""


class Oracle(ABC):
    """Abstract oracle interface."""

    @abstractmethod
    def choose(self, config):
        """Choose one gold transition for the current configuration."""

    def gold_transitions(self, config):
        """Return the set of gold transitions for future non-deterministic oracles."""
        return [self.choose(config)]


class FeatureExtractor(ABC):
    """Abstract feature extractor interface."""

    @abstractmethod
    def extract(self, config):
        """Extract features from the current configuration."""


class Decoder(ABC):
    """Abstract decoder interface."""

    @abstractmethod
    def parse(self, sentence, model, transition_system, feature_extractor, labels):
        """Parse one sentence with the provided model and parser components."""
