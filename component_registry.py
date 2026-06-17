"""Component resolution helpers for parser experiments."""

from features import SparseFeatureExtractor
from oracle import StaticArcEagerOracle, StaticArcStandardOracle
from parser import GreedyDecoder
from transitions import ArcEagerSystem, ArcStandardSystem


def resolve_transition_system(name):
    """Resolve a transition system name to a concrete component."""
    if name == "arc-eager":
        return ArcEagerSystem()
    if name == "arc-standard":
        return ArcStandardSystem()
    raise ValueError(f"Unknown transition system: {name}")


def resolve_oracle(name, transition_system_name=None):
    """Resolve an oracle name to a concrete component."""
    if name == "static":
        if transition_system_name == "arc-standard":
            return StaticArcStandardOracle()
        return StaticArcEagerOracle()
    raise ValueError(f"Unknown oracle: {name}")


def resolve_feature_extractor(name):
    """Resolve a feature extractor name to a concrete component."""
    if name in {
        "sparse",
        "sparse-handcrafted",
        "sparse handcrafted",
        "sparse handcrafted features",
    }:
        return SparseFeatureExtractor()
    raise ValueError(f"Unknown feature extractor: {name}")


def resolve_decoder(name):
    """Resolve a decoder name to a concrete component."""
    if name == "greedy":
        return GreedyDecoder()
    raise ValueError(f"Unknown decoder: {name}")
