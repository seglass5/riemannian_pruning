"""Pruning strategies."""

from src.pruning.base import BasePruner, HeadPruner, PruningMask
from src.pruning.geometry import GeometryPruner
from src.pruning.head_pruners import ActivationPruner, MagnitudePruner, RandomPruner, RicciPruner
from src.pruning.magnitude import HeadMagnitudePruner
from src.pruning.magnitude import MagnitudePruner as UnstructuredMagnitudePruner

__all__ = [
    # Base classes
    "BasePruner",
    "HeadPruner",
    "PruningMask",
    # Head-structured pruners
    "MagnitudePruner",
    "ActivationPruner",
    "RicciPruner",
    "RandomPruner",
    # Legacy / unstructured
    "UnstructuredMagnitudePruner",
    "HeadMagnitudePruner",
    "GeometryPruner",
]
