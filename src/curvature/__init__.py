"""Ricci curvature estimation for transformer attention graphs."""

from src.curvature.aggregator import CurvatureProfile, HeadStats, LayerCurvatureAggregator, LayerStats
from src.curvature.graph import AttentionGraphBuilder
from src.curvature.ricci import OllivierRicci, OllivierRicciEstimator
from src.curvature.task import TaskConditionedCurvatureEstimator, TaskCurvatureProfile

__all__ = [
    # graph
    "AttentionGraphBuilder",
    # estimator
    "OllivierRicciEstimator",
    # hook-based wrapper (used by GeometryPruner)
    "OllivierRicci",
    # aggregation
    "LayerCurvatureAggregator",
    "CurvatureProfile",
    "HeadStats",
    "LayerStats",
    # task-conditioned curvature
    "TaskConditionedCurvatureEstimator",
    "TaskCurvatureProfile",
]
