"""Utilities and deep-learning baselines for the UCI HAR Dataset."""

from .data import ACTIVITIES, SIGNALS, HARSplit, load_split, standardize

__all__ = [
    "ACTIVITIES",
    "SIGNALS",
    "HARSplit",
    "load_split",
    "standardize",
]
