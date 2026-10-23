"""Projected and RKHS stochastic-gradient distance-weighted discrimination."""

__version__ = "0.9.3"

from ._estimators import (KernelSGDDWD, LinearSGDDWD, NumericalFailure,
                          SelectionError, TrainingError, TwoStageDWDClassifier)
from ._core import dwd_loss, dwd_loss_derivative
from .model_selection import make_search, mnist_param_grid, neg_dwd_loss
from .profiling import prediction_profile
from .diagnostics import optimization_diagnostics

__all__ = ["LinearSGDDWD", "KernelSGDDWD", "TwoStageDWDClassifier",
           "TrainingError", "NumericalFailure", "SelectionError",
           "dwd_loss", "dwd_loss_derivative", "make_search", "mnist_param_grid",
           "neg_dwd_loss", "prediction_profile", "optimization_diagnostics"]
