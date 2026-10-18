"""Small descriptive receipts for an unchanged global observation selection."""
import math
import numbers

import numpy as np


def selection_composition(classes, training_counts=None, selected_counts=None):
    """Describe known class counts without inferring absent historical evidence.

    Labels and count vectors share their existing class order. Missing or invalid
    counts return an unavailable receipt; they never imply a balanced sample.
    All returned values are independent JSON-native scalars and containers.
    """
    unavailable = {"available": False, "reason": "training or selected class counts unavailable or invalid"}
    if training_counts is None or selected_counts is None:
        return unavailable
    try:
        labels = np.asarray(classes)
        training = np.asarray(training_counts)
        selected = np.asarray(selected_counts)
    except (TypeError, ValueError, OverflowError):
        return unavailable
    if (labels.ndim != 1 or not labels.size or training.shape != labels.shape
            or selected.shape != labels.shape or training.dtype.kind not in "iu"
            or selected.dtype.kind not in "iu"):
        return unavailable
    # Python counts avoid fixed-width overflow or mixed signed/unsigned coercion.
    training = [int(count) for count in training]
    selected = [int(count) for count in selected]
    if any(count <= 0 or retained < 0 or retained > count
           for count, retained in zip(training, selected)):
        return unavailable
    native_labels = []
    for label in labels:
        label = label.item() if isinstance(label, np.generic) else label
        if isinstance(label, (str, bool, int)):
            native_labels.append(label)
        elif isinstance(label, numbers.Real) and math.isfinite(float(label)):
            native_labels.append(float(label))
        else:
            return unavailable
    if len(set(native_labels)) != len(native_labels):
        return unavailable
    training_size, selected_size = sum(training), sum(selected)
    return {"available": True, "training_size": training_size,
            "selected_size": selected_size,
            "selection_ratio_realized": selected_size / training_size,
            "classes": [{"label": label, "training_count": count,
                         "training_fraction": count / training_size,
                         "selected_count": retained,
                         "selected_fraction": retained / selected_size if selected_size else None,
                         "class_retention_fraction": retained / count}
                        for label, count, retained in zip(native_labels, training, selected)],
            "interpretation": "Observed selection composition; no class balancing or ratio adjustment is performed."}
