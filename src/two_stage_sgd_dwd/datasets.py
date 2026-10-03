"""Strict, local-only IDX readers and reproducible binary MNIST loading.

No data are downloaded, generated, cached, or written by these utilities.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path
import struct

import numpy as np
from sklearn.utils import Bunch


__all__ = ["read_idx_images", "read_idx_labels", "load_mnist_pair"]


def _read_images(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"Truncated IDX image header: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"IDX image magic must be 2051; got {magic}: {path}")
    if count == 0 or rows == 0 or columns == 0:
        raise ValueError(f"IDX image count and dimensions must be positive: {path}")
    expected = 16 + count * rows * columns
    if len(raw) != expected:
        raise ValueError(
            f"IDX image payload length mismatch: expected {expected - 16} "
            f"bytes, got {len(raw) - 16}: {path}"
        )
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(
        count, rows * columns
    ).copy()
    return images, (rows, columns), hashlib.sha256(raw).hexdigest()


def _read_labels(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"Truncated IDX label header: {path}")
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"IDX label magic must be 2049; got {magic}: {path}")
    if count == 0:
        raise ValueError(f"IDX label count must be positive: {path}")
    if len(raw) != 8 + count:
        raise ValueError(
            f"IDX label payload length mismatch: expected {count} bytes, "
            f"got {len(raw) - 8}: {path}"
        )
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    return labels, hashlib.sha256(raw).hexdigest()


def read_idx_images(path):
    """Read uncompressed unsigned-byte IDX images into an ``(n, pixels)`` array.

    The returned array has dtype ``uint8`` and preserves file order. Image
    dimensions and observation count must be positive. Incorrect magic values,
    incomplete headers, truncated payloads, and trailing bytes are rejected.
    This general IDX reader permits dimensions other than MNIST's 28 by 28.
    """
    images, _, _ = _read_images(path)
    return images


def read_idx_labels(path):
    """Read uncompressed unsigned-byte IDX labels into an ``(n,)`` array.

    The returned array has dtype ``uint8`` and preserves file order. The count
    must be positive and the header and payload must match exactly. This
    general IDX reader does not restrict byte-valued labels to decimal digits.
    """
    labels, _ = _read_labels(path)
    return labels


def _resolve_file(directory, candidates, kind):
    matches = [directory / name for name in candidates if (directory / name).is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No {kind} file found in {directory}; expected one of: "
            + ", ".join(candidates)
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {kind} files in {directory}: "
            + ", ".join(path.name for path in matches)
            + ". Keep one supported file for this partition."
        )
    return matches[0]


def _validate_digits(digits):
    try:
        pair = tuple(digits)
    except TypeError as error:
        raise ValueError("digits must contain two distinct integers from 0 to 9") from error
    if (
        len(pair) != 2
        or any(isinstance(digit, (bool, np.bool_)) or not isinstance(digit, Integral)
               for digit in pair)
        or any(digit < 0 or digit > 9 for digit in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError("digits must contain two distinct integers from 0 to 9")
    return tuple(sorted(int(digit) for digit in pair))


def load_mnist_pair(data_dir, digits=(7, 8), partition="train"):
    """Load a binary digit pair from local, uncompressed MNIST IDX files.

    Parameters
    ----------
    data_dir : str or path-like
        Folder containing either the original ``train.idx3-ubyte`` and
        ``train_label.idx1-ubyte`` names (``test`` for the test partition), or
        standard ``train-images-idx3-ubyte`` and ``train-labels-idx1-ubyte`` names
        (``t10k`` for the test partition). Dotted ``train-images.idx3-ubyte``
        style names are also accepted. Duplicate supported names are rejected
        to avoid an ambiguous data source. Compressed files are not supported.
    digits : pair of int, default=(7, 8)
        Two distinct digits in 0 through 9. The smaller is mapped to -1 and
        the larger to +1, independently of the order supplied here. Both must
        occur in the selected partition.
    partition : {"train", "test"}, default="train"
        Official source partition. This loader makes no further split.

    Returns
    -------
    sklearn.utils.Bunch
        ``X`` contains float64 pixels divided by 255, with shape ``(n, 784)``.
        ``y`` contains int8 labels in {-1, +1}. ``ids`` contains strings such as
        ``"train:42"`` using original zero-based file indices before filtering.
        ``digits`` is the sorted digit pair. ``hashes`` maps ``"images"`` and
        ``"labels"`` to SHA256 hashes of the complete original files, including
        their headers. ``source_files`` gives the corresponding absolute paths
        and ``partition`` identifies the source partition. Source order is
        preserved; there is no sampling or shuffling.

    Raises
    ------
    ValueError
        If the digit pair, partition, file header, dimensions, payload, label
        range, image-label count, or presence of both classes is invalid.
    FileNotFoundError
        If a required source file is absent.
    """
    pair = _validate_digits(digits)
    if not isinstance(partition, str) or partition not in ("train", "test"):
        raise ValueError("partition must be 'train' or 'test'")
    directory = Path(data_dir)
    standard_prefix = "train" if partition == "train" else "t10k"
    image_path = _resolve_file(
        directory,
        (f"{partition}.idx3-ubyte", f"{standard_prefix}-images-idx3-ubyte",
         f"{standard_prefix}-images.idx3-ubyte"),
        f"{partition} image",
    )
    label_path = _resolve_file(
        directory,
        (f"{partition}_label.idx1-ubyte", f"{standard_prefix}-labels-idx1-ubyte",
         f"{standard_prefix}-labels.idx1-ubyte"),
        f"{partition} label",
    )
    images, dimensions, image_hash = _read_images(image_path)
    labels, label_hash = _read_labels(label_path)
    if dimensions != (28, 28):
        raise ValueError(f"MNIST images must be 28 by 28; got {dimensions}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Image-label count mismatch: {images.shape[0]} images, "
            f"{labels.shape[0]} labels"
        )
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be integers from 0 to 9")
    if any(not np.any(labels == digit) for digit in pair):
        raise ValueError(f"Both requested digits {pair} must occur in the partition")
    indices = np.flatnonzero((labels == pair[0]) | (labels == pair[1]))
    features = images[indices].astype(np.float64) / 255.0
    targets = np.where(labels[indices] == pair[0], -1, 1).astype(np.int8)
    return Bunch(
        X=features,
        y=targets,
        ids=np.asarray([f"{partition}:{index}" for index in indices]),
        digits=pair,
        hashes={"images": image_hash, "labels": label_hash},
        source_files={"images": str(image_path.resolve()), "labels": str(label_path.resolve())},
        partition=partition,
    )
