"""Tiny generated IDX fixtures; no real MNIST data or numerical study."""

import hashlib
import struct

import numpy as np
import pytest
from sklearn.utils import Bunch

from two_stage_sgd_dwd.datasets import load_mnist_pair, read_idx_images, read_idx_labels


def _image_bytes(images, dimensions):
    images = np.asarray(images, dtype=np.uint8)
    return struct.pack(">IIII", 2051, len(images), *dimensions) + images.tobytes()


def _label_bytes(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    return struct.pack(">II", 2049, len(labels)) + labels.tobytes()


def _write_pair(tmp_path, partition="train", style="original", labels=(2, 8, 7, 8)):
    images = np.empty((len(labels), 28 * 28), dtype=np.uint8)
    for index in range(len(labels)):
        images[index] = (index * 85) % 256
    if style == "original":
        image_name = f"{partition}.idx3-ubyte"
        label_name = f"{partition}_label.idx1-ubyte"
    else:
        prefix = "train" if partition == "train" else "t10k"
        separator = "-" if style == "standard" else "."
        image_name = f"{prefix}-images{separator}idx3-ubyte"
        label_name = f"{prefix}-labels{separator}idx1-ubyte"
    image_path, label_path = tmp_path / image_name, tmp_path / label_name
    image_path.write_bytes(_image_bytes(images, (28, 28)))
    label_path.write_bytes(_label_bytes(labels))
    return image_path, label_path, images


def test_read_idx_images_preserves_shape_order_and_bytes(tmp_path):
    path = tmp_path / "images"
    expected = np.array([[0, 255, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.uint8)
    path.write_bytes(_image_bytes(expected, (2, 3)))
    actual = read_idx_images(path)
    assert actual.shape == (2, 6)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def test_read_idx_labels_preserves_arbitrary_byte_values(tmp_path):
    path = tmp_path / "labels"
    path.write_bytes(_label_bytes([0, 9, 255]))
    actual = read_idx_labels(path)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, [0, 9, 255])


@pytest.mark.parametrize("raw, message", [
    (b"", "header"),
    (struct.pack(">III", 2051, 1, 28), "header"),
    (struct.pack(">IIII", 2049, 1, 1, 1) + b"\x00", "magic"),
    (struct.pack(">IIII", 2051, 0, 1, 1), "positive"),
    (struct.pack(">IIII", 2051, 1, 0, 1), "positive"),
    (struct.pack(">IIII", 2051, 1, 1, 0), "positive"),
    (struct.pack(">IIII", 2051, 2, 1, 1) + b"\x00", "payload"),
    (struct.pack(">IIII", 2051, 1, 1, 1) + b"\x00\x01", "payload"),
    (struct.pack(">IIII", 2051, 2**32 - 1, 2**32 - 1, 2**32 - 1), "payload"),
])
def test_read_idx_images_rejects_malformed_files(tmp_path, raw, message):
    path = tmp_path / "bad-images"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        read_idx_images(path)


@pytest.mark.parametrize("raw, message", [
    (b"", "header"),
    (struct.pack(">I", 2049), "header"),
    (struct.pack(">II", 2051, 1) + b"\x07", "magic"),
    (struct.pack(">II", 2049, 0), "positive"),
    (struct.pack(">II", 2049, 2) + b"\x07", "payload"),
    (struct.pack(">II", 2049, 1) + b"\x07\x08", "payload"),
    (struct.pack(">II", 2049, 2**32 - 1), "payload"),
])
def test_read_idx_labels_rejects_malformed_files(tmp_path, raw, message):
    path = tmp_path / "bad-labels"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        read_idx_labels(path)


@pytest.mark.parametrize("partition", ["train", "test"])
@pytest.mark.parametrize("style", ["original", "standard", "dotted"])
def test_mnist_mapping_scaling_original_ids_and_source_hashes(tmp_path, partition, style):
    image_path, label_path, images = _write_pair(tmp_path, partition, style)
    original_files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    loaded = load_mnist_pair(tmp_path, digits=(8, 7), partition=partition)
    assert isinstance(loaded, Bunch)
    assert loaded.X.dtype == np.float64
    assert loaded.X.shape == (3, 784)
    assert loaded.y.dtype == np.int8
    np.testing.assert_array_equal(loaded.X, images[[1, 2, 3]].astype(float) / 255)
    np.testing.assert_array_equal(loaded.y, [1, -1, 1])
    np.testing.assert_array_equal(loaded.ids, [f"{partition}:1", f"{partition}:2", f"{partition}:3"])
    assert loaded.digits == (7, 8)
    assert loaded.partition == partition
    assert loaded.hashes == {
        "images": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "labels": hashlib.sha256(label_path.read_bytes()).hexdigest(),
    }
    assert loaded.source_files == {"images": str(image_path.resolve()), "labels": str(label_path.resolve())}
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original_files
    repeated = load_mnist_pair(tmp_path, partition=partition)
    np.testing.assert_array_equal(repeated.X, loaded.X)
    np.testing.assert_array_equal(repeated.ids, loaded.ids)
    assert repeated.hashes == loaded.hashes


def test_partition_names_prevent_id_collision(tmp_path):
    _write_pair(tmp_path, "train")
    _write_pair(tmp_path, "test")
    train = load_mnist_pair(tmp_path)
    test = load_mnist_pair(tmp_path, partition="test")
    assert not set(train.ids).intersection(test.ids)


@pytest.mark.parametrize("digits", [(7, 7), (7,), (7, 8, 9), (-1, 8), (7, 10),
                                  (7.0, 8), (True, 8), (np.bool_(False), 8),
                                  "78", None, 7])
def test_invalid_digit_pair_rejected_before_data_access(tmp_path, digits):
    with pytest.raises(ValueError, match="two distinct integers"):
        load_mnist_pair(tmp_path, digits=digits)


def test_numpy_integer_digits_accepted(tmp_path):
    _write_pair(tmp_path)
    assert load_mnist_pair(tmp_path, digits=np.array([8, 7])).digits == (7, 8)


@pytest.mark.parametrize("partition", ["validation", "Train", "t10k", "", None, 1, ["train"]])
def test_invalid_partition_rejected_before_data_access(tmp_path, partition):
    with pytest.raises(ValueError, match="partition"):
        load_mnist_pair(tmp_path, partition=partition)


def test_missing_local_files_report_expected_names(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-images-idx3-ubyte"):
        load_mnist_pair(tmp_path)


def test_ambiguous_source_names_are_rejected(tmp_path):
    _write_pair(tmp_path, style="original")
    _write_pair(tmp_path, style="standard")
    with pytest.raises(ValueError, match="Ambiguous"):
        load_mnist_pair(tmp_path)


def test_mnist_requires_exact_image_dimensions(tmp_path):
    image_path, _, _ = _write_pair(tmp_path)
    image_path.write_bytes(_image_bytes(np.zeros((4, 784)), (14, 56)))
    with pytest.raises(ValueError, match="28 by 28"):
        load_mnist_pair(tmp_path)


def test_mnist_rejects_mismatched_image_label_counts(tmp_path):
    _, label_path, _ = _write_pair(tmp_path)
    label_path.write_bytes(_label_bytes([7, 8, 7]))
    with pytest.raises(ValueError, match="count mismatch"):
        load_mnist_pair(tmp_path)


def test_mnist_validates_all_labels_before_filtering(tmp_path):
    _write_pair(tmp_path, labels=(255, 8, 7, 8))
    with pytest.raises(ValueError, match="0 to 9"):
        load_mnist_pair(tmp_path)


@pytest.mark.parametrize("labels", [(7, 7, 2), (8, 8, 2), (1, 2, 3)])
def test_mnist_requires_both_requested_digits(tmp_path, labels):
    _write_pair(tmp_path, labels=labels)
    with pytest.raises(ValueError, match="Both requested digits"):
        load_mnist_pair(tmp_path)
