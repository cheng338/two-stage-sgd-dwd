"""Fit-local, lazy LRU storage for unchanged training kernel columns."""
from collections import OrderedDict
import numbers

import numpy as np


class LazyKernelColumnCache:
    """Retain at most ``max_bytes`` of float64 column array payload.

    The limit excludes Python metadata, allocator overhead, and the current
    kernel computation's working memory. Columns are requested individually;
    construction performs no kernel evaluation or large array allocation.
    ``clear`` releases entries while preserving cumulative diagnostics.
    """

    def __init__(self, max_bytes, column_size):
        for value, name, minimum in ((max_bytes, "max_bytes", 0),
                                      (column_size, "column_size", 1)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_bytes = int(max_bytes)
        self.column_size = int(column_size)
        self.column_bytes = self.column_size * np.dtype(np.float64).itemsize
        self.capacity_columns = min(self.column_size, self.max_bytes // self.column_bytes)
        self._columns = OrderedDict()
        self.hits = self.misses = self.evictions = self.peak_bytes = 0

    @property
    def retained_bytes(self):
        return len(self._columns) * self.column_bytes

    def get(self, index, compute):
        """Return ``(column, hit)``; call ``compute`` exactly once on a miss.

        The fitting caller supplies the original contiguous float64 column.
        Neither cache lookup nor storage changes its values or layout.
        """
        column = self._columns.get(index)
        if column is not None:
            self.hits += 1
            self._columns.move_to_end(index)
            return column, True
        self.misses += 1
        column = compute()
        if (not isinstance(column, np.ndarray) or column.dtype != np.float64 or
                column.shape != (self.column_size,) or not column.flags.c_contiguous):
            raise ValueError("Kernel cache requires a contiguous float64 training column")
        if self.capacity_columns:
            if len(self._columns) == self.capacity_columns:
                self._columns.popitem(last=False)
                self.evictions += 1
            self._columns[index] = column
            self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        return column, False

    def clear(self):
        self._columns.clear()

    def diagnostics(self):
        return {"kernel_cache_hits_": self.hits,
                "kernel_cache_misses_": self.misses,
                "kernel_cache_evictions_": self.evictions,
                "kernel_cache_peak_bytes_": self.peak_bytes,
                "kernel_cache_capacity_columns_": self.capacity_columns}
