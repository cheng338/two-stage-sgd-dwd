"""Exact represented-input references for finite extreme Gaussian cases."""
from fractions import Fraction

import numpy as np


def scaled_squared_distance(left, right, gamma):
    width = Fraction.from_float(float(gamma))
    return width * sum(((Fraction.from_float(float(x)) -
                        Fraction.from_float(float(y))) ** 2
                       for x, y in zip(left, right)), Fraction())


def zero_or_one_kernel(left, right, gamma):
    """Prove the expected entries without overflowing a floating distance.

    Each test entry is either an exact duplicate or has exponent at least 2048.
    Since e>2, exp(-2048)<2^-2048<half the minimum binary64 subnormal.
    This helper deliberately refuses intermediate, nontrivial exponent cases.
    """
    result = np.empty((len(left), len(right)))
    for i, x in enumerate(left):
        for j, y in enumerate(right):
            exponent = scaled_squared_distance(x, y, gamma)
            assert exponent == 0 or exponent >= 2048
            result[i, j] = 1.0 if exponent == 0 else 0.0
    return result
