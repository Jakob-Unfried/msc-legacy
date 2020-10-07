"""
Short-hands for array operations for readability
"""

import jax.numpy as np

Array = np.ndarray


# noinspection PyPep8Naming
def T(x: Array) -> Array:
    """
    Returns the transpose
    `T(x) == jax.numpy.transpose(x)`
    """
    return np.transpose(x)


# noinspection PyPep8Naming
def Hc(x: Array) -> Array:
    """
    Returns the is_hermitian conjugate
    `Hc(x) == jax.numpy.transpose(x.conjugate())`
    """
    return T(Cc(x))


# noinspection PyPep8Naming
def Cc(x: Array) -> Array:
    return np.conjugate(x)
