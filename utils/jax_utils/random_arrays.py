"""
Convenience functions to get random arrays from jax without dealing with PRNG keys explicitly.
Note that by feeding a seed from `random.randint`, a lot of properties of `jax.random` are lost.

See
https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers

"""

from random import randint
from typing import Tuple

import jax.numpy as np
from jax.random import PRNGKey, uniform as jax_uniform, normal as jax_normal


def uniform(shape: Tuple[int, ...], dtype: np.dtype, min_val: float = 0., max_val: float = 1.):
    if dtype == np.complex64:
        return uniform(shape, np.float32, min_val, max_val) + 1.j * uniform(shape, np.float32, min_val, max_val)
    if dtype == np.complex128:
        return uniform(shape, np.float64, min_val, max_val) + 1.j * uniform(shape, np.float64, min_val, max_val)

    key = PRNGKey(randint(0, 100000))
    return jax_uniform(key, shape, dtype, min_val, max_val)


def normal(shape: Tuple[int], dtype: np.dtype):
    if dtype == np.complex64:
        return normal(shape, np.float32) + 1.j * normal(shape, np.float32)
    if dtype == np.complex128:
        return normal(shape, np.float64) + 1.j * normal(shape, np.float64)
    key = PRNGKey(randint(0, 100000))
    return jax_normal(key, shape, dtype)
