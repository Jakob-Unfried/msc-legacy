"""
treating lists of ndarrays as flat vectors
"""

from typing import List, Union

import jax.numpy as np
from jax import tree_map, tree_multimap
from jax.numpy import ndarray

Vector = List[ndarray]
Scalar = Union[float, complex, ndarray]


# FUTURE speed comparison: list comprehension vs tree_map


def list_dot(x: Vector, y: Vector) -> Scalar:
    return np.sum(tree_multimap(lambda arr_x, arr_y: np.sum(arr_x * arr_y), x, y))


def list_conj(x: Vector) -> Vector:
    return tree_map(lambda arr: np.conj(arr), x)


def list_add_prefactor(x: Vector, a: Scalar, y: Vector) -> Vector:
    # mimics x + x_1*y
    return tree_multimap(lambda arr_x, arr_y: arr_x + a * arr_y, x, y)


def list_max(x: Vector) -> Scalar:
    return np.max(tree_map(lambda arr: np.max(arr), x))


def list_abs(x: Vector) -> Vector:
    return tree_map(lambda arr: np.abs(arr), x)


def list_norm(x: Vector) -> Scalar:
    return np.sqrt(np.real(list_dot(list_conj(x), x)))


def list_max_abs(x: Vector) -> Scalar:
    return list_max(list_abs(x))


def list_neg(x: Vector) -> Vector:
    return tree_map(lambda arr: -arr, x)


def list_scale(x: Vector, a: Scalar) -> Vector:
    return tree_map(lambda arr: a * arr, x)


def list_sum_abs(x: Vector) -> Scalar:
    # mimics np.sum(np.abs(x))
    return np.sum(tree_map(lambda arr: np.sum(np.abs(arr)), x))
