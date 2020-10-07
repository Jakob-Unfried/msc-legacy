"""
Convenience functions using jax.tree_util
"""

from functools import reduce

from jax import tree_flatten, tree_map, tree_multimap, numpy as np


def tree_allclose(*trees):
    """
    Determines if all elements of `trees`

    a) have the same tree structure
    and
    b) the corresponding leaves all fulfill ``np.allclose(leaf1, leaf2)``

    such that the trees are, up to numerical tolerances of `np.allclose`, equal
    """
    if len(trees) > 2:
        return tree_allclose(trees[0], trees[1]) and tree_allclose(*trees[1:])

    if len(trees) < 2:
        return True

    tree1, tree2 = trees
    _, tree_def1 = tree_flatten(tree1)
    _, tree_def2 = tree_flatten(tree2)

    if tree_def1 != tree_def2:
        return False

    return np.all(tree_flatten(tree_multimap(lambda arr1, arr2: np.allclose(arr1, arr2), tree1, tree2))[0])


def tree_zeros_like(tree):
    return tree_map(lambda arr: np.zeros_like(arr), tree)


def tree_sum(tree):
    """
    sum over all elements of all leaves
    """
    return np.sum(tree_flatten(tree_map(lambda arr: np.sum(arr), tree))[0])


def tree_distance(tree_a, tree_b):
    """
    sqrt( sum_i( (a_i - b_i)**2 ) )
    """
    return tree_sum(tree_multimap(lambda arr1, arr2: np.linalg.norm(arr1 - arr2) ** 2, tree_a, tree_b))


def tree_max(tree):
    return np.max(tree_flatten(tree_map(lambda arr: np.max(arr), tree))[0])


def tree_max_abs(tree):
    return np.max(tree_flatten(tree_map(lambda arr: np.max(np.abs(arr)), tree))[0])


def tree_max_abs_diff(tree_a, tree_b):
    """
    max_i ( abs(a_i - b_i) )
    """
    return tree_max(tree_multimap(lambda arr1, arr2: np.max(np.abs(arr1 - arr2)), tree_a, tree_b))


def tree_avg_abs_nonzero(tree, tolerance=0.):
    """
    average of the absolute values of all non-zero entries
    """
    _sum = tree_sum(tree_map(lambda arr: np.abs(arr), tree))
    _num = tree_sum(tree_map(lambda arr: np.sum(np.abs(arr) > tolerance), tree))
    return _sum / _num


def tree_reduce(function, tree):
    return reduce(function, tree_flatten(tree)[0])


def tree_any(tree):
    return np.any(tree_flatten(tree_map(np.any, tree))[0])


def tree_all(tree):
    return np.all(tree_flatten(tree_map(np.all, tree))[0])


def tree_conj(tree):
    return tree_map(lambda arr: np.conj(arr), tree)
