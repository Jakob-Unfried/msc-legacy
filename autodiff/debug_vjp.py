import pdb
import warnings

from jax import custom_vjp


@custom_vjp
def debug_identity(x):
    """
    acts as identity, but inserts a pdb trace on the backwards pass
    """
    warnings.warn('Using a module intended for debugging')
    return x


def _debug_fwd(x):
    warnings.warn('Using a module intended for debugging')
    return x, x


# noinspection PyUnusedLocal
def _debug_bwd(x, g):
    pdb.set_trace()
    return g


debug_identity.defvjp(_debug_fwd, _debug_bwd)
