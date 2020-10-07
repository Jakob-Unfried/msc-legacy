import jax.numpy as np
from jax import tree_map


def complex_to_real(z, ignore_im_part=False, ignore_nan=False, rtol=1e-05, atol=1e-08):
    """
    converts a complex datastructure to its real parts and applies checks
    imagine tree_map(np.real, z) but with extra checks

    Parameters
    ----------
    z
        the complex datastructue
    ignore_im_part : bool
        whether significant imaginary parts should be ignored or cause an error
    ignore_nan : bool
        whether nans should be ignored or cause an error
    rtol : float
        the relative tolerance for considering the imaginary part significant
    atol : float
        the absolute tolerance for considering the imaginary part significant
        defaults to the default of np.allclose

    Returns
    -------
    x
        datastructure of same type as z, but with real numbers
    """

    def fun_on_leaf(_z):
        if np.isnan(_z):
            if not ignore_nan:
                raise ValueError('NaN encountered')
            return np.real(_z)

        _z_re = np.real(_z)

        if not ignore_im_part:
            if not np.allclose(_z_re, _z_re + np.imag(_z), rtol=rtol, atol=atol):
                raise ValueError('Significant imaginary part encountered where it was not expected')

        return _z_re

    return tree_map(fun_on_leaf, z)
