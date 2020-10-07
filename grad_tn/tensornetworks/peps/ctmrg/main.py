from typing import List, Optional

import jax.numpy as np
from grad_tn.tensornetworks.peps.ctmrg import ti_c4v, ti
from utils.jax_utils.dtype_utils import complex_to_real

# symmetry keywords
C4V = 'c4v'
NO_SYMM = None


# TODO list
#  - fix gauge and use fixed-point VJP rule


def default_chi_ctm(ipeps):
    """
    default CTM bond dimension for an Ipeps state
    """
    return 3 * (ipeps.chi ** 2)


class CtmrgEnvironment:
    # TODO avoid overhead: save partial contraction (i.e. reduced density matrix) to a dict field

    def __init__(self, env_tensors: List[np.ndarray], ipeps_tensors: List[np.ndarray],
                 unit_cell: np.ndarray, symmetry: Optional[str]):

        if type(ipeps_tensors) != list:
            ipeps_tensors = [ipeps_tensors]
        num_tensors = len(ipeps_tensors)
        assert np.all(0 <= unit_cell < num_tensors)

        self.env_tensors = env_tensors
        self.ipeps_tensors = ipeps_tensors
        self.unit_cell = unit_cell
        self.symmetry = symmetry

    def expval(self, operator: np.ndarray, operator_geometry: np.ndarray, hermitian: bool = False):
        if self.unit_cell.shape == (1, 1):
            if self.symmetry == C4V:
                val = ti_c4v.expval(self.env_tensors, self.ipeps_tensors, operator, operator_geometry)
            else:
                val = ti.expval(self.env_tensors, self.ipeps_tensors, operator, operator_geometry)
        else:
            raise NotImplementedError

        if hermitian:
            val = complex_to_real(val)

        return val


def get_environment(ipeps_tensors, unit_cell, symmetry,
                    chi_ctm=None, bvar_threshold=1e-12, max_iter=100):
    # input checks
    if type(ipeps_tensors) != list:
        ipeps_tensors = [ipeps_tensors]
    num_tensors = len(ipeps_tensors)
    assert np.all(0 <= unit_cell < num_tensors)

    if unit_cell.shape == (1, 1):
        if symmetry == C4V:
            env_tensors = ti_c4v.get_env(ipeps_tensors, chi_ctm, bvar_threshold, max_iter)
            return CtmrgEnvironment(env_tensors, ipeps_tensors, unit_cell, symmetry)
        else:
            env_tensors = ti.get_env(ipeps_tensors, chi_ctm, bvar_threshold, max_iter)
            return CtmrgEnvironment(env_tensors, ipeps_tensors, unit_cell, symmetry)
    else:
        raise NotImplementedError
