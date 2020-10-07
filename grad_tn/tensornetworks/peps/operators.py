from abc import ABC, abstractmethod
from typing import List
from warnings import warn

import jax.numpy as np
from grad_tn.tensornetworks.peps import brute_force_contraction, bmps_contraction
from grad_tn.tensornetworks.peps.states import Peps, Ipeps
from jax import partial
from jax.ops import index_update, index
from utils.jax_utils.dtype_utils import complex_to_real
from utils.jax_utils.tree_util import tree_conj, tree_multimap, tree_map

# boundary condition keywords
PBC = 'pbc'
OBC = 'obc'
INFINITE = 'infinite'

# symmetry keywords
C4V = 'c4v'
VALID_SYMMETRIES = [None, C4V]


class Operator(ABC):
    @abstractmethod
    def expval(self, state, **kwargs):
        pass

    @abstractmethod
    def matrix_element(self, state1, state2, **kwargs):
        pass


class Pepo(Operator):
    """
    PEPO operator

    Attributes
    ----------
    lx : int
        horizontal system size
    ly : int
        vertical system size
    bc : {'obc', 'pbc'}
        boundary condition keyword
    hermitian : bool
        Whether the PEPO represents a hermitian operator.
        If so, expectation values are enforced to be real.
    d : int
        dimension of physical legs / local Hilbert space dimension
    """

    def __init__(self, tensors: List[List[np.ndarray]], bc: str, hermitian: bool = False):
        """
        An operator in PEPO form.

        Parameters
        ----------
        tensors : List of List of array
            the PEPO tensors. `tensors[x][y]` is the tensor at site (x, y)
            For leg convention see top of file
        bc : {'obc', 'pbc}
            Boundary condition keyword
        hermitian : bool
            Whether the PEPO represents a hermitian operator.
            If so, expectation values are enforced to be real.
        """

        # INPUT CHECKS
        d = tensors[0][0].shape[0]
        lx, ly = len(tensors), len(tensors[0])
        if bc not in [OBC, PBC]:
            raise ValueError(f'Unknown boundary condition keyword: {bc}')
        for col in tensors[1:]:
            if len(col) != ly:
                raise ValueError('Sub-lists of `tensors` need to have equal lengths')
        for x in range(lx):
            for y in range(ly):
                num_virt_legs = 4
                if bc != PBC:
                    num_virt_legs -= int(x == 0 or x == lx - 1) + int(y == 0 or y == ly - 1)
                if len(tensors[x][y].shape) != 2 + num_virt_legs:
                    raise ValueError(f'Illegal shape on tensors[{x}][{y}]. Expected {2 + num_virt_legs} legs. '
                                     f'Got {len(tensors[x][y].shape)}')
                if tensors[x][y].shape[:2] != (d, d):
                    raise ValueError(f'Inconsistent dimension of physical leg on tensors[{x}][{y}]')

        self.d = d
        self.bc = bc
        self._tensors = [sublist[:] for sublist in tensors]
        self.lx, self.ly = lx, ly
        self.hermitian = hermitian

    def get_tensors(self):
        return [sublist[:] for sublist in self._tensors]

    def expval(self, state, **contraction_options):
        """
        Expectation value of self in the state |state>

        Parameters
        ----------
        state : PEPS or SymmetricPEPS
            The state
        contraction_options : dict
            see Pepo.matrix_element

        Returns
        -------
        value : float or complex
            <state|self|state>.
            ``if self.hermitian``, output is enforced to be real
        """
        val = self.matrix_element(state, state, **contraction_options)

        if self.hermitian:
            val = complex_to_real(val)

        return val

    def matrix_element(self, state1, state2, **contraction_options):
        """

        Parameters
        ----------
        state1 : PEPS or SymmetricPEPS
            The 1st state. Will be used as bra.
        state2 : PEPS or SymmetricPEPS
            The 2nd state. Will be used as ket.
        contraction_options : dict
            'contraction_method' : {'brute', 'bmps'}
                the contraction method

            further options are method-specific.
            for brute-force contraction:
                no further options
            for bmps contraction:
                'chi_bmps' : maximum bond-dimension of boundary MPS
                'cutoff' : threshold for discarding small singular values

        Returns
        -------
        mat_elem : complex
            <state1|self|state2>
        """

        if not isinstance(state1, Peps):
            raise ValueError(f'Expected a PEPS object. Got {type(state1)} for state1')
        if not isinstance(state2, Peps):
            raise ValueError(f'Expected a PEPS object. Got {type(state2)} for state2')
        if not ((self.lx, self.ly) == (state1.lx, state1.ly) == (state2.lx, state2.ly)):
            raise ValueError('Incompatible System sizes')
        if not (self.d == state1.d == state2.d):
            raise ValueError('Incompatible local Hilbert space dimensions')
        if not (self.bc == state1.bc == state2.bc):
            raise ValueError('Incompatible boundary conditions')

        bra_tensors = tree_conj(state1.get_tensors())
        op_tensors = self.get_tensors()
        ket_tensors = state2.get_tensors()

        # CALL HELPER FUNCTION
        method = contraction_options['contraction_method']
        if method == 'brute':
            if self.bc == OBC:
                val = brute_force_contraction.obc.three_layers(bra_tensors, op_tensors, ket_tensors)
            elif self.bc == PBC:
                val = brute_force_contraction.pbc.three_layers(bra_tensors, op_tensors, ket_tensors)
            else:
                raise RuntimeError('Invalid BC. This should have been caught by the constructor...')

        elif method == 'bmps':
            if self.bc == OBC:
                warn('bMPS method is not yet tested')
                val = bmps_contraction.obc.three_layers(bra_tensors, op_tensors, ket_tensors, **contraction_options)
            elif self.bc == PBC:
                raise NotImplementedError
            else:
                raise RuntimeError('Invalid BC. This should have been caught by the constructor...')

        else:
            raise ValueError(f'Contraction method not supported: {method}')

        # NORMALISE
        normalisation = state1.norm(**contraction_options) * state2.norm(**contraction_options)

        return val / normalisation


class NnPepo(Operator):

    # TODO documentation

    def __init__(self, C, D, bc, lx, ly, hermitian=False, vL=None, vR=None, vB=None, vT=None):
        d, _, chi_i, _, chi_k, _ = D.shape
        # input checks
        if C.shape != (d, d, chi_i, chi_i):
            raise ValueError(f'Incompatible shape on tensor C. Expected {(d, d, chi_i, chi_i)}. Got {C.shape}')
        if D.shape != (d, d, chi_i, chi_i, chi_k, chi_k):
            raise ValueError(f'Illegal shape on tensor D. Expected {(d, d, chi_i, chi_i, chi_k, chi_k)}. Got {D.shape}')
        # FUTURE: special checks on boundary vectors for PBC?
        if (vL is not None) and (vL.shape != (chi_i,)):
            raise ValueError(f'Illegal shape on tensor vL. Expected {(chi_i,)}. Got {vL.shape}')
        if (vR is not None) and (vR.shape != (chi_i,)):
            raise ValueError(f'Illegal shape on tensor vR. Expected {(chi_i,)}. Got {vR.shape}')
        if (vB is not None) and (vB.shape != (chi_k,)):
            raise ValueError(f'Illegal shape on tensor vB. Expected {(chi_k,)}. Got {vB.shape}')
        if (vT is not None) and (vT.shape != (chi_k,)):
            raise ValueError(f'Illegal shape on tensor vT. Expected {(chi_k,)}. Got {vT.shape}')

        if bc == OBC:
            vL = vL if vL is not None else index_update(np.zeros([chi_i]), index[-1], 1.)
            vR = vR if vR is not None else index_update(np.zeros([chi_i]), index[0], 1.)
            vB = vB if vB is not None else index_update(np.zeros([chi_k]), index[-1], 1.)
            vT = vT if vT is not None else index_update(np.zeros([chi_k]), index[0], 1.)
            self.pepo_hor, self.pepo_vert = _parse_nn_pepo_obc(C, D, vL, vR, vB, vT, lx, ly)

        elif bc == PBC:
            raise NotImplementedError  # FUTURE implement

        else:
            raise ValueError(f'Unknown boundary conditions: {bc}')

        self.d = d
        self.bc = bc
        self.hermitian = hermitian

    def expval(self, state, **contraction_options):
        hor = self.pepo_hor.expval(state, **contraction_options)
        vert = self.pepo_vert.expval(state, **contraction_options)
        val = hor + vert
        if self.hermitian:
            val = complex_to_real(val)
        return val

    def matrix_element(self, state1, state2, **contraction_options):
        hor = self.pepo_hor.matrix_element(state1, state2, **contraction_options)
        vert = self.pepo_vert.matrix_element(state1, state2, **contraction_options)
        return hor + vert


# TODO implement
class Ipepo(Operator):

    def __init__(self, tensors, unit_cell, symmetry, hermitian):
        raise NotImplementedError

    def expval(self, state, **kwargs):
        assert isinstance(state, Ipepo)
        raise NotImplementedError

    def matrix_element(self, state1, state2, **kwargs):
        raise NotImplementedError


class SingleSiteOperator(Operator):
    """
    Operator that acts on a single site.
    This allows us to directly compute a PEPS for operator|ket> and use `PEPS.scalar_product` for contraction

    Attributes
    ----------
    d : int
        dimension of physical legs / local Hilbert space dimension
    tensor : jax.numpy.ndarray
        the operator (in matrix form). legs (p,p*)
    x : int
        the horizontal position of the operator
    y : int
        the vertical position of the operator
    hermitian : bool
        Whether the PEPO represents a hermitian operator.
        If so, expectation values are enforced to be real.
    """

    def __init__(self, tensor: np.ndarray, x: int, y: int, hermitian: bool = False):
        """

        Parameters
        ----------
        tensor : jax.numpy.ndarray
            the operator (in matrix form). legs (p,p*)
        x : int
            the horizontal position of the operator
        y : int
            the vertical position of the operator
        hermitian : bool
            Whether the PEPO represents a hermitian operator.
            If so, expectation values are enforced to be real.
        """
        d = tensor.shape[0]
        assert tensor.shape == (d, d)

        self.d = d
        self.tensor = tensor
        self.x = x
        self.y = y
        self.hermitian = hermitian

    def apply_to_peps(self, state):
        assert (0 <= self.x <= state.lx) and (0 <= self.y <= state.ly)
        assert self.d == state.d

        tensors = state.get_tensors()
        # (p,p*) & (p,...) -> (p,...)
        tensors[self.x][self.y] = np.tensordot(self.tensor, tensors[self.x][self.y], [1, 0])
        return state.with_different_tensors(tensors)

    def expval(self, state, **contraction_options):
        """
        Expectation value of self in the state |state>

        Parameters
        ----------
        state : PEPS
            The state

        Returns
        -------
        value : float or complex
            <state|self|state>.
            ``if self.hermitian``, output is enforced to be real
        """

        if isinstance(state, Peps):
            assert (0 <= self.x <= state.lx) and (0 <= self.y <= state.ly)
        else:
            assert (0 <= self.x <= state.unit_cell_lx) and (0 <= self.y <= state.unit_cell_ly)
        assert self.d == state.d

        if isinstance(state, Peps):
            res = state.scalar_product(self.apply_to_peps(state), **contraction_options) \
                  / state.norm_squared(**contraction_options)
        elif isinstance(state, Ipeps):
            op = LocalOperator(self.tensor, geometry=[[0]], hermitian=self.hermitian)
            return op.expval(state, **contraction_options)
        else:
            raise ValueError(f'Expected PEPS or IPEPS state. Got {type(state)}')

        if self.hermitian:
            res = complex_to_real(res)

        return res

    def matrix_element(self, state1, state2, **contraction_options):
        """
        Matrix element <state1|self|state2>

        Parameters
        ----------
        state1 : PEPS
            The 1st state. Will be used as bra.
        state2 : PEPS
            The 2nd state. Will be used as ket.

        Returns
        -------
        mat_elem : complex
            <state1|self|state2>
        """
        if isinstance(state1, Peps) and isinstance(state2, Peps):
            assert (state1.lx, state1.ly) == (state2.lx, state2.ly)
            assert (0 <= self.x <= state1.lx) and (0 <= self.y <= state1.ly)
        elif isinstance(state1, Ipeps) and isinstance(state2, Ipeps):
            assert (state1.unit_cell_lx, state1.unit_cell_ly) == (state2.unit_cell_lx, state2.unit_cell_ly)
            assert (0 <= self.x <= state1.unit_cell_lx) and (0 <= self.y <= state1.unit_cell_ly)
        assert self.d == state1.d == state2.d

        if isinstance(state1, Peps) and isinstance(state2, Peps):
            res = state1.scalar_product(self.apply_to_peps(state2), **contraction_options) \
                  / state1.norm(**contraction_options) / state2.norm(**contraction_options)
        elif isinstance(state1, Ipeps) and isinstance(state2, Ipeps):
            op = LocalOperator(self.tensor, geometry=[[0]], hermitian=self.hermitian)
            return op.matrix_element(state1, state2, **contraction_options)
        else:
            raise ValueError(f'Expected two PEPS or two iPEPS states. Got {type(state1)} and {type(state2)}.')

        return res


class LocalOperator(Operator):
    """
    Operator that acts on a few sites. intended for iPEPS

    Attributes
    ----------
    tensor : jax.numpy.ndarray
        the operator. legs (p1,p2,...,pn,p1*,p2*,...,pn*)
    geometry : jax.numpy.ndarray
        `operator_geometry[x, y] == i` -> leg pi of `operator` is applied at site (x, y)
    hermitian : bool
        Whether the PEPO represents a hermitian operator.
        If so, expectation values are enforced to be real.
    """

    def __init__(self, tensor, geometry, hermitian=False):
        # DOC
        d = tensor.shape[0]
        n_sites = len(tensor.shape) // 2
        geometry = np.asarray(geometry, dtype=np.int64)

        # input checks
        assert tensor.shape == (d, d) * n_sites
        assert all([np.sum(geometry == n) == 1 for n in range(n_sites)])
        geometry = np.asarray(geometry)
        tensor = np.asarray(tensor)

        self.tensor = tensor
        self.geometry = geometry
        self.hermitian = hermitian

    def expval(self, state, **contraction_options):
        # DOC

        if isinstance(state, Peps):
            raise NotImplementedError

        elif isinstance(state, Ipeps):
            env = state.get_env(**contraction_options)
            return env.expval(self.tensor, self.geometry, self.hermitian)

        else:
            raise ValueError(f'Expected PEPS or iPEPS state. Got {type(state)}')

    def matrix_element(self, state1, state2, **contraction_options):
        # DOC
        if isinstance(state1, Peps) and isinstance(state2, Peps):
            raise NotImplementedError  # would need to generate the ncon connects...

        elif isinstance(state1, Ipeps) and isinstance(state2, Ipeps):
            raise NotImplementedError  # dont know how to normalise this

        else:
            raise ValueError(f'Expected two PEPS or two iPEPS states. Got {type(state1)} and {type(state2)}')


def product_pepo(local_operator, chi: int, bc: str, lx: int, ly: int, dtype=None, hermitian=False):
    """
    Creates a PEPO that represents a product operator (tensor_product) of the same `local_operator` on all sites

    Parameters
    ----------
    local_operator : jax.numpy.ndarray
        The local operator. 2D array. legs (p,p*)
    chi : int
        The PEPO bond-dimension
    bc : {'obc', 'pbc'}
        Boundary condition keyword
    lx : int
        horizontal system size
    ly : int
        vertical system size
    dtype : jax.numpy.dtype
        dtype for the tensors. per default, uses the dtype of `local_operator`
    hermitian : bool
        Whether the PEPO represents a hermitian operator.
        If so, expectation values are enforced to be real.

    Returns
    -------
    operator : PEPO
    """

    if bc not in [OBC, PBC]:
        raise ValueError(f'Unknown/Unsupported boundary-condition keyword: {bc}')

    dtype = local_operator.dtype if dtype is None else dtype
    d = local_operator.shape[0]
    assert local_operator.shape == (d, d)

    tensor_bulk = index_update(np.zeros([d, d, chi, chi, chi, chi], dtype), index[:, :, 0, 0, 0, 0], local_operator)

    if bc == OBC:
        tensor_corner = index_update(np.zeros([d, d, chi, chi], dtype), index[:, :, 0, 0], local_operator)
        tensor_edge = index_update(np.zeros([d, d, chi, chi, chi], dtype), index[:, :, 0, 0, 0], local_operator)

        tensors = [[tensor_corner] + [tensor_edge for _ in range(1, ly - 1)] + [tensor_corner]] \
                  + [[tensor_edge] + [tensor_bulk for _ in range(1, ly - 1)]
                     + [tensor_edge] for _ in range(1, lx - 1)] \
                  + [[tensor_corner] + [tensor_edge for _ in range(1, ly - 1)] + [tensor_corner]]
    else:
        tensors = [[tensor_bulk for _ in range(ly)] for _ in range(lx)]

    pepo = Pepo(tensors, bc=bc, hermitian=hermitian)
    return pepo


def expval_snapshot(state, operator, hermitian, **contraction_options):
    """
    expectation values of an on-site operator <state|operator|state>, for all sites

    Parameters
    ----------
    state : PEPS
    operator : jax.numpy.ndarray
        2D array. legs (p,p*)
    hermitian : bool
        whether the operator is hermitian

    Returns
    -------
    values : jax.numpy.ndarray
        `value[x][y]` is <state| operator_{x,y} |state>
        If `hermitian`, output is enforced to be real
    """
    xs = [[x for _ in range(state.ly)] for x in range(state.lx)]
    ys = [[y for y in range(state.ly)] for _ in range(state.lx)]

    def get_value(x, y):
        return SingleSiteOperator(operator, x, y, hermitian).expval(state, **contraction_options)

    values = tree_multimap(get_value, xs, ys)
    return np.array(values)


def expval_average(state, operator, hermitian, **contraction_options):
    """
    expectation value of an on-site operator <state|operator|state>, averaged over all sites

    Parameters
    ----------
    state : PEPS
    operator : jax.numpy.ndarray
        2D array. legs (p,p*)
    hermitian : bool
        whether the operator is hermitian

    Returns
    -------
    value : float or complex
        If `hermitian`, output is enforced to be real
    """
    snap = expval_snapshot(state, operator, hermitian, **contraction_options)
    return np.average(snap)


def correlator_timeslice(state, evolved_quenched_state, quench_operator, state_energy, t, **contraction_options):
    """
    Convenience function

    Correlation function <A(t)A(0)> where < . > is the expval w.r.t `state` which NEEDS to be an energy-eigenstate
    for all positions of A(t)  (output array dimensions)
    where the position of A(0) is implicitly defined via `evolved_quenched`
    A is `quench_operator`

    Computed as:    exp(i E t) * <state| A |quenched_evolved>
    where H |state> = E |state>

    Parameters
    ----------
    state : PEPS or IPEPS or RoiPEPS
        Assumed to be an eigenstate of H
    evolved_quenched_state : PEPS or IPEPS or RoiPEPS
        PEPS for exp(-iHt)A|state>
    quench_operator : jax.numpy.ndarray
        the operator A
    state_energy : float, optional
        the energy <state|H|state>
    t : float, optional
        the time of A(t)

    Returns
    -------
    correlators : jax.numpy.ndarray
        `correlators[x,y]` is <A(t)A(0)> for A(t) acting on site (x,y)
    """
    xs = [[x for _ in range(state.ly)] for x in range(state.lx)]
    ys = [[y for y in range(state.ly)] for _ in range(state.lx)]

    if (state_energy is None) or (t is None):
        prefactor = 1.
    else:
        prefactor = np.exp(1.j * state_energy * t)

    def get_value(x, y):
        op = SingleSiteOperator(quench_operator, x, y)
        return prefactor * op.matrix_element(state, evolved_quenched_state, **contraction_options)

    # TODO can we vmap instead of tree_multimap?

    values = tree_multimap(get_value, xs, ys)
    arr = np.array(values)
    return arr


def approximate_pepo_peps(operator: Pepo, state: Peps) -> Peps:
    """
    CRUDE approximation of operator applied to state
    as a PEPS of same bond-dimension as state.
    Intended as an initial guess for optimisation

    Parameters
    ----------
    operator
    state

    Returns
    -------

    """
    lx, ly = operator.lx, operator.ly
    assert (lx, ly) == (state.lx, state.ly)

    op_tensors = operator.get_tensors()
    state_tensors = state.get_tensors()
    num_virt_legs = [[2] + [3] * (ly - 2) + [2]] \
                    + [[3] + [4] * (ly - 2) + [3]] * (lx - 2) \
                    + [[2] + [3] * (ly - 2) + [2]]

    chi = state_tensors[0][0].shape[0]
    new_tensors = tree_multimap(_contract, op_tensors, state_tensors, num_virt_legs)

    right_legs = [[1] + [2] * (ly - 2) + [2]] \
                 + [[1] + [4] * (ly - 2) + [3]] * (lx - 2) \
                 + [[None] + [None] * (ly - 2) + [None]]
    left_legs = [[None] + [None] * (ly - 2) + [None]] \
                + [[3] + [2] * (ly - 2) + [1]] * (lx - 2) \
                + [[2] + [2] * (ly - 2) + [1]]
    up_legs = [[2] + [3] * (ly - 2) + [None]] \
              + [[2] + [1] * (ly - 2) + [None]] * (lx - 2) \
              + [[1] + [1] * (ly - 2) + [None]]
    down_legs = [[None] + [1] * (ly - 2) + [1]] \
                + [[None] + [3] * (ly - 2) + [2]] * (lx - 2) \
                + [[None] + [3] * (ly - 2) + [2]]

    # compress horizontal bonds (first even than odd)
    for x in list(range(0, lx - 1, 2)) + list(range(1, lx - 1, 2)):
        for y in range(ly):
            new_tensors[x][y], new_tensors[x + 1][y] = _compress_bond(new_tensors[x][y], new_tensors[x + 1][y], chi,
                                                                      axes=(right_legs[x][y], left_legs[x + 1][y]))
    # compress vertical bonds (first even than odd)
    for y in list(range(0, ly - 1, 2)) + list(range(1, ly - 1, 2)):
        for x in range(lx):
            new_tensors[x][y], new_tensors[x][y + 1] = _compress_bond(new_tensors[x][y], new_tensors[x][y + 1], chi,
                                                                      axes=(up_legs[x][y], down_legs[x][y + 1]))

    return state.with_different_tensors(new_tensors)


def absorb_u_site(state: Peps, u_site: np.ndarray) -> Peps:
    tensors = state.get_tensors()
    # (p,...) & (p,p*) -> (p,...)
    tensors = tree_map(lambda arr: np.tensordot(arr, u_site, [0, 1]), tensors)
    return state.with_different_tensors(tensors)


def _parse_nn_pepo_obc(C, D, vL, vR, vB, vT, lx, ly):
    assert (lx > 2) and (ly > 2)  # (otherwise there is no bulk, to put the Ds in)
    x_d = lx // 2
    y_d = ly // 2

    # HORIZONTAL
    vL_C = np.tensordot(vL, C, [0, 2])  # (p,p*,r)
    C_vR = np.tensordot(vR, C, [0, 3])  # (p,p*,l)
    vB_D = np.tensordot(vB, D, [0, 4])  # (p,p*,l,r,u)
    D_vT = np.tensordot(vT, D, [0, 5])  # (p,p*,l,r,d)

    left_col = [vL_C[:, :, :, None]] + [vL_C[:, :, None, :, None]] * (ly - 2) + [vL_C[:, :, None, :]]

    # bottom C:  (p,p*,i,j) = (p,p*,l,r) -> (p,p*,r,l) -> (p,p*,r,u,l)
    # bulk C: (p,p*,i,j) = (p,p*,l,r) -> (p,p*,u,l,d,r)
    # top C: (p,p*,i,j) = (p,p*,l,r) -> (p,p*,l,d,r)
    mid_col = [np.transpose(C, [0, 1, 3, 2])[:, :, :, None, :]] \
              + [C[:, :, None, :, None, :]] * (ly - 2) \
              + [C[:, :, :, None, :]]

    # vB_D: (p,p*,ijl) = (p,p*,lru) -> (p,p*,rul)
    # D: (p,p*,ijkl) -> (p,p*,likj) = (p,p*,uldr)
    # D_vT: (p,p*,ijk) = (p,p*,lrd) -> (p,p*,ldr)
    d_col = [np.transpose(vB_D, [0, 1, 3, 4, 2])] \
            + [np.transpose(D, [0, 1, 5, 2, 4, 3])] * (ly - 2) \
            + [np.transpose(D_vT, [0, 1, 2, 4, 3])]

    right_col = [C_vR[:, :, None, :]] + [C_vR[:, :, None, :, None]] * (ly - 2) + [C_vR[:, :, :, None]]
    tensors = [left_col] \
              + [mid_col] * (x_d - 1) \
              + [d_col] \
              + [mid_col] * (lx - x_d - 2) \
              + [right_col]
    pepo_hor = Pepo(tensors, OBC, False)  # even if the NnPepo is hermitian, the two separate Pepos could be not.

    # VERTICAL
    # rotate tensors clockwise

    # (p,p*,u,l,d,r) -> (p,p*,l,d,r,u)
    _rotate90 = partial(np.transpose, axes=[0, 1, 3, 4, 5, 2])

    # tensor at new location (x,y) was at (-y-1,x) before
    tensors = [[tensors[-y - 1][0] for y in range(ly)]] \
              + [[tensors[-1][x]] + [_rotate90(tensors[-y - 1][x]) for y in range(1, ly - 1)]
                 + [tensors[0][x]] for x in range(1, lx - 1)] \
              + [[tensors[-y - 1][-1] for y in range(ly)]]

    pepo_vert = Pepo(tensors, OBC, False)  # even if the NnPepo is hermitian, the two separate Pepos could be not.

    return pepo_hor, pepo_vert


# FIXME needed?
def _contract(op_tens, state_tens, _num_virt_legs):
    # (p,p*,...) & (p,...) -> (p,...,...)
    res = np.tensordot(op_tens, state_tens, [1, 0])

    if _num_virt_legs == 2:
        _p, _a, _b, _a_, _b_ = res.shape
        # (p,a,b,A,B) -> (p,a,A,b,B) -> (p,A,B)
        res = np.reshape(np.transpose(res, (0, 1, 3, 2, 4)), (_p, _a * _a_, _b * _b_))
    elif _num_virt_legs == 3:
        _p, _a, _b, _c, _a_, _b_, _c_ = res.shape
        # (p,a,b,c,A,B,C) -> (p,a,A,b,B,c,C) -> (p,A,B,C)
        res = np.reshape(np.transpose(res, (0, 1, 4, 2, 5, 3, 6)),
                         (_p, _a * _a_, _b * _b_, _c * _c_))
    elif _num_virt_legs == 4:
        _p, _a, _b, _c, _d, _a_, _b_, _c_, _d_ = res.shape
        # (p,a,b,c,d,A,B,C,D) -> (p,a,A,b,B,c,C,d,D) -> (p,A,B,C,D)
        res = np.reshape(np.transpose(res, (0, 1, 5, 2, 6, 3, 7, 4, 8)),
                         (_p, _a * _a_, _b * _b_, _c * _c_, _d * _d_))
    else:
        raise ValueError

    return res


# FIXME needed?
def _compress_bond(a, b, chi, axes):
    a_leg, b_leg = axes
    u_leg_dims = a.shape[:a_leg] + a.shape[a_leg + 1:]
    vh_leg_dims = b.shape[:b_leg] + b.shape[b_leg + 1:]
    assert isinstance(a_leg, int) and isinstance(b_leg, int)
    theta = np.tensordot(a, b, (a_leg, b_leg))
    U_mat, S, Vh_mat = np.linalg.svd(np.reshape(theta, (np.prod(u_leg_dims), np.prod(vh_leg_dims))),
                                     full_matrices=False)
    k = len(S)

    if k < chi:
        raise ValueError

    if k == chi:
        pass  # TODO warn
    else:
        U_mat = U_mat[:, :chi]
        S = S[:chi]
        Vh_mat = Vh_mat[:chi, :]

    # absorb singular values symmetrically
    U_mat = U_mat * np.sqrt(S[None, :])
    Vh_mat = Vh_mat * np.sqrt(S[:, None])

    # reshape to tensor legs
    U = np.reshape(U_mat, u_leg_dims + (chi,))
    Vh = np.reshape(Vh_mat, (chi,) + vh_leg_dims)

    # transpose to get common leg to correct position again
    u_transpose = list(range(len(u_leg_dims)))
    U = np.transpose(U, u_transpose[:a_leg] + [len(u_leg_dims)] + u_transpose[a_leg:])

    vh_transpose = list(range(1, len(vh_leg_dims) + 1))
    Vh = np.transpose(Vh, vh_transpose[:b_leg] + [0] + vh_transpose[b_leg:])

    return U, Vh
