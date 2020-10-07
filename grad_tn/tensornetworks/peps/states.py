from abc import ABC, abstractmethod
from typing import List, Optional, Union

from warnings import warn

import jax.numpy as np
from grad_tn.tensornetworks.peps import ctmrg, bmps_contraction, brute_force_contraction
from jax.ops import index_update, index
from utils.jax_utils.dtype_utils import complex_to_real
from utils.jax_utils.random_arrays import uniform
from utils.jax_utils.tree_util import tree_conj, tree_all, tree_map, tree_multimap, tree_max_abs


# boundary condition keywords
PBC = 'pbc'
OBC = 'obc'
INFINITE = 'infinite'

# symmetry keywords (for iPEPS)
C4V = 'c4v'
NO_SYMM = None

# leg labelling and order conventions:
"""
--------------------------
PEPS / PEPO leg convention
--------------------------

- labels:   p = physical leg (of a ket / ket-like index of an operator)
            p* = physical leg (of a bra / bra-like index of an operator)
            u = up virtual leg
            l = left virtual leg
            d = down virtual leg
            r = right virtual leg
- physical legs go first.
- in the bulk, the virtual legs are ordered counter-clockwise, starting at the top. e.g. (p,u,l,d,r)
- at the top right corner and top edge, the virtual legs are also ordered counter-clockwise, starting at the top
- the other corners and edges are rotated versions of this, such that the order is always counter-clockwise, but the
    starting points may differ
- Illustration (for a state, e.g. PEPS, in general with OBC)
    
    ┌───┐                          ┌───┐                          ┌───┐    
    │   ├── r                  l ──┤   ├── r                  l ──┤   │
    └─┬─┘                          └─┬─┘                          └─┬─┘
      │    (p,d,r)                   │    (p,l,d,r)                 │    (p,l,d)
      d                              d                              d
      
      
      
      u                              u                              u
      │                              │                              │
    ┌─┴─┐                          ┌─┴─┐                          ┌─┴─┐
    │   ├──  r                 l ──┤   ├── r                  l ──┤   │
    └─┬─┘                          └─┬─┘                          └─┬─┘
      │    (p,d,r,u)                 │    (p,u,l,d,r)               │    (p,u,l,d)
      d                              d                              d
      
      
      
      u                              u                              u
      │                              │                              │
    ┌─┴─┐                          ┌─┴─┐                          ┌─┴─┐
    │   ├──  r                 l ──┤   ├── r                  l ──┤   │
    └───┘                          └───┘                          └───┘
           (p,r,u)                      (p,r,u,l)                    (p,u,l)
           
- for PBC and iPEPS, all tensors are like the bulk one, (p,u,l,d,r)
- for PEPO operators, we get an additional p* leg, e.g. (p,p*,u,l,d,r) in the bulk

"""


# TODO list
#  -


class State(ABC):
    @abstractmethod
    def get_tensors(self):
        pass

    @abstractmethod
    def norm_squared(self):
        pass

    @abstractmethod
    def norm(self):
        pass

    @abstractmethod
    def scalar_product(self, other, **kwargs):
        pass

    @abstractmethod
    def with_different_tensors(self, tensors):
        pass

    @abstractmethod
    def expval(self, operator, **kwargs):
        pass


class Peps(State):
    """
    PEPS state

    Attributes
    ----------
    lx : int
        horizontal system size
    ly : int
        vertical system size
    bc : {'obc', 'pbc'}
        boundary condition keyword
    d : int
        local Hilbert-space dimension

    """

    def __init__(self, tensors: List[List[np.ndarray]], bc: str):
        """
        Parameters
        ----------
        tensors : List of List of jax.numpy.ndarray
            the PEPS tensors. `tensors[x][y]` is the tensor at site (x, y)
            For leg convention see top of file
        bc : {'obc', 'pbc'}
            boundary condition keyword
        """

        # INPUT CHECKS
        d = tensors[0][0].shape[0]
        lx, ly = len(tensors), len(tensors[0])
        for col in tensors[1:]:
            if len(col) != ly:
                raise ValueError('Sub-lists of `tensors` need to have equal lengths')
        if bc not in [OBC, PBC]:
            raise ValueError(f'Unknown/Unsupported boundary-condition keyword: {bc}')
        for x in range(lx):
            for y in range(ly):
                num_virt_legs = 4
                if bc != PBC:
                    num_virt_legs -= int(x == 0 or x == lx - 1) + int(y == 0 or y == ly - 1)
                if len(tensors[x][y].shape) != 1 + num_virt_legs:
                    raise ValueError(f'Illegal shape on tensors[{x}][{y}]. Expected {1 + num_virt_legs} legs. '
                                     f'Got {len(tensors[x][y].shape)}')
                if tensors[x][y].shape[0] != d:
                    raise ValueError(f'Inconsistent dimension of physical leg on tensors[{x}][{y}]')

        # SET FIELDS
        self._tensors = [sublist[:] for sublist in tensors]
        self.d = d
        self.bc = bc
        self.lx, self.ly = lx, ly
        self._norm_squared = None

    def get_tensors(self):
        # return "deep" copy to avoid list-mutation of the class fields
        return [sublist[:] for sublist in self._tensors]

    def get_tensor(self, x, y):
        return self._tensors[x][y]

    def norm_squared(self, **contraction_options):
        """
        returns the squared norm <self|self>.
        enforces it to be real. only recomputes if necessary.

        Parameters
        ----------
        contraction_options : dict
            see Peps.scalar_product

        Returns
        -------
        norm_sq : float
            <self|self>
        """
        if self._norm_squared is not None:
            return self._norm_squared

        self_prod = self.scalar_product(self, **contraction_options)
        self_prod = complex_to_real(self_prod)
        self._norm_squared = self_prod
        return self_prod

    def norm(self, **contraction_options):
        # FIXME DOC options
        """
        returns the norm sqrt(<self|self>).
        see `PEPS.norm_squared`

        Parameters
        ----------
        contraction_options : dict
            see Peps.scalar_product

        Returns
        -------
        _norm : float
            sqrt(<self|self>)
        """
        return np.sqrt(self.norm_squared(**contraction_options))

    def scalar_product(self, other, **contraction_options):
        # FIXME DOC options
        """
        the scalar product <self|other>

        Parameters
        ----------
        other : Peps
            the other state
        contraction_options : dict
            'method' : {'brute', 'bmps'}
                the contraction method

            further options are method-specific.
            for brute-force contraction:
                no further options
            for bmps contraction:
                'chi_bmps' : maximum bond-dimension of boundary MPS
                'cutoff' : threshold for discarding small singular values

        Returns
        -------
        prod : complex
            <self|other>
        """

        # INPUT CHECKS
        if not isinstance(other, Peps):
            raise ValueError(f'Expected a PEPS object. Got {type(other)}')
        if (self.lx, self.ly) != (other.lx, other.ly):
            raise ValueError(f'Incompatible system sizes: {(self.lx, self.ly)} and {(other.lx, other.ly)}.')
        if self.d != other.d:
            raise ValueError(f'Incompatible dimensions of local Hilbert spaces: {self.d} and {other.d}')
        if self.bc != other.bc:
            raise ValueError(f'Incompatible boundary conditions: {self.bc} and {other.bc}')

        # CALL HELPER FUNCTION
        method = contraction_options['contraction_method']
        if method == 'brute':
            if self.bc == OBC:
                return brute_force_contraction.obc.two_layers(tree_conj(self.get_tensors()), other.get_tensors())
            elif self.bc == PBC:
                return brute_force_contraction.pbc.two_layers(tree_conj(self.get_tensors()), other.get_tensors())
            else:
                raise RuntimeError('Invalid BC. This should have been caught by the constructor...')

        elif method == 'bmps':
            if self.bc == OBC:
                warn('bMPS method is not yet tested')
                return bmps_contraction.obc.two_layers(tree_conj(self.get_tensors()), other.get_tensors(),
                                                       **contraction_options)
            elif self.bc == PBC:
                raise NotImplementedError
            else:
                raise RuntimeError('Invalid BC. This should have been caught by the constructor...')

        else:
            raise ValueError(f'Contraction method not supported: {method}')

    def with_different_tensors(self, tensors):
        if (self.lx, self.ly) != (len(tensors), len(tensors[0])):
            raise ValueError('Incompatible system size')
        if not tree_all(tree_multimap(lambda a1, a2: a1.shape == a2.shape, tensors, self._tensors)):
            raise ValueError('Incompatible tensor shapes')
        return Peps(tensors, self.bc)

    def expval(self, operator, **contraction_options):
        # Expectation values are handled by the operator classes
        return operator.expval(self, **contraction_options)

    def absorb_factor(self, factor):
        """
        Absorbs a `factor` into self, such that
        |self_new> = factor * |self_old>

        The factor is absorbed into the center tensor, to preserve symmetry as best as possible
        """
        self._tensors[self.lx // 2][self.ly // 2] *= factor
        if self._norm_squared:
            self._norm_squared *= (np.abs(factor) ** 2)


class Ipeps(State):
    """
    infinite PEPS (iPEPS) state

    Attributes
    ----------
    unit_cell : jax.numpy.ndarray
        unit_cell : 2D ndarray, optional
        Encoded unit cell. Entries are list-indices of `tensors`.
        `self._tensors[unit_cell[x, y]]` is the tensor at site (x, y) of the unit cell.
    unit_cell_lx : int
        i.e. `unit_cell.shape[0]`
    unit_cell_ly : int
        i.e. `unit_cell.shape[1]`
    symmetry : {None, 'c4v'}
        a keyword for the symmetry of the state
    """

    def __init__(self, tensors: List[np.ndarray],
                 unit_cell: Optional[Union[np.ndarray, List[List[int]]]],
                 symmetry: Optional[str]):
        """
        Parameters
        ----------
        tensors : List of jax.numpy.ndarray
            The iPEPS tensors
        unit_cell : jax.numpy.ndarray, optional
            2D array. Encoded unit cell. Entries are indices of `tensors`.
            `tensors[unit_cell[x, y]]` is the tensor at site (x, y) of the unit cell.
            Default: None, which means single-site unit cell -> translationally invariant
        symmetry : {None, 'c4v'}
            a keyword for the symmetry of the state
        """

        unit_cell = np.asarray(unit_cell)

        # INPUT CHECKS
        if symmetry not in [C4V, NO_SYMM]:
            raise ValueError(f'Unknown/Unsupported symmetry: {symmetry}')
        # FUTURE : check that unit_cell is compatible with symmetry
        if not unit_cell:
            unit_cell = np.zeros([1, 1], dtype=np.int64)
        if not np.issubdtype(unit_cell.dtype, np.integer):
            raise ValueError(f'Unit cell must be an integer array, got {unit_cell.dtype}')
        n_tensors = len(tensors)
        if not (np.all(0 <= unit_cell) and np.all(unit_cell < n_tensors)):
            raise ValueError(f'Illegal values in unit-cell')
        if not all([np.sum(unit_cell == n) > 0 for n in range(n_tensors)]):
            raise ValueError(f'Illegal unit-cell: unused tensors')
        if not all([len(arr.shape) == 5 for arr in tensors]):
            raise ValueError(f'Illegal shape on tensors')

        # symmetrise if necessary
        if symmetry == C4V:
            # FUTURE: for non-trivial unit-cell this needs more work (tensors are not indiviually symmetric...)
            tensors = tree_map(c4v_symmetrise, tensors)

        # SET FIELDS
        self._tensors = tensors[:]
        self._envs = {}  # ctmrg environments can be saved to this dict to avoid recomputation
        self.symmetry = symmetry
        self.unit_cell = unit_cell
        self.unit_cell_lx, self.unit_cell_ly = unit_cell.shape
        self.d = tensors[0].shape[0]
        self.chi = tensors[0].shape[1]

    def get_tensor(self, x: int, y: int):
        """
        Return the tensor at site (x, y)

        Parameters
        ----------
        x : int
        y : int

        Returns
        -------
        tens : jax.numpy.ndarray

        """
        x = x % self.unit_cell_lx
        y = y % self.unit_cell_ly
        return self._tensors[self.unit_cell[x, y]]

    def get_tensors(self):
        return self._tensors[:]

    def with_different_tensors(self, tensors: List[np.ndarray]):
        # TODO possible source for errors: if `tensors` dont have the symmetry of `self`
        assert len(self._tensors) == len(tensors)
        assert tree_all(tree_multimap(lambda a1, a2: a1.shape == a2.shape, self._tensors, tensors))
        return Ipeps(tensors, self.unit_cell, self.symmetry)

    def get_env(self, **contraction_options):
        """
        get CTMRG environment for iPEPS state `self`.
        Avoids recomputation.

        Parameters
        ----------
        contraction_options : dict
            'chi_ctm' : int, the CTM bond dimension
            'bvar_threshold' : float, threshold for boundary variance as a convergence criterion
            'max_iter' : int, maximum number of renormalisation iterations
            TODO find good defaults, 1e-12 seems too low

        Returns
        -------
        env : CtmrgEnvironment

        """

        if 'chi_ctm' in contraction_options:
            chi_ctm = contraction_options['chi_ctm']
        else:
            chi_ctm = ctmrg.default_chi_ctm(self)
            contraction_options = contraction_options.copy()
            contraction_options['chi_ctm'] = chi_ctm

        if chi_ctm in self._envs:
            return self._envs[chi_ctm]

        env = ctmrg.get_environment(self.get_tensors(), self.unit_cell, self.symmetry, **contraction_options)
        self._envs[chi_ctm] = env
        return env

    def expval(self, operator, **contraction_options):
        # Expectation values are handled by opertor classes
        return operator.expval(self, **contraction_options)

    def scalar_product(self, other, **kwargs):
        raise NotImplementedError  # FUTURE how to do this?

    def norm_squared(self):
        raise NotImplementedError  # FUTURE how to do this?

    def norm(self):
        raise NotImplementedError  # FUTURE how to do this?


def product_peps(chi: int, bc: str, state: Union[np.ndarray, List[List[np.ndarray]]],
                 dtype: Optional[np.dtype] = None, lx: Optional[int] = None, ly: Optional[int] = None):
    """
    Peps for a product state

    Parameters
    ----------
    chi : int
        the bond dimension
    bc : {'obc', 'pbc'}
        boundary conditions
    state : (array) or (List of List of array)
        Either a single 1D array to be used as the local state everywhere,
            optional kwargs `lx` and `ly` are required for this option
        Or a list of lists of 1D arrays, such that `state[x][y]` is the local state at site (x,y).
    dtype : optional
        datatype for the Peps tensors, per default use the dtype of `state`
    lx : int, optional
        System size (horizontal)
    ly : int, optional
        System size (vertical)

    Returns
    -------
    state : Peps
    """

    # set lx, ly, dtype, d
    if type(state) is list:
        lx = len(state)
        ly = len(state[0])
        assert all(len(state[x]) == ly for x in range(1, lx))
        if dtype is None:
            dtype = state[0][0].dtype
        d = len(state[0][0])
        assert all(len(state[x][y]) == d for x in range(lx) for y in range(ly))
    else:
        if (lx is None) or (ly is None):
            raise ValueError('System size required for a single local state')
        if dtype is None:
            dtype = state.dtype
        d = len(state)
        state = [[state for y in range(ly)] for x in range(lx)]

    # normalise
    state = tree_map(lambda arr: arr / np.linalg.norm(arr), state)

    # construct tensors
    def get_bulk(local_state):
        return index_update(np.zeros([d, chi, chi, chi, chi], dtype), index[:, 0, 0, 0, 0], local_state)

    if bc == OBC:
        def get_edge(local_state):
            return index_update(np.zeros([d, chi, chi, chi], dtype), index[:, 0, 0, 0], local_state)

        def get_corner(local_state):
            return index_update(np.zeros([d, chi, chi], dtype), index[:, 0, 0], local_state)

        tensors = [[get_corner(state[0][0])]
                   + [get_edge(state[0][y]) for y in range(1, ly - 1)]
                   + [get_corner(state[0][ly - 1])]] \
                  + [[get_edge(state[x][0])]
                     + [get_bulk(state[x][y]) for y in range(1, ly - 1)]
                     + [get_edge(state[x][ly - 1])] for x in range(1, lx - 1)] \
                  + [[get_corner(state[lx - 1][0])]
                     + [get_edge(state[lx - 1][y]) for y in range(1, ly - 1)]
                     + [get_corner(state[lx - 1][ly - 1])]]

    elif bc == PBC:
        tensors = [[get_bulk(state[x][y]) for y in range(ly)] for x in range(lx)]

    else:
        raise ValueError(f'Illegal boundary conditions: {bc}')

    # construct Peps
    state = Peps(tensors, bc)
    state._norm_squared = 1.
    return state


def product_ipeps(chi: int, state: np.ndarray, dtype: Optional[np.dtype] = None):
    """
    iPEPS for a product state

    TODO support non-trivial unit cell and symmetry

    Parameters
    ----------
    chi : int
        bond dimension
    state : array
        1D array for the local state
    dtype : optional
        datatype for iPEPS tensors

    Returns
    -------
    state: Ipeps
    """

    if dtype is None:
        dtype = state.dtype
    d = len(state)

    tensor = index_update(np.zeros([d, chi, chi, chi, chi], dtype), index[:, 0, 0, 0, 0], state)
    return Ipeps([tensor], unit_cell=[[0]], symmetry=C4V)


def random_peps(d: int, chi: int, bc: str, lx: int, ly: int, dtype: np.dtype = np.complex128,
                random_generator: Optional[callable] = None):
    """
    Peps with random tensors

    Parameters
    ----------
    d : int
        local Hilbert space dimension (= dimension of physical legs)
    chi : int
        bond dimension of virtual legs
    bc : {'obc', 'pbc'}
        boundary conditions
    lx : int
        system size (horizontal)
    ly : int
        system size (vertical)
    dtype : optional
        datatype for tensors
    random_generator : callable, optional
        random_generator(shape: tuple, dtype) -> np.ndarray
        per default, draw uniformly from [-1, 1] (for both real and imag part)
        must be able to handle complex dtypes

    Returns
    -------
    state : Peps
    """

    if random_generator is None:
        def random_generator(shape, _dtype):
            return uniform(shape, _dtype, min_val=-1., max_val=1.)

    if bc == OBC:
        num_virt_legs = [[2] + [3] * (ly - 2) + [2]] \
                        + [[3] + [4] * (ly - 2) + [3]] * (lx - 2) \
                        + [[2] + [3] * (ly - 2) + [2]]
    elif bc == PBC:
        num_virt_legs = [[4] * ly] * lx
    else:
        raise ValueError(f'Illegal boundary conditions: {bc}')

    tensors = tree_map(lambda n: random_generator((d,) + n * (chi,), dtype), num_virt_legs)
    return Peps(tensors, bc)


def random_ipeps(d: int, chi: int, dtype: np.dtype = np.complex128, random_generator: Optional[callable] = None):
    """
    Ipeps with random tensor
    TODO support non-trivial unit-cell, symmetry

    Parameters
    ----------
    d : int
        local Hilbert space dimension (= dimension of physical legs)
    chi : int
        bond dimension of virtual legs
    dtype : optional
        datatype for tensors
    random_generator : callable, optional
        random_generator(shape: tuple, dtype) -> np.ndarray
        per default, draw uniformly from [-1, 1] (for both real and imag part)
        must be able to handle complex dtypes

    Returns
    -------
    state : Ipeps
    """

    if random_generator is None:
        def random_generator(shape, _dtype):
            return uniform(shape, _dtype, min_val=-1., max_val=1.)

    tensor = random_generator((d, chi, chi, chi, chi), dtype)
    # generator automatically symmetrises the tensor
    return Ipeps([tensor], unit_cell=[[0]], symmetry=C4V)


def random_deviation(state: State, rel_strength=.01) -> State:
    """
    Introduces a random (uniform) deviation of the same dtype to the tensors of a State.

    Parameters
    ----------
    state : State
    rel_strength : float
        relative strength of deviation

    Returns
    -------
    state : State
    """
    tensors = state.get_tensors()
    abs_strength = rel_strength * tree_max_abs(tensors)
    tensors = tree_map(lambda arr: arr + uniform(arr.shape, arr.dtype, min_val=-abs_strength, max_val=abs_strength),
                       tensors)
    if complex:
        tensors = tree_map(lambda arr: arr + 1.j * uniform(arr.shape, arr.dtype, min_val=-abs_strength,
                                                           max_val=abs_strength), tensors)

    return state.with_different_tensors(tensors)


def ipeps_to_peps(ipeps: Ipeps, lx: int, ly: int, bc: str, use_ctmrg: bool = True):
    """
    Creates a PEPS for a finite system from an iPEPS for an infinite system

    Parameters
    ----------
    ipeps : IPEPS
    lx : int
        The horizontal size of the new system. Must be commensurate with the unit-cell of `ipeps`
    ly : int
        The vertical size of the new system. Must be commensurate with the unit-cell of `ipeps`
    bc : {'obc', 'pbc'}
        Boundary condition keyword
    use_ctmrg : bool
        if bc == 'obc': if True, use CTMRG with bond-dimension 1 to find boundary vectors
                        if False, use naive guess np.ones([chi])
        if bc == 'pbc': no effect

    Returns
    -------
    peps : PEPS

    """
    if (ipeps.unit_cell is None) or (ipeps.unit_cell.shape == (1, 1)):
        ipeps_tensor = ipeps.get_tensors()[0]
        peps_tensors = [[ipeps_tensor for _ in range(ly)] for _ in range(lx)]

        if bc == PBC:
            return Peps(peps_tensors, PBC)

        elif bc == OBC:
            # get boundary vectors
            if use_ctmrg:
                if ipeps.symmetry != C4V:
                    raise NotImplementedError  # env would have different structure...
                env = ipeps.get_env(chi_ctm=1, max_iter=10)
                c, t = env.env_tensors
                double_layer_bvect = t[0, :, :, 0]
                U, S, Vh = np.linalg.svd(double_layer_bvect, full_matrices=False)
                bvect = Vh[0, :]
                vT = vL = vB = vR = bvect
            else:
                chi = ipeps_tensor.shape[1]
                vT = vL = vB = vR = np.ones([chi])

            # contract tensors at boundary with boundary vectors
            # bottom-left corner, (x=0,y=0)
            _tens = np.tensordot(vL, peps_tensors[0][0], [0, 2])  # (r) & (p,u,l,d,r) -> (p,u,d,r)
            _tens = np.tensordot(vB, _tens, [0, 2])  # (u) & (p,u,d,r) -> (p,u,r)
            peps_tensors[0][0] = np.transpose(_tens, [0, 2, 1])  # (p,u,r) -> (p,r,u)

            # top-left corner (x=0,y=-1)
            _tens = np.tensordot(vL, peps_tensors[0][-1], [0, 2])  # (r) & (p,u,l,d,r) -> (p,u,d,r)
            _tens = np.tensordot(vT, _tens, [0, 1])  # (d) & (p,u,d,r) -> (p,d,r)
            peps_tensors[0][-1] = _tens

            # bottom-right corner (x=-1,y=0)
            _tens = np.tensordot(vR, peps_tensors[-1][0], [0, 4])  # (l) & (p,u,l,d,r) -> (p,u,l,d)
            _tens = np.tensordot(vB, _tens, [0, 3])  # (u) & (p,u,l,d) -> (p,u,l)
            peps_tensors[-1][0] = _tens

            # top-right corner (x=-1,y=-1)
            # (l) & (p,u,l,d,r) -> (p,u,l,d)
            _tens = np.tensordot(vR, peps_tensors[-1][-1], [0, 4])
            _tens = np.tensordot(vT, _tens, [0, 1])  # (d) & (p,u,l,d) -> (p,l,d)
            peps_tensors[-1][-1] = _tens

            # left & right edges (x=0,:) and (x=-1,:)
            for y in range(1, ly - 1):
                # left edge
                # (r) & (p,u,l,d,r) -> (p,u,d,r)
                _tens = np.tensordot(vL, peps_tensors[0][y], [0, 2])
                peps_tensors[0][y] = np.transpose(_tens, [0, 2, 3, 1])  # (p,u,d,r) -> (p,d,r,u)
                # right edge
                # (l) & (p,u,l,d,r) -> (p,u,l,d)
                _tens = np.tensordot(vR, peps_tensors[-1][y], [0, 4])
                peps_tensors[-1][y] = _tens

            # upper & lower edges (:,y=0) and (:,y=-1)
            for x in range(1, lx - 1):
                # upper edge
                # (d) & (p,u,l,d,r) -> (p,l,d,r)
                _tens = np.tensordot(vT, peps_tensors[x][0], [0, 1])
                peps_tensors[x][0] = _tens
                # lower edge
                # (u) & (p,u,l,d,r) -> (p,u,l,r)
                _tens = np.tensordot(vB, peps_tensors[x][-1], [0, 3])
                peps_tensors[x][-1] = np.transpose(_tens, [0, 3, 1, 2])  # (p,u,l,r) -> (p,r,u,l)

        else:
            raise ValueError(f'Illegal bc: {bc}')

    else:
        # for obc, will need to change how bvects are obtained
        # can no longer assume all right vectors are equal
        raise NotImplementedError  # FUTURE implement


def c4v_symmetrise(tensor, num_phys_legs=1):
    """
    symmetrises a PEPS or PEPO tensor
    """
    num_virt_legs = len(tensor.shape) - num_phys_legs
    if num_phys_legs == 1:
        if num_virt_legs == 4:
            tensor = (tensor
                      + np.transpose(tensor, [0, 2, 3, 4, 1])
                      + np.transpose(tensor, [0, 3, 4, 1, 2])
                      + np.transpose(tensor, [0, 4, 1, 2, 3])) / 4.
            tensor = (tensor
                      + np.transpose(tensor, [0, 3, 2, 1, 4])
                      + np.transpose(tensor, [0, 1, 4, 3, 2])
                      + np.transpose(tensor, [0, 3, 4, 1, 2])) / 4.
        elif num_virt_legs == 3:
            raise NotImplementedError
        elif num_virt_legs == 2:
            raise NotImplementedError
        else:
            raise ValueError('Illegal shape for peps tensor')

    elif num_phys_legs == 2:
        if num_virt_legs == 4:
            tensor = (tensor
                      + np.transpose(tensor, [0, 1, 3, 4, 5, 2])
                      + np.transpose(tensor, [0, 1, 4, 5, 2, 3])
                      + np.transpose(tensor, [0, 1, 5, 2, 3, 4])) / 4.
            tensor = (tensor
                      + np.transpose(tensor, [0, 1, 4, 3, 2, 5])
                      + np.transpose(tensor, [0, 1, 2, 5, 4, 3])
                      + np.transpose(tensor, [0, 1, 4, 5, 2, 3])) / 4.
        elif num_virt_legs == 3:
            raise NotImplementedError
        elif num_virt_legs == 2:
            raise NotImplementedError
        else:
            raise ValueError('Illegal shape for peps tensor')

    else:
        raise NotImplementedError

    return tensor
