"""

Transverse field Ising model on the 2D square lattice
H = - XX - g Z
H = - \sum_{ij} S^x_i S^x_j  - g \sum_i S^z_i

"""
from typing import Optional, Tuple
from warnings import warn

from grad_tn.models.spin_one_half import state_z_plus, state_x_plus, s0, sx, sy, sz
from grad_tn.tensornetworks.peps import Peps, Pepo, NnPepo, Ipeps, LocalOperator, Ipepo, product_peps, product_ipeps, \
    random_deviation
from grad_tn.tensornetworks.peps.operators import Operator, correlator_timeslice, expval_snapshot, \
    approximate_pepo_peps, absorb_u_site
from grad_tn.tensornetworks.peps.states import State
from utils.dict_utils import parse_options
from jax import numpy as np, partial, tree_map
from jax.numpy import ndarray as array
from jax.ops import index_update, index
from jax.scipy.linalg import expm
from jax_optimise import minimise, maximise
from utils.jax_utils.ncon_jax import ncon
from utils.jax_utils.random_arrays import uniform
from utils.jax_utils.tree_util import tree_avg_abs_nonzero

# boundary condition keywords
OBC = 'obc'
PBC = 'pbc'
INFINITE = 'infinite'
VALID_BCS = [OBC, PBC, INFINITE]

# default optimisation options
OPTIMISATION_DEFAULTS_GS = dict(
    algorithm='L-BFGS',
    dtype=np.complex128,
    line_search_fn='strong_wolfe',
    display_fun=partial(print, flush=True),
    cost_print_name='Energy',
)

OPTIMISATION_DEFAULTS_EVOLUTION = dict(
    algorithm='L-BFGS',
    dtype=np.complex128,
    line_search_fn='strong_wolfe',
    display_fun=partial(print, flush=True),
    cost_print_name='Overlap',
)


class TFIM:
    """
    Transverse Field Ising model on a 2D square lattice.
    H = - \sum_{ij} S^x_i S^x_j  - g \sum_i S^z_i

    Attributes
    ----------
    g : float
        Parameter of the Hamiltonian. Relative strength of the transverse field-
    bc : {'obc', 'pbc', 'infinite'}
        The boundary conditions:
        'obc': Open boundary conditions. Model is on a finite system.
        'pbc': Periodic boundary conditions. Model is on a finite system.
        'infinite': Infinite, translationally invariant and C4v symmetric system
    hamiltonian : NnPEPO
        The hamiltonian as a nearest-neighbour PEPO (nnPEPO)
        TODO type of hamiltonian for iPEPS?
    """

    def __init__(self, g: float, bc: str, lx=None, ly=None, dtype_hamiltonian: Optional[np.dtype] = np.float64):
        """
        Parameters
        ----------
        g : float
            Parameter of the Hamiltonian. Relative strength of the transverse field-
        bc : {'obc', 'pbc', 'infinite'}
            The boundary conditions:
            'obc': Open boundary conditions. Model is on a finite system.
            'pbc': Periodic boundary conditions. Model is on a finite system.
            'infinite': Infinite, translationally invariant and C4v symmetric system
        dtype_hamiltonian : np.dtype, optional
            The dtype for the Hamiltonian
        """
        if bc not in VALID_BCS:
            raise ValueError(f'Unknown boundary conditions: {bc}')

        self.g = g
        self.bc = bc
        self.hamiltonian = get_hamiltonian(g=g, bc=bc, lx=lx, ly=ly, dtype=dtype_hamiltonian)
        self._dtype_ham = dtype_hamiltonian

    def energy(self, state: State, **contraction_options):
        """
        Energy of a state (normalised expectation value of `self.hamiltonian` w.r.t `state`

        Parameters
        ----------
        state : PEPS or IPEPS

        Returns
        -------
        energy : float
        """

        return self.hamiltonian.expval(state, **contraction_options)

    def groundstate(self, chi: int, system_size: Optional[Tuple[int, int]] = None,
                    initial_state: Optional[str] = 'ps',
                    initial_noise: Optional[float] = None,
                    contraction_options: Optional[dict] = None,
                    optimisation_options: Optional[dict] = None,
                    ):
        """
        Computes the groundstate of the model by minimising a trial states energy.

        Parameters
        ----------
        chi : int
            The bond-dimension of the (i)PEPS
        system_size : (int, int), optional
            for OBC and PBC: the system size, for INFINITE: no effect
        initial_state : str or jax.numpy.ndarray or PEPS or IPEPS, optional
            The initial state for the optimisation.
            If a string keyword:
                'ps' : the product state of the respective phase (z+ for g > gc ~ 3.5, x+ for g < gc)
                        `initial_noise>0` recommended, since product-states might have zero-gradient
                'z+' : the z+ product state, `initial_noise>0` recommended
                'x+' : the x+ product state, `initial_noise>0` recommended
                'ipeps' : (only for finite systems)
                        finds the (iPEPS) groundstate of the same model on the infinite lattice,
                        which can directly be used for PBC, or is cut off at the boundary for OBC
                TODO 'random' keyword
            If an ndarray:
                1D array of the local state, start from the product state of this local state,
                if it is a z or x eigenstate, `initial_noise=True` recommended
            If PEPS:
                (only for finite systems): PEPS of the initial state
            If IPEPS:
                (only for infinite systems): iPEPS of the initial state
            Default: like 'ps'
        initial_noise : float, optional
            If `initial_noise > 0`: add a random deviation to the initial_state.
            then, `initial_noise` is the relative strength of the deviation
        contraction_options : dict, optional
            options for the PEPS contraction of the energy-expectationvalue, see `TFIM.energy`
        optimisation_options : dict, optional
            options for optimisation. kwargs for `jax_optimise.minimise`

        Returns
        -------
        gs : PEPS or IPEPS
        gs_energy : float
        """

        # parse dicts
        optimisation_options = parse_options(optimisation_options, OPTIMISATION_DEFAULTS_GS)
        contraction_options = parse_options(contraction_options)

        # parse system size
        if self.bc == INFINITE:
            lx, ly = None, None
        else:
            lx, ly = system_size
            assert lx > 0
            assert ly > 0

        # parse initial guess
        initial_guess = _parse_initial_state(initial_state, chi, self.bc, lx, ly, self.g, initial_noise,
                                             complex_tensors=np.iscomplexobj(optimisation_options['dtype']))

        # define cost_function
        def cost_function(new_tensors):
            new_state = initial_guess.with_different_tensors(new_tensors)
            energy = self.energy(new_state, **contraction_options)
            return np.reshape(energy, ())

        # optimisation
        optimal_tensors, optimal_energy, info = minimise(cost_function, initial_guess.get_tensors(),
                                                         **optimisation_options)
        optimal_state = initial_guess.with_different_tensors(optimal_tensors)

        return optimal_state, optimal_energy


# FIXME option for polarised environment
# FIXME system size
class TimeEvolution:
    """
    Time Evolution (in real or imaginary time) w.r.t to the Hamiltonian of the TFIM
    H = - \sum_{ij} S^x_i S^x_j  - g \sum_i S^z_i

    Uses a suzuki-trotter decomposition:
    U(dt) ~ U_vert(dt/2) U_bond(dt) U_vert(dt/2)

    and decomposes the bond operators in U_bond analytically to obtain a PEPO structure for U(dt)

    Attributes
    ----------
    g : float
        Model parameter. Relative strength of transverse field
    dt : float
        Time step dt for evolution
    bc : {'obc', 'pbc', 'infinite'}
        The boundary conditions
    real_time : bool
        real time evolution or imaginary time evolution
    evolution_pepo: PEPO or IPEPO or RoiPEPO
        The evolution operator U(dt) in PEPO form
    """

    def __init__(self, g: float, dt: float, bc: str, real_time: bool = True,
                 lx: Optional[int] = None, ly: Optional[int] = None,
                 pepo_dtype: Optional[np.dtype] = None):

        if bc not in VALID_BCS:
            raise ValueError(f'Unknown bc keyword: {bc}')

        if (bc in [PBC, OBC]) and (lx is None or ly is None):
            raise ValueError(f'Need to specify system size for finite systems')

        self.lx = lx
        self.ly = ly
        self.g = g
        self.dt = dt
        self.bc = bc
        self.real_time = real_time

        if real_time:
            self.pepo_dtype = np.complex128 if pepo_dtype is None else pepo_dtype
            self.evolution_pepo = evolution_pepo_real_time(g, dt, bc, self.pepo_dtype, lx, ly)
        else:
            self.pepo_dtype = np.float64 if pepo_dtype is None else pepo_dtype
            self.evolution_pepo = evolution_pepo_imag_time(g, dt, bc, self.pepo_dtype, lx, ly)

    def evolve(self, state: State, n_steps: int = 1,
               overlap_threshold: Optional[float] = 0.9,
               contraction_options: Optional[dict] = None,
               optimisation_options: Optional[dict] = None,
               initial: str = 'old',
               random_dev: Optional[float] = 0.05,
               state_energy: Optional[float] = None,
               ) -> State:
        """
        evolve a state

        Parameters
        ----------
        state : PEPS or IPEPS or RoiPEPS
        n_steps : int, optional
            The number of evolution steps. total time == `n_steps * self.dt`
            per default: `1`
        overlap_threshold : float, optional
            A warning is issued if the maximised overlap is below this threshold
        contraction_options : dict, optional
            options about the contraction that are specific to the boundary conditions
            if `self.bc in [PBC, OBC]`:
                'contraction_method': {'brute', 'bmps', 'bmps_var', ...}, see tensors_networks.two_dim.peps_helpers
                'chi_bmps': int, the bond-dimension of boundary-MPS
            if `self.bc in [INFINITE]`:
                'chi_ctm': int, the bond-dimension of the CTMRG environment
                'bvar_threshold': float, the threshold for the CTMRG boundary variance
        optimisation_options : dict, optional
            options for optimisation. kwargs for `jax_optimise.maximise`
        initial : {'old', 'approximate', 'u_site'}
            initial guess protocol (paired with random_dev)
                'old' :         use the old state
                'approximate' : approximate the PEPO - PEPS product
                'u_site' :      absorb the time evolution w.r.t. the site term
        random_dev : float, optional
            if not None, the initial guess for the optimisation is `state` plus a random deviation of this relative
            strength
        state_energy : float, optional
            The energy of `state` or `None`. If it is not `None`, the overlap for imaginary time evolution
            will be properly normalised

        Returns
        -------
        evolved_state : PEPS or IPEPS or RoiPEPS
        """
        assert (overlap_threshold is None) or (0 <= overlap_threshold < 1)
        optimisation_options = parse_options(optimisation_options, OPTIMISATION_DEFAULTS_EVOLUTION)
        contraction_options = parse_options(contraction_options)

        if n_steps < 1:
            raise ValueError('n_steps must be a positive (non-zero) integer')

        # reduce n_steps > 1 to multiple calls with n_steps == 1
        if n_steps > 1:
            for n in range(n_steps):
                state = self.evolve(state, n_steps=1, overlap_threshold=overlap_threshold,
                                    contraction_options=contraction_options, optimisation_options=optimisation_options,
                                    initial=initial, random_dev=random_dev, state_energy=state_energy)
            return state

        # can assume n_steps == 1 from now on

        # define cost function
        if self.bc in [PBC, OBC]:
            if self.real_time or (state_energy is None):
                def cost_function(new_tensors):
                    new_state = state.with_different_tensors(new_tensors)
                    overlap = self.evolution_pepo.matrix_element(new_state, state, **contraction_options)
                    res = np.abs(overlap)
                    return res
            else:
                # for imaginary time evolution only: normalise so that optimal overlap is ~1
                def cost_function(new_tensors):
                    new_state = state.with_different_tensors(new_tensors)
                    overlap = self.evolution_pepo.matrix_element(new_state, state, **contraction_options)
                    res = np.abs(overlap)
                    return res * np.exp(self.dt * state_energy)

        else:
            def cost_function(new_tensors):
                new_state = state.with_different_tensors(new_tensors)
                overlap = self.evolution_pepo.matrix_element(new_state, state, **contraction_options)
                return np.abs(overlap)

        # initial guess
        if initial == 'old':
            initial_tensors = state.get_tensors()
        elif initial == 'approximate':
            initial_tensors = approximate_pepo_peps(self.evolution_pepo, state).get_tensors()
        elif initial == 'u_site':
            u_site = get_u_site(g=self.g, dt=self.dt, dtype=self.pepo_dtype, real_time=self.real_time)
            initial_tensors = absorb_u_site(state, u_site).get_tensors()
        else:
            raise ValueError(f'Invalid initial keyword: {initial}')

        # random deviation
        if random_dev:
            amplitude = abs(random_dev * tree_avg_abs_nonzero(initial_tensors))
            initial_tensors = tree_map(lambda arr: arr + uniform(arr.shape, arr.dtype, min_val=-amplitude,
                                                                 max_val=amplitude), initial_tensors)

        # maximise overlap
        optimal_tensors, optimal_overlap, info = maximise(cost_function, initial_tensors, **optimisation_options)
        optimal_state = state.with_different_tensors(optimal_tensors)

        if overlap_threshold and self.real_time and (optimal_overlap < overlap_threshold):
            warn(f'Optimal overlap is below threshold: overlap = {optimal_overlap} < {optimal_overlap} = threshold')

        # phase correction
        if isinstance(optimal_state, Peps):
            ov = self.evolution_pepo.matrix_element(optimal_state, state, **contraction_options)
            phase_factor = ov / np.abs(ov)
            optimal_state.absorb_factor(phase_factor)
        else:
            warn('Losing global phase information in time evolution.')

        return optimal_state


def get_hamiltonian(g: float, bc: str, lx: int, ly: int, dtype: np.dtype) -> Operator:
    if bc in [OBC, PBC]:
        o = np.zeros_like(s0)
        C = np.array([[s0, o, o], [-sx, o, o], [-.5 * g * sz, sx, s0]], dtype=dtype)
        C = np.transpose(C, [2, 3, 0, 1])  # (i,j,p,p*) -> (p,p*,i,j)
        I = np.array([[o, o, o], [o, o, o], [s0, o, o]], dtype=dtype)
        I = np.transpose(I, [2, 3, 0, 1])  # (i,j,p,p*) -> (p,p*,i,j)
        O = np.zeros_like(I)
        D = np.array([[I, O], [C, I]], dtype=dtype)
        D = np.transpose(D, [2, 3, 4, 5, 0, 1])  # (k,l,p,p*,i,j) -> (p,p*,i,j,k,l)
        return NnPepo(C, D, bc=bc, lx=lx, ly=ly, hermitian=True)

    elif bc == INFINITE:
        xx = np.einsum('ij,kl->ikjl', sx, sx)  # (p,p*) & (p,p*) -> (p1,p2,p1*,p2*)
        Iz = np.einsum('ij,kl->ikjl', s0, sz)
        zI = np.einsum('ij,kl->ikjl', sz, s0)

        h_bond = 2 * (- xx - g / 4. * Iz - g / 4. * zI)
        # factor 2: hor + vert = 2 * hor
        # factors 1/4. : four bonds per site

        operator_geometry = np.array([[0, 1]])
        return LocalOperator(h_bond, operator_geometry, hermitian=True)

    else:
        raise ValueError('invalid bc')


def get_u_site(g: float, dt: float, dtype: np.dtype, real_time: bool):
    if real_time:
        return np.asarray(expm(1j * g * dt * sz), dtype=dtype)
    else:
        np.asarray(expm(g * dt * sz), dtype=dtype)


def evolution_pepo_real_time(g: float, dt: float, bc: str, dtype: np.dtype,
                             lx: Optional[int] = None, ly: Optional[int] = None) -> Operator:
    # PEPO for U(dt) ~ U_vert(dt/2) U_bond(dt) U_vert(dt/2)
    #
    # half bond operators:
    #
    #      |    |           |    |
    #      U_bond     =     A -- A
    #      |    |           |    |
    #
    # expm(-i H_bond dt) = expm(-i (-XX) dt) = expm(i dt XX) = cos(dt) + i sin(dt) XX = A_0 A_0 + A_1 A_1
    # with A_0 = (cos(dt) ** 0.5) * 1  ,  A_1 = (i sin(dt)) ** 0.5 * X
    # A & B legs: (p,p*,k)

    A = np.zeros([2, 2, 2], dtype=dtype)  # u_hb(p,p*,a)

    # A[:,:,0] = (np.cos(dt)) ** 0.5 * s0
    # A[:,:,1] = (1.j * np.sin(dt)) ** 0.5 * sx
    A = index_update(A, index[:, :, 0], (np.cos(dt)) ** 0.5 * s0)
    A = index_update(A, index[:, :, 1], (1.j * np.sin(dt)) ** 0.5 * sx)

    # expm(- i H_vert dt/2) = expm(- i(-gZ) dt/2) = expm(i/2 g dt Z)
    u_vert = np.asarray(expm(.5j * g * dt * sz), dtype=dtype)

    return _build_evolution_pepo(u_vert, A, bc, lx, ly)


def evolution_pepo_imag_time(g: float, dt: float, bc: str, dtype: np.dtype,
                             lx: Optional[int] = None, ly: Optional[int] = None) -> Operator:
    # PEPO for U(dt) ~ U_vert(dt/2) U_bond(dt) U_vert(dt/2)
    #
    # half bond operators:
    #
    #      |    |           |    |
    #      U_bond     =     A -- A
    #      |    |           |    |
    #
    # expm(- H_bond dt) = expm(- (-XX) dt) = expm(dt XX) = cosh(dt) + sinh(dt) XX = A_0 A_0 + A_1 A_1
    # with A_0 = (cosh(dt) ** 0.5) * 1  ,  A_1 = (sinh(dt) ** 0.5) * X
    # A & B legs: (p,p*,k)

    A = np.empty([2, 2, 2], dtype=dtype)
    A = index_update(A, index[:, :, 0], (np.cosh(dt) ** 0.5) * s0)
    A = index_update(A, index[:, :, 1], (np.sinh(dt) ** 0.5) * sx)
    # expm(- H_vert dt/2) = expm(- (-gZ) dt/2) = expm(g dt/2 Z)
    u_vert = np.asarray(expm(g * (dt / 2) * sz), dtype=dtype)

    return _build_evolution_pepo(u_vert, A, bc, lx, ly)


def _build_evolution_pepo(u_vert, A, bc: str,
                          lx: Optional[int] = None, ly: Optional[int] = None) -> Operator:
    # DOC : lx, ly have no effect for INFINITE

    u_bulk = ncon([u_vert, A, A, A, A, u_vert], [[1, -2], [2, 1, -6], [3, 2, -3], [4, 3, -4], [5, 4, -5], [-1, 5]])

    if bc == PBC:
        if (lx is None) or (ly is None):
            raise ValueError
        tensors = [[u_bulk for _ in range(ly)] for _ in range(lx)]
        return Pepo(tensors, bc=PBC, hermitian=False)

    elif bc == OBC:
        if (lx is None) or (ly is None):
            raise ValueError

        u_edge = ncon([u_vert, A, A, A, u_vert], [[1, -2], [2, 1, -5], [3, 2, -3], [4, 3, -4], [-1, 4]])
        u_corner = ncon([u_vert, A, A, u_vert], [[1, -2], [2, 1, -3], [3, 2, -4], [-1, 3]])

        tensors = [[u_corner] + [u_edge for _ in range(ly - 2)] + [u_corner]] \
                  + [[u_edge] + [u_bulk for _ in range(ly - 2)] + [u_edge] for _ in range(lx - 2)] \
                  + [[u_corner] + [u_edge for _ in range(ly - 2)] + [u_corner]]
        return Pepo(tensors, bc=OBC, hermitian=False)

    elif bc == INFINITE:
        return Ipepo([u_bulk], unit_cell=np.array([[0]]), symmetry='c4v', hermitian=False)

    raise ValueError(f'Unknown boundary conditions: {bc}')


def xx_correlator_timeslice(groundstate: State, evolved_quenched_state: State,
                            gs_energy: float, t: float, contraction_options: Optional[dict] = None) -> np.ndarray:
    """
    Correlation function <S^x(t)S^x(0)> where < . > is the expval w.r.t the groundstate
    for all positions of S^x(t)  (output array dimensions)
    where the position of S^x(0) is implicitly defined via `evolved_quenched`

    Parameters
    ----------
    groundstate
        PEPS for the groundstate
    evolved_quenched_state
        PEPS for exp(-iHt)S^x|GS>
    gs_energy
        <GS| H |Gs>
    t
    contraction_options

    Returns
    -------
    xx_correlators : jax.numpy.ndarray
        `xx_correlators[x,y]` is <S^x(t)S^x(0)> for S^x(t) acting on site (x,y)
    """

    if contraction_options is None:
        contraction_options = {}

    return correlator_timeslice(groundstate, evolved_quenched_state, sx, gs_energy, t, **contraction_options)


def zz_correlator_timeslice(groundstate: State, evolved_quenched_state: State,
                            gs_energy: float, t: float, contraction_options: Optional[dict] = None) -> np.ndarray:
    """
    Correlation function <S^z(t)S^z(0)> where < . > is the expval w.r.t the groundstate
    for all positions of S^z(t)  (output array dimensions)
    where the position of S^z(0) is implicitly defined via `evolved_quenched`

    Parameters
    ----------
    groundstate
        PEPS for the groundstate
    evolved_quenched_state
        PEPS for exp(-iHt)S^z|GS>
    gs_energy
        <GS| H |Gs>
    t
    contraction_options

    Returns
    -------
    xx_correlators : jax.numpy.ndarray
        `xx_correlators[x,y]` is <S^x(t)S^x(0)> for S^x(t) acting on site (x,y)
    """

    if contraction_options is None:
        contraction_options = {}

    return correlator_timeslice(groundstate, evolved_quenched_state, sz, gs_energy, t, **contraction_options)


def x_snapshot(state: State, contraction_options: Optional[dict] = None) -> array:
    """
    <state| S^x | state> for all positions of S^x
    """
    if contraction_options is None:
        contraction_options = {}
    return expval_snapshot(state, sx, hermitian=True, **contraction_options)


def y_snapshot(state: State, contraction_options: Optional[dict] = None) -> array:
    """
    <state| S^x | state> for all positions of S^x
    """
    if contraction_options is None:
        contraction_options = {}
    return expval_snapshot(state, sy, hermitian=True, **contraction_options)


def z_snapshot(state: State, contraction_options: Optional[dict] = None) -> array:
    """
    <state| S^z | state> for all positions of S^z
    """
    if contraction_options is None:
        contraction_options = {}
    return expval_snapshot(state, sz, hermitian=True, **contraction_options)


def _parse_initial_state(initial_state, chi, bc, lx, ly, g, initial_noise, complex_tensors=False):
    # Default
    if not initial_state:
        initial_state = 'ps'
        if initial_noise is None:
            initial_noise = 0.05
    gc = 3.05

    state = None

    if isinstance(initial_state, Peps) or isinstance(initial_state, Ipeps):
        if isinstance(initial_state, Peps) and (bc not in [PBC, OBC]):
            raise ValueError(f'initial state of type PEPS is invalid for {bc} boundary conditions')
        if isinstance(initial_state, Ipeps) and (bc != INFINITE):
            raise ValueError(f'initial state of type IPEPS is invalid for {bc} boundary conditions')

        state = initial_state

    if type(initial_state) == str:
        if initial_state == 'ps':
            initial_state = state_z_plus if g > gc else state_x_plus
        elif initial_state == 'z+':
            initial_state = state_z_plus
        elif initial_state == 'x+':
            initial_state = state_x_plus
        else:
            raise ValueError(f'Initial state keyword {initial_state} not supported')

    if not state:
        if bc in [OBC, PBC]:
            state = product_peps(chi=chi, bc=bc, state=initial_state, lx=lx, ly=ly)
        else:
            state = product_ipeps(chi=chi, state=initial_state)

    if complex_tensors:
        state = state.with_different_tensors(tree_map(lambda arr: np.asarray(arr, dtype=np.complex128),
                                                      state.get_tensors()))

    if initial_noise:
        state = random_deviation(state, rel_strength=initial_noise)

    return state
