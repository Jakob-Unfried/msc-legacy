import jax.numpy as np
from jax.ops import index, index_update
from autodiff import fixed_points
from autodiff.svd_safe import svd_truncated
from utils.jax_utils.ncon_jax import ncon
from utils.jax_utils.tree_util import tree_map


def expval(env_tensors, ipeps_tensors, operator, operator_geometry):
    ipeps_tensor, = ipeps_tensors
    c, t = env_tensors
    chi_ctm = c.shape[0]
    chi_peps = ipeps_tensor.shape[1]
    d = operator.shape[0]

    n_sites = len(operator.shape) // 2
    assert c.shape == (chi_ctm, chi_ctm)
    assert t.shape == (chi_ctm, chi_peps, chi_peps, chi_ctm)
    assert ipeps_tensor.shape == (d,) + 4 * (chi_peps,)
    assert operator.shape == (d,) * (2 * n_sites)
    assert np.all(operator_geometry < n_sites)

    if operator_geometry.shape == (1, 1):  # single-site operator
        #                                            /
        #     c - t - c                           - a -
        #     |   |   |                 |         / |
        #     t - B - t               - B -   =     OP
        #     |   |   |                 |           | /
        #     c - t - c                           - a -
        #                                          /

        tensors = [c, c, c, c, t, t, t, t, ipeps_tensor, np.conj(ipeps_tensor), operator]
        connects = [[6, 1], [7, 2], [3, 8], [4, 5], [1, 13, 18, 7], [2, 12, 17, 3], [8, 14, 15, 4], [5, 11, 16, 6],
                    [9, 13, 11, 14, 12], [10, 18, 16, 15, 17], [10, 9]]
        cont_order = [4, 10, 6, 8, 5, 16, 15, 7, 2, 1, 3, 18, 17, 11, 14, 9, 13, 12, ]
        val = ncon(tensors, connects, cont_order)  # tensortrace: expval_1_1.ttp network 1

        tensors = [c, c, c, c, t, t, t, t, ipeps_tensor, np.conj(ipeps_tensor)]
        connects = [[6, 1], [7, 2], [3, 8], [4, 5], [1, 10, 11, 7], [2, 16, 15, 3], [8, 13, 14, 4], [5, 9, 12, 6],
                    [17, 10, 9, 13, 16], [17, 11, 12, 14, 15]]
        cont_order = [5, 7, 8, 4, 14, 12, 1, 2, 6, 11, 15, 3, 10, 16, 13, 9, 17, ]
        normalisation = ncon(tensors, connects, cont_order)  # tensortrace: expval_1_1.ttp network 2

        return val / normalisation

    elif operator_geometry.shape in [(2, 1), (1, 2)]:  # nearest-neighbour operator
        #                                                    /   /
        #     c - t - t - c                               - a - a -
        #     |   |   |   |                 |   |         / | / |
        #     t - **B** - t               - **B** -   =     **OP*
        #     |   |   |   |                 |   |           | / | /
        #     c - t - t - c                               - a - a -
        #                                                  /   /

        # computed as horizontal, but due to rotational symmetry, vertical is the same
        tensors = [c, c, c, c, t, t, t, t, t, t, ipeps_tensor, ipeps_tensor, np.conj(ipeps_tensor),
                   np.conj(ipeps_tensor), operator]
        connects = [[10, 1], [3, 4], [5, 6], [8, 9], [1, 22, 23, 2], [2, 26, 27, 3], [4, 17, 18, 5], [6, 25, 28, 7],
                    [7, 21, 24, 8], [9, 15, 20, 10], [11, 22, 15, 21, 16], [13, 26, 16, 25, 17], [12, 23, 20, 24, 19],
                    [14, 27, 19, 28, 18], [12, 14, 11, 13]]
        cont_order = [5, 8, 3, 6, 17, 25, 1, 10, 18, 28, 4, 26, 27, 13, 14, 23, 20, 7, 2, 9, 12, 24, 19, 22, 15, 16, 11,
                      21, ]
        val = ncon(tensors, connects, cont_order)  # tensortrace: expval_2_1.ttp

        tensors = [c, c, c, c, t, t, t, t, t, t, ipeps_tensor, ipeps_tensor, np.conj(ipeps_tensor),
                   np.conj(ipeps_tensor)]
        connects = [[10, 1], [3, 4], [5, 6], [8, 9], [1, 18, 19, 2], [2, 22, 23, 3], [4, 13, 14, 5], [6, 21, 24, 7],
                    [7, 17, 20, 8], [9, 11, 16, 10], [26, 18, 11, 17, 12], [25, 22, 12, 21, 13], [26, 19, 16, 20, 15],
                    [25, 23, 15, 24, 14]]
        cont_order = [8, 5, 6, 14, 24, 10, 3, 13, 21, 25, 4, 22, 23, 7, 1, 15, 20, 9, 16, 19, 2, 11, 18, 12, 17, 26, ]
        normalisation = ncon(tensors, connects, cont_order)  # tensortrace: expval_2_1_normalisation.ttp

        return val / normalisation

    else:
        raise NotImplementedError


def get_env(ipeps_tensors, chi_ctm, bvar_threshold, max_iter):
    # TODO should we symmetrise the ipeps tensor?

    a, = ipeps_tensors
    chi_peps = a.shape[1]

    # initialise environment
    # (p*,uldr) & (p,uldr) -> (uldr,uldr) -> (uu,ll,dd,rr)
    flat_tens = np.transpose(np.tensordot(np.conj(a), a, [0, 0]), [0, 4, 1, 5, 2, 6, 3, 7])
    u, _u, l, _l, d, _d, r, _r = flat_tens.shape
    # (uu,ll,dd,rr) -> (U,L,d,d',R)
    flat_tens = np.reshape(flat_tens, [u * _u, l * _l, d, _d, r * _r])
    c_init = np.sum(flat_tens, axis=(2, 3, 4))  # (D,R)
    t_init = np.sum(flat_tens, axis=0)  # (L,d,d',R)

    if c_init.shape[0] > chi_ctm:
        c_init = c_init[:chi_ctm, :chi_ctm]
        t_init = t_init[:chi_ctm, :, :, :chi_ctm]

    # enforce c4v symmetry
    c_init, t_init = _c4v_symmetrise(c_init, t_init, normalise=True)
    # expand to full chi_ctm, for traceability
    _chi = c_init.shape[0]
    if _chi < chi_ctm:
        c_init = index_update(np.zeros([chi_ctm, chi_ctm], dtype=c_init.dtype),
                              index[:_chi, :_chi], c_init)
        t_init = index_update(np.zeros([chi_ctm, chi_peps, chi_peps, chi_ctm], dtype=t_init.dtype),
                              index[:_chi, :, :, :_chi], t_init)

    env_init = c_init, t_init

    def update(b, env):
        c, t = env

        # C insertion
        c_tilde = ncon([c, t, t, b, np.conj(b)],
                       [[1, 2], [2, 3, 4, -4], [-1, 5, 6, 1], [7, 3, 5, -2, -5], [7, 4, 6, -3, -6]],
                       [1, 2, 3, 5, 4, 6, 7])
        # (D,d,d',R,r,r') -> (D~,R~)
        _D, _d, _d_, _R, _r, _r_ = c_tilde.shape
        c_tilde = np.reshape(c_tilde, [_D * _d * _d_, _R * _r * _r_])
        # T insertion
        t_tilde = ncon([t, b, np.conj(b)], [[-1, 1, 2, -6], [3, 1, -2, -4, -7], [3, 2, -3, -5, -8]])
        # (L,l,l',d,d',R,r,r') -> (L~,d,d',R~)
        _L, _l, _l_, _d, _d_, _R, _r, _r_ = t_tilde.shape
        t_tilde = np.reshape(t_tilde, [_L * _l * _l_, _d, _d_, _R * _r * _r_])

        # enforce symmetry
        c_tilde = _c4v_symmetrise_c(c_tilde)

        # find projector
        P, _, _ = svd_truncated(c_tilde, chi_max=chi_ctm, cutoff=0.)  # (D~,R)

        # renormalise
        c = np.transpose(P) @ c_tilde @ P
        t = ncon([np.conj(P), t_tilde, np.conj(P)], [[1, -1], [1, -2, -3, 2], [2, -4]])

        # enforce symmetry
        env = _c4v_symmetrise(c, t)

        return env

    def convergence_condition(b, env, _):
        c, t = env
        return _variance3(c, t, b) < bvar_threshold

    env_star = fixed_points.fixed_point_novjp(update, a, env_init, convergence_condition, max_iter=max_iter)
    return env_star


def _c4v_symmetrise(cs, ts, normalise=True):
    cs = tree_map(lambda c: _c4v_symmetrise_c(c, normalise), cs)
    ts = tree_map(lambda t: _c4v_symmetrise_t(t, normalise), ts)
    return cs, ts


def _c4v_symmetrise_c(c: np.ndarray, normalise=True) -> np.ndarray:
    c = c + np.transpose(c)
    if normalise:
        return c / np.linalg.norm(c)
    else:
        return c / 2.


def _c4v_symmetrise_t(t: np.ndarray, normalise=True) -> np.ndarray:
    t = t + np.transpose(t, [3, 1, 2, 0])
    if normalise:
        return t / np.linalg.norm(t)
    else:
        return t / 2.


def _variance(c: np.ndarray, t: np.ndarray) -> float:
    """
    Returns the variance |<aa aa>| - <aa>**2 that would be zero for the fixed point of the RG
    This version is only checking if c and t are ANY environment, see also `_variance3`

    Parameters
    ----------
    c
        The corner matrix c(d, r) with `chi`-dimensional legs
    t
        The half-column / half-row tensor t(l, d, d', r)
    Returns
    -------
    var : float
        The variance

    """

    #          c --- c               c --- t --- c                     c --- t --- t --- c
    #  lr  =   |     |      ltr  =   |     |     |         lttr   =    |     |     |     |
    #          c --- c               c --- t --- c                     c --- t --- t --- c
    # if RG fixed point is reached then lttr/lr = <tt> = <t>**2 = (ltr/lr)**2

    lr = np.trace(c @ c @ c @ c)

    connects = [[1, 6], [6, 7, 8, 5], [5, 4], [4, 3], [2, 1], [2, 7, 8, 3]]
    cont_order = [4, 1, 3, 2, 6, 5, 7, 8, ]
    ltr = ncon([c, t, c, c, c, t], connects, cont_order)

    connects = [[1, 4], [4, 5, 6, 9], [10, 3], [3, 11], [2, 1], [2, 5, 6, 12], [9, 7, 8, 10], [12, 7, 8, 11]]
    cont_order = [1, 2, 3, 11, 10, 7, 8, 12, 4, 5, 6, 9, ]
    lttr = ncon([c, t, c, c, c, t, t, t], connects, cont_order)

    return abs(lttr / lr) - (ltr / lr) ** 2


def _variance3(c: np.ndarray, t: np.ndarray, a: np.ndarray) -> float:
    """
    Returns the variance |<aa aa>| - <aa>**2 that would be zero for the fixed point of the RG
    This version also checks if `c` and `t` are an infinite environment of `aa`

    Parameters
    ----------
    c
        The corner matrix c(d, r) with `chi`-dimensional legs
    t
        The half-column / half-row tensor t(l, d, d', r)
    a
        A iPEPS tensor a(p, u, l, d, r)

    Returns
    -------
    var : float
        The variance

    """

    #          c --- c               c --- t --- c                     c --- t --- t --- c
    #          |     |               |     |     |                     |     |     |     |
    #  lr  =   t --- t      ltr  =   t --- aa--- t         lttr   =    t --- aa--- aa--- t
    #          |     |               |     |     |                     |     |     |     |
    #          c --- c               c --- t --- c                     c --- t --- t --- c
    # if RG fixed point is reached then lttr/lr = <tt> = <t>**2 = (ltr/lr)**2

    connects = [[1, 5], [2, 7, 8, 1], [5, 4], [3, 6], [6, 2], [3, 7, 8, 4]]
    cont_order = [6, 5, 3, 4, 1, 2, 7, 8, ]
    lr = ncon([c, t, c, c, c, t], connects, cont_order)

    connects = [[1, 10], [2, 14, 17, 1], [11, 4], [3, 13], [12, 2], [3, 15, 16, 4], [13, 8, 9, 12], [10, 6, 7, 11],
                [5, 6, 14, 8, 15], [5, 7, 17, 9, 16]]
    cont_order = [4, 3, 11, 1, 2, 12, 16, 7, 10, 17, 13, 9, 14, 8, 15, 6, 5, ]
    ltr = ncon([c, t, c, c, c, t, t, t, a, np.conj(a)], connects, cont_order)

    connects = [[1, 15], [2, 21, 26, 1], [17, 4], [3, 20], [18, 2], [3, 23, 24, 4], [19, 8, 9, 18], [15, 6, 7, 16],
                [5, 6, 21, 8, 22], [5, 7, 26, 9, 25], [20, 13, 14, 19], [16, 11, 12, 17], [10, 11, 22, 13, 23],
                [10, 12, 25, 14, 24]]
    cont_order = [18, 15, 20, 1, 3, 7, 26, 17, 14, 24, 13, 23, 10, 4, 11, 12, 19, 16, 2, 9, 25, 6, 21, 5, 22, 8, ]
    lttr = ncon([c, t, c, c, c, t, t, t, a, np.conj(a), t, t, a, np.conj(a)], connects, cont_order)

    return abs(lttr / lr) - (ltr / lr) ** 2
