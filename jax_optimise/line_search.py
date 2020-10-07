from warnings import warn

import jax.numpy as np
from jax import grad
from jax_optimise.list_util import list_dot, list_add_prefactor, list_conj

"""

CONVENTIONS:
g_k = grad(f)(z_k) = (∂f/∂z*)(z_k)
p_k : search direction
f_k = f(z_k)
alpha : step length -> z_{k+1} = z_k + alpha * p_k
function on the line: phi(alpha) = f(z_k + alpha * p_k)

LINE SEARCH CONDITIONS:
(1)  Sufficient decrease (Armijo) condition:
        f(z_k + alpha * p_k) <= f(z_k) + c1 alpha Re{conj(gc_k) * p_k}
        phi(alpha) <= phi(0) + c1 * alpha * phi'(0)
(2a) Curvature condition:
        Re{conj(grad(f)(z_k)) * p_k} >= c2 Re{conj(gc_k) * p_k}
        phi'(alpha) >= c2 * phi'(0)
(2b) Strong curvature condition:
        | Re{conj(grad(f)(z_k + alpha p_k)) * p_k} | <= c2 | Re{conj(gc_k) * p_k} |
        | phi'(alpha) | <= c2 * | phi'(0) |
Wolfe conditions: (1) & (2a)
Strong Wolfe conditions: (1) & (2b)

DATA STRUCTURES:
all vectors, like z_k, p_k, gc_k, gradients are lists of np.ndarrays.
to perform actions on them, use the optimisation.list_util module

"""


def armijo_backtracking(f, z_k, p_k, f_k, der_phi_0, alpha_0=1., alpha_min=0, c1=1e-4,
                        max_fun_evals=30, warnings=True):
    """
    finds a step-length `alpha` such that `alpha_min < alpha <= alpha_0` and that the armijo condition (1) is fulfilled
    
    alpha_0 is assumed to be large enough, that no curvature condition needs to be considered.
    this allows the line search to only evaluate function values and not require gradients
    
    Parameters
    ----------
    f : callable
        The cost function of the optimisation problem  `f(z: List[NumericType]) -> float`
    z_k : list of {ndarray or complex or float}
        The last iterate from which the line search for the next iterate is started
    p_k : list of {ndarray or complex or float}
        The search direction
    f_k : float
        The function value `f(z_k)`
    der_phi_0 : float
        The derivative phi'(0) = Re{∂f/∂z(z_k) * p_k} = `np.real(list_dot(list_conj(g_k), p_k))`
    alpha_0 : float, optional
        The initial guess for alpha.
        For quasi-Newton methods, this should be 1. to make use of their convergence properties
    alpha_min : float, optional
        The minimal acceptable alpha.
    c1 : float, optional
        The parameter in the Armijo condition. Must fulfill `0 < c1 < 1`.
        Typically much smaller than 1.
    max_fun_evals : int, optional
        Maximum number of function evaluations before aborting
    warnings: bool
        can disable warnings

    Returns
    -------
    alpha : float
        An acceptable step length
        or, if not converged, the last trial value
    f_new : float
        The function value after the proposed step `f_new = f(z_k + alpha p_k)`
    fun_evals : int
        The number of function evaluations
    has_converged : bool
    """

    assert 0 < c1 < 1

    if not max_fun_evals:
        # set to a ridiculously high value
        max_fun_evals = 10000

    fun_evals = 0
    phi_0 = f_k

    # convenience wrapper
    def evaluate_phi(_alpha, _num_evals):
        return f(list_add_prefactor(z_k, _alpha, p_k)), _num_evals + 1

    # evaluate at alpha_0
    phi_a0, fun_evals = evaluate_phi(alpha_0, fun_evals)
    # already sufficient?
    if phi_a0 <= phi_0 + c1 * alpha_0 * der_phi_0:
        return alpha_0, phi_a0, fun_evals, True

    # quadratic interpolation (since only two points of information: alpha=0 and alpha=alpha_0
    alpha = _quadratic_min_01(phi_0, der_phi_0, alpha_0, phi_a0)
    if alpha is None:
        raise RuntimeError('This should not happen')
    phi_a, fun_evals = evaluate_phi(alpha, fun_evals)
    # sufficient?
    if phi_a <= phi_0 + c1 * alpha * der_phi_0:
        return alpha, phi_a, fun_evals, True

    # loop with cubic interpolation
    alpha_prev, phi_a_prev = alpha_0, phi_a0
    while fun_evals <= max_fun_evals:
        # new trial alpha from interpolation
        _a = _cubic_min_012(phi_0, der_phi_0, alpha_prev, phi_a_prev, alpha, phi_a)
        if _a is not None:
            alpha, alpha_prev, phi_a_prev = _a, alpha, phi_a
        else:
            _a = _quadratic_min_01(phi_0, der_phi_0, alpha_prev, phi_a_prev)
            alpha, alpha_prev, phi_a_prev = _a, alpha, phi_a

        phi_a, fun_evals = evaluate_phi(alpha, fun_evals)

        # sufficient?
        if phi_a <= phi_0 + c1 * alpha * der_phi_0:
            return alpha, phi_a, fun_evals, False

        # resources exceeded?
        if fun_evals > max_fun_evals:
            if warnings:
                warn('Maximum number of function evaluations exceeded. Armijo backtracking is aborting.')
            return alpha, phi_a, fun_evals, False

        # alpha too small
        if alpha < alpha_min:
            warn('Step size became too small. Armijo backtracking is aborting.')
            return alpha, phi_a, fun_evals, False
    if warnings:
        warn('Line search failed to converge')
    return alpha, phi_a, fun_evals, False


def strong_wolfe(f, z_k, p_k, f_k, der_phi_0, alpha_init, alpha_max=100., c1=1e-4, c2=.9,
                 max_fun_evals=30, max_grad_evals=20, warnings=True):
    # return alpha, cost, gkp1, fun_evals, grad_evals, has_converged

    assert 0 < c1 < c2 < 1
    assert 0 < alpha_init < alpha_max

    if not max_fun_evals:
        # set to a ridiculously high value
        max_fun_evals = 10000

    if not max_grad_evals:
        # set to a ridiculously high value
        max_grad_evals = 10000

    phi_0 = f_k

    fun_evals = 0
    grad_evals = 0

    # convenience wrappers
    def evaluate_val(_alpha, _num_fun_evals):
        return f(list_add_prefactor(z_k, _alpha, p_k)), _num_fun_evals + 1

    def evaluate_grad(_alpha, _num_grad_evals):
        jax_grad_f = grad(f)(list_add_prefactor(z_k, _alpha, p_k))
        _der_phi = 2 * np.real(list_dot(jax_grad_f, p_k))
        _g = list_conj(jax_grad_f)  # 2 * ∂f/∂z*
        return _der_phi, _g, _num_grad_evals + 1

    alpha = alpha_init
    phi = None
    gkp1 = None  # g_{k+1} = 2 * (∂f/∂z*)(z_{k+1})
    alpha_prev = 0.
    phi_prev = phi_0
    der_phi_prev = der_phi_0

    for i in range(1, max_fun_evals + 10):
        phi, fun_evals = evaluate_val(alpha, fun_evals)
        if (phi > phi_0 + c1 * alpha * der_phi_0) or (phi >= phi_prev and i > 1):
            alpha, cost, gkp1, _fun_evals, _grad_evals, has_converged = \
                _zoom(alpha_prev, alpha, evaluate_val, evaluate_grad, phi_prev, phi, der_phi_prev, phi_0,
                      der_phi_0, c1, c2, max_fun_evals - fun_evals, max_grad_evals - grad_evals, warnings)
            return alpha, cost, gkp1, _fun_evals + fun_evals, _grad_evals + grad_evals, has_converged

        der_phi, gkp1, grad_evals = evaluate_grad(alpha, grad_evals)
        if np.abs(der_phi) <= - c2 * der_phi_0:
            return alpha, phi, gkp1, fun_evals, grad_evals, True
        if der_phi >= 0:
            alpha, cost, gkp1, _fun_evals, _grad_evals, has_converged = \
                _zoom(alpha, alpha_prev, evaluate_val, evaluate_grad, phi, phi_prev, der_phi, phi_0,
                      der_phi_0, c1, c2, max_fun_evals - fun_evals, max_grad_evals - grad_evals, warnings)
            return alpha, cost, gkp1, _fun_evals + fun_evals, _grad_evals + grad_evals, has_converged

        # new trial alpha
        alpha_prev = alpha
        phi_prev = phi
        der_phi_prev = der_phi
        alpha = min(2 * alpha, alpha_max)  # FUTURE extrapolate instead?

        if alpha_prev >= alpha_max:
            if warnings:
                warn('alpha_max reached in strong_wolfe line search. aborting without convergence')
            break

        if fun_evals > max_fun_evals:
            if warnings:
                warn(f'Maximum number of function evaluations exceeded. Line search is aborting.')
            break

        if grad_evals > max_grad_evals:
            if warnings:
                warn(f'Maximum number of function evaluations exceeded. Line search is aborting.')
            break

    return alpha, phi, gkp1, fun_evals, grad_evals, False


def _zoom(a_lo, a_hi, evaluate_val, evaluate_grad, phi_lo, phi_hi, der_phi_lo, phi_0, der_phi_0,
          c1, c2, max_fun_evals, max_grad_evals, warnings):
    # return alpha, phi, gkp1, fun_evals, grad_evals, has_converged

    d_cubic = 0.2  # cubic interpolation is not trusted if result is in lower or upper 20% of interval
    d_quad = 0.1  # quadratic interpolation is not trusted if result is in lower or upper 10% of interval

    a_j = None
    phi_aj = None
    gkp1 = None
    fun_evals = 0
    grad_evals = 0

    for i in range(max_fun_evals + 10):
        # Interpolation: find a_j between a_lo and a_hi
        interpolation_done = False
        # try cubic interpolation
        if i > 0:
            a_j = _cubic_min_123(a_lo, phi_lo, der_phi_lo, a_hi, phi_hi, a_j, phi_aj)
            gkp1 = None
            threshold = d_cubic * abs(a_lo - a_hi)
            interpolation_done = (a_j is not None) and (abs(a_j - a_lo) < threshold) \
                                 and (abs(a_j - a_hi) < threshold)
        # try quadratic interpolation
        if not interpolation_done:
            a_j = _quadratic_min_12(a_lo, phi_lo, der_phi_lo, a_hi, phi_hi)
            gkp1 = None
            threshold = d_quad * abs(a_lo - a_hi)
            interpolation_done = (a_j is not None) and (abs(a_j - a_lo) < threshold) \
                                 and (abs(a_j - a_hi) < threshold)
        # default to bisection
        if not interpolation_done:
            a_j = (a_hi + a_lo) / 2.
            gkp1 = None

        # Evaluate
        phi_aj, fun_evals = evaluate_val(a_j, fun_evals)

        if (phi_aj > phi_0 + c1 * a_j * der_phi_0) or (phi_aj >= phi_lo):
            a_hi = a_j
        else:
            der_phi_j, gkp1, grad_evals = evaluate_grad(a_j, grad_evals)
            if np.abs(der_phi_j) <= -c2 * der_phi_0:
                return a_j, phi_aj, gkp1, fun_evals, grad_evals, True
            if der_phi_j * (a_hi - a_lo) >= 0:
                a_hi = a_lo
            a_lo = a_j

        if fun_evals > max_fun_evals:
            if warnings:
                warn(f'Maximum number of function evaluations exceeded in zoom phase. Line search is aborting.')
            break

        if grad_evals > max_grad_evals:
            if warnings:
                warn(f'Maximum number of function evaluations exceeded in zoom phase. Line search is aborting.')
            break

    return a_j, phi_aj, gkp1, fun_evals, grad_evals, False


def _quadratic_min_01(f_0, df_0, x_1, f_1):
    """
    Computes the minimiser of the quadratic polynomial q with
    q(0) = f_0  ;  q'(0) = df_0  ;  q(x_1) = f_1

    '01' means points are (0, x_1)
    """
    denominator = 2 * (f_1 - f_0 - df_0 * x_1)
    if denominator == 0:
        return None
    return - (df_0 * (x_1 ** 2)) / denominator


def _quadratic_min_12(x_1, f_1, df_1, x_2, f_2):
    """
    Computes the minimiser of the quadratic polynomial q with
    q(x_1) = f_10  ;  q'(x_1) = df_1  ;  q(x_2) = f_2

    '12' means points are (x_1, x_2)
    """
    # interpolate g(x) = f(x-x_1)
    g_minimiser = _quadratic_min_01(f_1, df_1, x_2 - x_1, f_2)
    if g_minimiser is None:
        return None
    return g_minimiser + x_1


def _cubic_min_012(f_0, df_0, x_1, f_1, x_2, f_2):
    """
    Computes the local minimum of the cubic polynomial c with
    c(0) = f_0  ;  c'(0) = df_0  ;  c(x_1) = f_1  ;  c(x_2) = f_2

    '012' means points are (0, x_1, x_2)
    """
    denominator = (x_1 * x_2) ** 2 * (x_1 - x_2)
    M = np.array([[x_2 ** 2, - x_1 ** 2], [- x_2 ** 3, x_1 ** 3]])
    v = np.array([f_1 - f_0 - df_0 * x_1, f_2 - f_0 - df_0 * x_2])
    A, B = np.dot(M, v) / denominator

    discriminant = B ** 2 - 3 * A * df_0
    if discriminant < 0:
        return None

    return (-B + np.sqrt(discriminant)) / (3 * A)


def _cubic_min_123(x_1, f_1, df_1, x_2, f_2, x_3, f_3):
    """
    Computes the local minimum of the cubic polynomial c with
    c(x_1) = f_1  ;  c'(x_1) = df_1  ;  c(x_2) = f_2  ;  c(x_3) = f_3

    '123' means points are (x_1, x_2, x_3)
    """
    # interpolate g(x) = f(x-x_1)
    g_min = _cubic_min_012(f_1, df_1, x_2 - x_1, f_2, x_3 - x_1, f_3)
    if g_min is None:
        return None
    return g_min + x_1
