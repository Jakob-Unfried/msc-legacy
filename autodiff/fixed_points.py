from typing import Optional
from warnings import warn

import jax.numpy as np
from jax import tree_multimap, tree_flatten, lax, partial, custom_vjp, vjp
from jax.lax import while_loop
from jax.tree_util import pytree
from utils.jax_utils import lax_pure_python
from utils.jax_utils.tree_util import tree_zeros_like

DEFAULT_TOLERANCE = 1e-6


def make_convergence_condition(tolerance=DEFAULT_TOLERANCE):
    """
    Returns the default convergence condition (see arg of `fixed_point`
    The condition is that all entries of `x` and `x_last` differ by at most `tolerance`
    """

    def convergence_condition(_, x, x_last):
        close_tree = tree_multimap(lambda arr1, arr2: np.all(np.abs(arr1 - arr2) < tolerance), x, x_last)
        return np.all(tree_flatten(close_tree)[0])

    return convergence_condition


def fixed_point_novjp(f: callable, a, x_init, convergence_condition: Optional[callable] = None,
                      max_iter: Optional[int] = 100, warn_max_iter: bool = True):
    """
    same as fixed_point, but does not implement the vjp rule.
    the vjp rule fails, for example, in CTMRG, where the environment
    is only a fixed point up to the gauge freedom on the virtual bonds.

    i.e. the relation x = f(a, x) is in general not true.
    A CTMRG environment is "converged" if the contracted environment does not change under a renormalisation step.
    Due to the gauge freedom, this does not imply that the tensors (C and T) are unchanged.

    Parameters
    ----------
    f : callable
        The iterative map `x = f(a, x)`
    a
        The parameter of the iterative map
    x_init
        The initial guess for the fixed point
    convergence_condition : callable, optional
        `convergence_condition(a: pytree, x: pytree, x_last: pytree) -> bool`
        checks if a given `x` is sufficiently converged.
        or None, then the default condition via `make_convergence_condition()` is used
    max_iter : int
        Maximum number of iterations.
        A warning is issued, if the loop is aborted due to iteration number.
        This is also used for the iteration in the backwards pass.
    warn_max_iter : bool
        Whether to issue a warning, when max_iter is reached

    Returns
    -------
    x_star : pytree
        The fixed point, s.t. `x_star == f(a, x_star)`
    """
    if not convergence_condition:
        convergence_condition = make_convergence_condition()

    def body_fun(carry):
        _n, _, x = carry
        return lax.add(_n, 1), x, f(a, x)

    def cond_fun(carry):
        _n, x_last, x = carry
        converged = convergence_condition(a, x, x_last)
        not_exceeded = lax.lt(_n, max_iter)
        return np.logical_and(np.logical_not(converged), not_exceeded)

    # carry = (n, x_n, x_{n+1})
    carry_init = (0, x_init, f(a, x_init))
    # need lax_pure python since lax.while_loop is not suitable for reverse mode autodiff
    carry_final = lax_pure_python.while_loop(cond_fun, body_fun, carry_init)
    n, _, x_star = carry_final
    if warn_max_iter and (n == max_iter):
        print('[WARNING] jax_utils.autodiff.fixed_points has reached maximum number of iterations', flush=True)
        warn('Max iter reached')
    return x_star


@partial(custom_vjp, nondiff_argnums=(0, 3, 4, 5))
def fixed_point(f: callable, a, x_init, convergence_condition: Optional[callable] = None,
                max_iter: Optional[int] = 100, warn_max_iter: bool = True):
    """
    Finds the fixed point `x_star` of an iterative map `x = f(a, x)`, such that `x_star == f(a, x_star)`
    The fixed point `x_star` is an implicit function of the "parameter" `a` and differentiable w.r.t `a`.

    Is differentiable w.r.t `a`, and technically also w.r.t `x_init` (though this derivative is 0.)

    Parameters
    ----------
    f : callable
        The iterative map `x = f(a, x)`
    a
        The parameter of the iterative map
    x_init
        The initial guess for the fixed point
    convergence_condition : callable, optional
        `convergence_condition(a: pytree, x: pytree, x_last: pytree) -> bool`
        checks if a given `x` is sufficiently converged.
        or None, then the default condition via `make_convergence_condition()` is used
    max_iter : int
        Maximum number of iterations.
        A warning is issued, if the loop is aborted due to iteration number.
        This is also used for the iteration in the backwards pass.
    warn_max_iter : bool
        Whether to issue a warning, when max_iter is reached

    Returns
    -------
    x_star : pytree
        The fixed point, s.t. `x_star == f(a, x_star)`
    """

    if not convergence_condition:
        convergence_condition = make_convergence_condition()

    def body_fun(carry):
        _n, _, x = carry
        return lax.add(_n, 1), x, f(a, x)

    def cond_fun(carry):
        _n, x_last, x = carry
        converged = convergence_condition(a, x, x_last)
        not_exceeded = lax.lt(_n, max_iter)
        return np.logical_and(np.logical_not(converged), not_exceeded)

    # carry = (n, x_n, x_{n+1})
    carry_init = (0, x_init, f(a, x_init))
    carry_final = while_loop(cond_fun, body_fun, carry_init)
    n, _, x_star = carry_final
    if warn_max_iter and (n == max_iter):
        print('[WARNING] jax_utils.autodiff.fixed_points has reached maximum number of iterations', flush=True)
        warn('Max iter reached')
    return x_star


def _fixed_point_fwd(f, a, x_init, convergence_condition, max_iter, warn_max_iter):
    x_star = fixed_point(f, a, x_init, convergence_condition, max_iter, warn_max_iter)
    res = (a, x_star)
    return x_star, res


# noinspection PyUnusedLocal
def _fixed_point_bwd(f, convergence_condition, max_iter, warn_max_iter, res, x_star_bar):
    a, x_star = res
    _, vjp_a = vjp(lambda _a: f(_a, x_star), a)  # vjp_a(cotangents_out) -> cotangent_in
    update = partial(_bwd_iter, f)
    packed = (a, x_star, x_star_bar)
    u_init = x_star_bar
    wT = fixed_point(update, packed, u_init, max_iter=max_iter, warn_max_iter=warn_max_iter)

    a_bar, = vjp_a(wT)
    x_init_bar = tree_zeros_like(x_star)

    return a_bar, x_init_bar


def _bwd_iter(f, packed, u):
    # map uT -> vT + uT * A    ;    A = (∂_x f)(a, x_star)

    a, x_star, vT = packed
    _, vjp_x = vjp(lambda x: f(a, x), x_star)
    uT_A = vjp_x(u)[0]
    return tree_multimap(lambda arr1, arr2: arr1 + arr2, vT, uT_A)


fixed_point.defvjp(_fixed_point_fwd, _fixed_point_bwd)
