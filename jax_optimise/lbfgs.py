import warnings
from typing import Union, Optional

from utils.warnings import custom_warn

warnings.showwarning = custom_warn

from jax import partial, value_and_grad
import jax.numpy as np

from jax_optimise.scheduling import make_schedule
from jax_optimise.list_util import list_neg, list_conj, list_add_prefactor, list_dot, \
    list_scale, list_norm
from jax_optimise.line_search import armijo_backtracking, strong_wolfe

# line-search keywords
ARMIJO = 'armijo_backtracking'
STRONG_WOLFE = 'strong_wolfe'

# for csv-style display
CSV_SEPARATOR = '  '
COL_WIDTH = 20
SLIM_WIDTH = 5


def _disp_csv(display_fun: callable, values: list, slim=None):
    if slim is None:
        slim = [False for _ in values]

    msg = ''
    for val, s in zip(values, slim):
        width = SLIM_WIDTH if s else COL_WIDTH
        if type(val) == int:
            s = f'{val:d}'.rjust(width)[:width]
        elif type(val) == str:
            s = val.ljust(width)[:width]
        else:
            try:
                s = f'{val:.20f}'.rjust(width)[:width]
            except ValueError:
                s = str(val).rjust(width)[:width]
        msg = msg + s + CSV_SEPARATOR
    msg = msg[:-2]  # remove the last two spaces
    display_fun(msg)


def get_disp(maximising, display_fun):
    def disp(n_iter, cost, cost_change, grad_norm, step_dist, alpha, num_evals, num_grad_evals):
        if maximising:
            if cost_change != 'N\\A':
                cost_change = - cost_change
            if cost != 'N\\A':
                cost = - cost
        vals = [n_iter, cost, cost_change, grad_norm, step_dist, alpha, num_evals, num_grad_evals]
        slim = [True, False, False, False, False, False, True, True]
        _disp_csv(display_fun, vals, slim)

    return disp


def minimise(cost_function, initial_guess,
             maximising: bool = False,
             display_fun: callable = partial(print, flush=True),
             callback: Optional[callable] = None,
             cost_print_name: str = 'Cost',
             step_size: Union[callable, float] = 1.,
             step_size_max: float = 100.,
             history_size: int = 20,
             max_iter: Optional[int] = None,
             max_fun_evals: Optional[int] = None,
             max_grad_evals: Optional[int] = None,
             tolerance_grad: float = 1e-7,
             tolerance_change: float = 1e-11,
             line_search_fn: str = STRONG_WOLFE,
             c1: float = 1e-4,
             c2: float = 0.9,
             max_fun_evals_ls: Optional[int] = 200,
             max_grad_evals_ls: Optional[int] = 200,
             abort_on_failed_curvature: bool = False,
             abort_on_failed_armijo: bool = True,
             **kwargs):
    """
    Algorithm-specific version of jax_optimise.main._minimise
    Operates on flat function inputs only, i.e. lists of arrays or scalars

    Parameters
    ----------
    cost_function : callable
        The function to be minimised.  `cost_function(x) -> float`
        where `x` is a list of (arrays or scalars)
    initial_guess : list of (array or float or complex)
    maximising : bool, optional
        If minimisation is used to maximise something.
        If so, logs are adjusted with a sign. The output is unchanged.
    display_fun : callable or None, optional
        Function that is called to display convergence info.
        or `None`: no display. Default: built-in `print` function
    callback : callable, optional
        Function that is called as `callback(xk)` for every iterate `xk`, including the initial guess.
        or `None` (default): No callback.
        This allows, e.g., monitoring of the convergence,
    cost_print_name : str, optional
        A string representation for the name of cost-function values for logging.
        Default: `Cost`
    step_size : callable or float, optional
        Either a float (the step size) or a function `step_size(k: int) -> float` of the iteration number
        See jax_optimise.scheduling for related tools.
        For no line search: determines the step length
        For line search: gives the initial guess for the step length
    step_size_max : float, optional
        Only relevant for strong_wolfe line search.
        The maximum step length.
    history_size : int, optional
        Number of vectors stored to approximate inverse hessian.
        Leading contribution to memory cost is `2 * history_size * vect_mem`
        where `vect_mem` is the memory cost of an object like `initial_guess`
    max_iter : int, optional
        Maximum number of iterations of the main loop (number of steps).
        Default: if max_fun_evals or max_grad_evals are set, they determine when the loop ends.
            else defaults to 10'000
    max_fun_evals : int, optional
        Maximum number of times the `cost_function` will be called
        Default: no restriction
    max_grad_evals : int, optional
        Maximum number of times the gradient of `cost_function` will be computed via `jax.value_and_grad`
        Default: no restriction
    tolerance_grad : float
        Gradient entries below this threshold are considered negligible
    tolerance_change : float
        Changes in cost or in parameter space below this threshold are considered negligible
    line_search_fn : str
        Line search method. Currently available:

            - `None`                    no line search, just use `step_size`
            - 'armijo_backtracking'     step lengths will be <= `step_size` and fulfill the armijo condition only
                                            requires only function-values, no gradients.
                                            `step_size` will be accepted as often as possible.
            - 'strong_wolfe'            step lengths will be <= `step_size_max` and fulfill both strong wolfe conditions
                                            requires function values and gradients.
                                            `step_size` will be accepted as often as possible.

        Strongly recommend 'strong_wolfe' to guarantee convergence
    c1 : float
        The Parameter in the Armijo condition.
        Must fulfill `0 < c1 < c2 < 1`. Typically much smaller than `1`.
    c2 : float
        The Parameter in the 2nd Wolfe condition (curvature condition).
        Must fulfill `0 < c1 < c2 < 1`. Typically is not much smaller than `1`.
    max_fun_evals_ls : int, optional
        Maximum number of function evaluations per line search.
        Default: no restriction.
    max_grad_evals_ls : int, optional
        Maximum number of gradient evaluations per line search.
        Default: no restriction.
    abort_on_failed_curvature : bool
        Whether a failure to find a step-length that fulfills the curvature condition should
        trigger the algorithm to abort.
        This is only relevant for strong_wolfe line search.
    abort_on_failed_armijo : bool
        Whether a failure to find a step-length that fulfills the armijo condition should
        trigger the algorithm to abort.
        This is only relevant for strong_wolfe and armijo_backtracking line search
    kwargs : dict
        Buffer for additional unintended kwargs.
        A warning (but no error) is raised if there are any.

    Returns
    -------
    x_optim : list of (array or float or complex)
    cost_optim : float
        Note that for maximising, no sign correction is made here
    info : dict
        Convergence info:

            converged : bool
                If the algorithm was successful
            reason : str
                A short description of why the algorithm terminated

    """

    if display_fun is None:
        def display_fun(_):
            pass
    disp = get_disp(maximising, display_fun)

    if kwargs:
        warnings.warn(f'There were unexpected keyword arguments: {[key for key in kwargs.keys()]}. They were ignored.')

    if (not max_iter) and (not max_fun_evals) and (not max_grad_evals):
        # need some termination condition to avoid infinite loops. set max_iter very high.
        max_iter = 10000

    if line_search_fn in [None, ARMIJO]:
        warnings.warn(f'Linesearch method {line_search_fn} is discouraged for L-BFGS')

    if not (0 < c1 < c2 < 1):
        raise ValueError(f'Expected 0 < c1 < c2 < 1. Got c1={c1}, c2={c2}')

    step_size = make_schedule(step_size)

    def evaluate(_x):
        """
        Returns: fk, gk
        fk = f(zk)
        gk = 2 * ∂f/∂z* (zk)
        """
        _val, jax_grad = value_and_grad(cost_function)(_x)
        _g = list_conj(jax_grad)
        return _val, _g

    # display header line
    vals = ['k', cost_print_name, f'{cost_print_name} change', 'Gradient Norm', 'Step distance', 'alpha', '# f', '# df']
    slim = [True, False, False, False, False, False, True, True]
    _disp_csv(display_fun, vals, slim)
    display_fun('-' * 130)

    fk, gk = evaluate(initial_guess)
    grad_norm = list_norm(gk)
    num_fun_evals = 1
    num_grad_evals = 1
    n_iter = 0

    # display initial evaluation
    disp(0, fk, 'N\\A', grad_norm, 'N\\A', 'N\\A', 'N\\A', 'N\\A')

    # callback
    if callback:
        callback(initial_guess, 0)

    # already optimal?
    if grad_norm <= tolerance_grad:
        display_fun('The initial guess is already optimal.')
        return initial_guess, fk, dict(converged=True, reason='gradient_vanishes')

    # initialise variables for main loop
    xk = initial_guess[:]
    y_history = []
    s_history = []
    rho_history = []
    gamma = 1.

    while (not max_iter) or (n_iter <= max_iter):
        # if (not max_iter), the loop is terminated either by max_fun_evals or max_grad_evals

        # --------------------------------
        #    compute descent direction
        # --------------------------------
        pk = list_neg(gk)
        current_history_size = len(y_history)
        a_list = [np.nan] * current_history_size

        for i in range(current_history_size - 1, -1, -1):  # iterate over history newest to oldest
            a_list[i] = rho_history[i] * np.real(list_dot(list_conj(s_history[i]), pk))
            pk = list_add_prefactor(pk, -a_list[i], y_history[i])

        pk = list_scale(pk, gamma)

        for i in range(current_history_size):  # oldest to newest
            b = rho_history[i] * np.real(list_dot(list_conj(y_history[i]), pk))
            pk = list_add_prefactor(pk, a_list[i] - b, s_history[i])

        # --------------------------------
        #    check directional derivative
        # --------------------------------
        der_phi_0 = np.real(list_dot(list_conj(gk), pk))
        if np.abs(der_phi_0) < np.abs(tolerance_change):
            display_fun('Directional derivative vanishes up to tolerance. Aborting.')
            return xk, fk, dict(converged=True, reason='dir_deriv_vanishes')

        if der_phi_0 > tolerance_change:
            display_fun('[WARNING] Not a descent direction')

        # --------------------------------
        #       compute step length
        # --------------------------------

        # initial guess
        alpha_init = step_size(n_iter)
        if n_iter == 0:
            alpha_init = min(alpha_init, alpha_init / grad_norm)

        # determine resources
        _max_fun_evals_ls = None
        if max_fun_evals:
            _max_fun_evals_ls = max_fun_evals - num_fun_evals
        if max_fun_evals_ls:
            if _max_fun_evals_ls:
                _max_fun_evals_ls = min(_max_fun_evals_ls, max_fun_evals_ls)
            else:
                _max_fun_evals_ls = max_fun_evals_ls
        _max_grad_evals_ls = None
        if max_grad_evals:
            _max_grad_evals_ls = max_grad_evals - num_grad_evals
        if max_grad_evals_ls:
            if _max_grad_evals_ls:
                _max_grad_evals_ls = min(_max_grad_evals_ls, max_grad_evals_ls)
            else:
                _max_grad_evals_ls = max_grad_evals_ls

        # call algorithm specific function
        if line_search_fn is None:
            fun_evals_ls = 0
            grad_evals_ls = 0
            alpha_k = alpha_init
            fkp1 = None
            gkp1 = None

        elif line_search_fn in [ARMIJO, 'armijo']:
            alpha_k, fkp1, fun_evals_ls, has_converged = \
                armijo_backtracking(cost_function, xk, pk, fk, der_phi_0, alpha_init, c1=c1,
                                    max_fun_evals=_max_fun_evals_ls)
            grad_evals_ls = 0
            num_fun_evals += fun_evals_ls
            gkp1 = None

            if not has_converged:
                if abort_on_failed_armijo:
                    display_fun('Line search did not converge. aborting.')
                    return xk, fk, dict(converged=False, reason='line_search_failed')
                alpha_k = alpha_init
                fkp1 = None

        elif line_search_fn in [STRONG_WOLFE, 'strong wolfe']:
            alpha_k, fkp1, gkp1, fun_evals_ls, grad_evals_ls, has_converged = \
                strong_wolfe(cost_function, xk, pk, fk, der_phi_0, alpha_init, c1=c1, c2=c2,
                             max_fun_evals=_max_fun_evals_ls, max_grad_evals=_max_grad_evals_ls,
                             alpha_max=step_size_max)
            num_fun_evals += fun_evals_ls
            num_grad_evals += grad_evals_ls

            if not has_converged:
                if abort_on_failed_curvature:
                    display_fun('Line search did not converge. aborting.')
                    return xk, fk, dict(converged=False, reason='line_search_failed')

                # if we can come up with a step that at least fulfills armijo, we can possibly continue

                if not (fkp1 <= fk + c1 * alpha_k * der_phi_0):  # else alpha_k fulfills armijo and can be used
                    # check alpha_init
                    _f_alpha_init = cost_function(list_add_prefactor(xk, alpha_init, pk))
                    num_fun_evals += 1
                    fun_evals_ls += 1
                    if not (_f_alpha_init <= fk + c1 * alpha_init * der_phi_0):
                        if abort_on_failed_armijo:
                            display_fun('Line search did not converge. aborting.')
                            return xk, fk, dict(converged=False, reason='line_search_failed')
                        # both dont fulfill armijo, but we do not abort. check if alpha_init is better than alpha_k
                        if _f_alpha_init < fkp1:
                            alpha_k = alpha_init
                            fkp1 = _f_alpha_init
                            gkp1 = None

        else:
            raise ValueError(f'Unknown Line search function: {line_search_fn}')

        # --------------------------------
        #    compute values at new point
        # --------------------------------
        xkp1 = list_add_prefactor(xk, alpha_k, pk)
        if (fkp1 is None) or (gkp1 is None):
            if gkp1 is None:
                fkp1, gkp1 = evaluate(xkp1)
                num_grad_evals += 1
                num_fun_evals += 1
                grad_evals_ls += 1
                fun_evals_ls += 1
            else:
                fkp1 = cost_function(xkp1)
                num_fun_evals += 1
                fun_evals_ls += 1

        sk = list_scale(pk, alpha_k)
        yk = list_add_prefactor(gkp1, -1, gk)
        rho_k_inv = np.real(list_dot(list_conj(yk), sk))

        distance = list_norm(sk)
        cost_change = fkp1 - fk

        # update values
        xk, xkp1 = xkp1, None
        gk, gkp1 = gkp1, None
        grad_norm = list_norm(gk)
        fk, fkp1 = fkp1, None
        n_iter += 1

        # display
        disp(n_iter, fk, cost_change, grad_norm, distance, alpha_k, fun_evals_ls, grad_evals_ls)

        # CALLBACK
        if callback:
            callback(xk, n_iter)

        # --------------------------------
        #       update history
        # --------------------------------
        if rho_k_inv < 1e-12:
            display_fun('Skipping L-BFGS update')
        else:
            if len(s_history) == history_size:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

            s_history.append(sk)
            y_history.append(yk)
            rho_history.append(1. / rho_k_inv)
            gamma = rho_k_inv / np.real(list_dot(list_conj(yk), yk))

        # --------------------------------
        #       check conditions
        # --------------------------------
        # optimal?
        if grad_norm <= tolerance_grad:
            display_fun(f'The gradient vanishes up to numerical tolerance -> Optimum found')
            return xk, fk, dict(converged=True, reason='gradient_vanishes')

        # resources exceeded?
        if max_iter and (n_iter >= max_iter):
            display_fun(f'[WARNING] Maximum number of iterations reached before optimisation has converged.')
            return xk, fk, dict(converged=False, reason='max_iter')
        if max_fun_evals and (num_fun_evals >= max_fun_evals):
            display_fun(f'[WARNING] Maximum number of function evaluations exceeded before optimisation has '
                        f'converged.')
            return xk, fk, dict(converged=False, reason='fun_evals')
        if max_grad_evals and (num_grad_evals >= max_grad_evals):
            display_fun(f'[WARNING] Maximum number of gradient evaluations exceeded before optimisation has '
                        f'converged.')
            return xk, fk, dict(converged=False, reason='grad_evals')

        # lack of progress?
        if alpha_k <= 0:
            display_fun(f'[WARNING] alpha <= 0. This should not happen.')
            return xk, fk, dict(converged=False, reason='negative_alpha')
        if distance < tolerance_change:
            display_fun('Lack of progress in parameter space.')
            return xk, fk, dict(converged=True, reason='x_progress')
        if np.abs(cost_change) < tolerance_change:
            display_fun('Lack of progress in cost.')
            return xk, fk, dict(converged=False, reason='cost_progress')

    raise RuntimeError('This line should be unreachable')
