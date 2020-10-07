import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_multimap

from jax_optimise import lbfgs, gradient_descent

LBFGS = 'L-BFGS'
GD = 'gradient descent'


def maximise(cost_function, initial_guess, algorithm, dtype=None, **options):
    """
    Calculates the inputs that maximise a real-valued cost function.

    The `cost_function` is considered either a function of real variables (and the real maximiser is returned)
    or as a function of complex variables (and the complex maximiser is returned), depending on `dtype`

    The `cost_function` must take a single argument, which can be an arbitrarily nested container structure
    of real or complex (or mixed) scalars and/or arrays.
    e.g. a dict of lists of array, or just a single array or number.

    The cost function must be traceable by jax.grad


    Parameters
    ----------
    cost_function : callable
        The function to be maximised

            ``cost_function(x) -> float``

        where `x` is an arbitrary pytree of scalars and/or arrays
    initial_guess : pytree
        The initial best guess. `cost_function(initial_guess)` must be valid.
    algorithm : str, optional
        Which algorithm to be uses. currently supported:

            - 'L-BFGS'  see lbfgs.py
            - 'GD'      see gradient_descent.py

        defaults to 'L-BFGS'
    dtype : jax.numpy.dtype or pytree, optional
        The datatype for the arguments of `cost_function`.
        Note that this determines if `cost_function` is considered a function of real or of complex variables.
        Either a single `jax.numpy.dtype` used for all entries, or a pytree of matching the structure of `initial_guess`
        or None: per default, the dtype of `initial_guess` is used.
    options : dict
        Algorithm specific keyword arguments.
        See the docstring of the corresponding algorithm-specific function (e.g. `jax_optimise.lbfgs.minimise`)
        for details.
        All algorithms accept these generic options:

            max_iter : int or None
                Maximum number of iterations. Default: algorithm specific.
                Note that the meaning of an iteration, and in particular the number of calls per iteration of
                `cost_function` or its gradient can be different for different algorithms.
            max_fun_evals : int or None
                Maximum number of function evaluations.
                Default: no restriction.
            max_gad_evals : int or None
                Maximum number of gradient evaluations.
                Default: no restriction.
            cost_print_name : str
                A string representation for the name of `cost_function` function-values for logging.
                Default: `Cost`
            display_fun : callable or None
                Function that is called to display convergence info after each iteration.
                or `None`: no display.
                Default: built-in `print` function
            callback : callable or None
                Function that is called as `callback(xk, k)` for every iterate `xk`, with iteration number `k`
                including the initial guess (k=0).
                or `None` (default): No callback.
                This allows, e.g., monitoring of the convergence.

    Returns
    -------
    x_optim : pytree
        Same structure as `initial_guess`. The optimal input to `cost_function`, that maximises it
    cost_optim : float
        The optimal value `cost_function(x_optim)`
    info : dict
        A dictionary of convergence info. Keys are algorithm-specific but generically include

            converged : bool
                If the algorithm was successful
            reason : str
                A short description of why the algorithm terminated

    """
    return _minimise(cost_function, initial_guess, algorithm, dtype, maximising=True, **options)


def minimise(cost_function, initial_guess, algorithm, dtype=None, **options):
    """
    Calculates the inputs that minimise a real-valued cost function.

    The `cost_function` is considered either a function of real variables (and the real minimiser is returned)
    or as a function of complex variables (and the complex minimiser is returned), depending on `dtype`

    The `cost_function` must take a single argument, which can be an arbitrarily nested container structure
    of real or complex (or mixed) scalars and/or arrays.
    e.g. a dict of lists of array, or just a single array or number.

    The cost function must be traceable by jax.grad


    Parameters
    ----------
    cost_function : callable
        The function to be minimised

            ``cost_function(x) -> float``

        where `x` is an arbitrary pytree of scalars and/or arrays
    initial_guess : pytree
        The initial best guess. `cost_function(initial_guess)` must be valid.
    algorithm : str, optional
        Which algorithm to be uses. currently supported:

            - 'L-BFGS'  see lbfgs.py
            - 'GD'      see gradient_descent.py

        defaults to 'L-BFGS'
    dtype : jax.numpy.dtype or pytree, optional
        The datatype for the arguments of `cost_function`.
        Note that this determines if `cost_function` is considered a function of real or of complex variables.
        Either a single `jax.numpy.dtype` used for all entries, or a pytree of matching the structure of `initial_guess`
        or None: per default, the dtype of `initial_guess` is used.
    options : dict
        Algorithm specific keyword arguments.
        See the docstring of the corresponding algorithm-specific function (e.g. `jax_optimise.lbfgs.minimise`)
        for details.
        All algorithms accept these generic options:

            max_iter : int or None
                Maximum number of iterations. Default: algorithm specific.
                Note that the meaning of an iteration, and in particular the number of calls per iteration of
                `cost_function` or its gradient can be different for different algorithms.
            max_fun_evals : int or None
                Maximum number of function evaluations.
                Default: no restriction.
            max_gad_evals : int or None
                Maximum number of gradient evaluations.
                Default: no restriction.
            cost_print_name : str
                A string representation for the name of cost-function values for logging.
                Default: `Cost`
            display_fun : callable or None
                Function that is called to display convergence info after each iteration.
                or `None`: no display.
                Default: built-in `print` function
            callback : callable or None
                Function that is called as `callback(xk, k)` for every iterate `xk`, with iteration number `k`
                including the initial guess (k=0).
                or `None` (default): No callback.

    Returns
    -------
    x_optim : pytree
        Same structure as `initial_guess`. The optimal input to `cost_function`, that minimises it
    cost_optim : float
        The optimal value `cost_function(x_optim)`
    info : dict
        A dictionary of convergence info. Keys are algorithm-specific but generically include

            converged : bool
                If the algorithm was successful
            reason : str
                A short description of why the algorithm terminated

    """
    return _minimise(cost_function, initial_guess, algorithm, dtype, maximising=False, **options)


def _minimise(cost_function, initial_guess, algorithm, dtype=None, maximising=False, **options):
    """
    Internal version of `minimise`.
    Allows interface for maximisation.
    In that case, the negative of the cost_function is minimised and the sign is readjusted in
    logs and output
    """

    if dtype:
        if tree_flatten(dtype)[1] == tree_flatten(np.float32)[1]:  # not a tree
            initial_guess = tree_map(lambda arr: np.asarray(arr, dtype=dtype), initial_guess)
        else:
            initial_guess = tree_multimap(lambda arr, dt: np.asarray(arr, dtype=dt), initial_guess, dtype)

    initial_guess_flat, tree_def = tree_flatten(initial_guess)

    if maximising:
        def cost_function_flat(x_flat):
            x = tree_unflatten(tree_def, x_flat)
            return - cost_function(x)
    else:
        def cost_function_flat(x_flat):
            x = tree_unflatten(tree_def, x_flat)
            return cost_function(x)

    if 'callback' in options:
        callback = options['callback']

        def callback_flat(x_flat, k):
            x = tree_unflatten(tree_def, x_flat)
            callback(x, k)

        options['callback'] = callback_flat

    if algorithm == LBFGS:
        x_optim_flat, cost_optim, info = lbfgs.minimise(cost_function_flat, initial_guess_flat, maximising=maximising,
                                                        **options)
    elif algorithm == GD:
        x_optim_flat, cost_optim, info = gradient_descent.minimise(cost_function_flat, initial_guess_flat,
                                                                   maximising=maximising, **options)
    else:
        raise ValueError(f'Algorithm Keyword "{algorithm}" is not a valid keyword or the algorithm is not implemented')

    x_optim = tree_unflatten(tree_def, x_optim_flat)
    if maximising:
        cost_optim = - cost_optim

    return x_optim, cost_optim, info
