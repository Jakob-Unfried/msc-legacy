from jax import partial


def minimise(cost_function, initial_guess,
             maximising=False,
             display_fun=partial(print, flush=True),
             callback=None,
             cost_print_name='Cost',
             step_size=1.,
             max_iter=10000,
             tolerance_grad=1e-7,
             tolerance_change=1e-11,
             **kwargs):
    raise NotImplementedError
