from utils.jax_utils.jax_startup import startup

startup()

import jax.numpy as np
from pathlib import Path
import pickle
from datetime import datetime

from msc_projects.cluster_tools.misc import KwargsParser
from utils.logging import Logger, get_display_fun
from grad_tn.models.two_dim_square_lattice.tfim import TFIM

DEFAULTS = dict(contraction_method='brute',
                g=3.5,
                L=3,
                chi=2,
                initial_state='ps',
                initial_noise=True,
                max_iter=5000,
                save_interval=50,
                )


def run_simulation(**kwargs):
    kp = KwargsParser(kwargs, DEFAULTS)
    folder = Path(kp.folder).expanduser()
    folder.mkdir(exist_ok=True, parents=True)

    file_str = f'L_{kp.L}_chi_{kp.chi}_g_{kp.g}_{kp.contraction_method}'
    logger = Logger(None, True)
    opt_logger = Logger(folder.joinpath(file_str + '.opt.log'), True)
    kp.log(opt_logger)
    opt_logger.lineskip()
    outfile = folder.joinpath(file_str + '.pkl')
    statefile = folder.joinpath(file_str + '.state.pkl')
    kp.log(logger)

    def callback(tensors, k):
        if (k % kp.save_interval == 0) and (k > 0):
            with open(statefile, 'wb') as _f:
                pickle.dump(dict(kwargs=kp.kwargs(), k=k, tensors=tensors), _f)

    opt_opts = dict(display_fun=get_display_fun(opt_logger),
                    line_search_fn='strong_wolfe',
                    max_iter=kp.max_iter,
                    callback=callback,
                    dtype=np.complex128)
    cont_opts = dict(contraction_method=kp.contraction_method)

    model = TFIM(kp.g, 'obc', lx=kp.L, ly=kp.L, dtype_hamiltonian=np.float64)
    gs, gs_energy = model.groundstate(kp.chi, (kp.L, kp.L), kp.initial_state, kp.initial_noise,
                                      cont_opts, opt_opts)

    results = dict(kwargs=kp.kwargs(),
                   gs_energy=gs_energy,
                   gs_tensors=gs.get_tensors(),
                   logfile=str(logger.logfile))

    print(f'saving results to {outfile}')
    # noinspection PyTypeChecker
    with open(outfile, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    folder = '~/Desktop/tmp_runs/A_peps_gs/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    run_simulation(folder=folder, max_iter=5)
