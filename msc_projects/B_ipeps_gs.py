from utils.jax_utils.jax_startup import startup

startup()

import jax.numpy as np
from pathlib import Path
import pickle
import os
import sys
from datetime import datetime

from msc_projects.cluster_tools import KwargsParser
from utils.logging import Logger, get_display_fun
from grad_tn.models.two_dim_square_lattice.tfim import TFIM, sx
from grad_tn.tensornetworks.peps import LocalOperator

DEFAULTS = dict(D_opt=10,
                chi=2,
                g_list=[3.5],
                init='last',  # ps, z+, x+, load, last    |   load=load from file with closest g
                max_iter=5000,
                )


def run_simulation(**kwargs):
    kp = KwargsParser(kwargs, DEFAULTS)
    folder = Path(kp.folder).expanduser()
    folder.mkdir(exist_ok=True, parents=True)
    logger = Logger(folder.joinpath(f'chi_{kp.chi}_Dopt_{kp.D_opt}_{kp.init}.log'), True)
    kp.log(logger)

    gs = None

    for g in kp.g_list:
        logger.lineskip()
        logger.log(f'g={g}')

        file_str = f'chi_{kp.chi}_Dopt_{kp.D_opt}_g_{g}_{kp.init}'
        opt_logger = Logger(folder.joinpath(file_str + '.opt.log'), True)
        outfile = folder.joinpath(file_str + '.pkl')

        opt_opts = dict(display_fun=get_display_fun(opt_logger),
                        line_search_fn='strong_wolfe',
                        max_iter=kp.max_iter,
                        dtype=np.float64)
        cont_opts = dict(chi_ctm=kp.D_opt)

        model = TFIM(g, bc='infinite', dtype_hamiltonian=np.float64)

        # initial state
        if kp.init == 'load':
            # find the file with the closest g
            g_closest = np.inf
            file_closest = None
            for f in os.listdir(folder):
                if f[-3:] != 'pkl':
                    continue
                start = f.rfind('_g_')
                if start == -1:
                    continue
                ends = [f.rfind(f'_{init}') for init in ['ps', 'z+', 'x+', 'load']]
                end = [e for e in ends if e != -1][0]
                _g = float(f[start + 3:end])
                if np.abs(g - _g) < np.abs(g - g_closest):  # closer then previous
                    g_closest = _g
                    file_closest = f

            # noinspection PyBroadException
            try:
                with open(folder.joinpath(file_closest), 'rb') as f:
                    results = pickle.load(f)
                init = results['gs']
                initial_noise = 0.
                print(f'loaded initial guess from {file_closest}')
            except Exception:
                if gs:  # if gs is available from previous loop iteration, use that
                    # failed to load even though this is not the first loop iteration
                    print('warning: loading from file failed', file=sys.stderr)
                    init = gs
                    initial_noise = 0.001
                else:  # if nothing found, use product state
                    init = 'ps'
                    initial_noise = 0.05
        elif kp.init == 'last':
            if gs:
                init = gs
                initial_noise = 0.
            else:
                init = 'ps'
                initial_noise = 0.05
        else:
            init = kp.init
            initial_noise = 0.05

        gs, gs_energy = model.groundstate(chi=kp.chi, initial_state=init, initial_noise=initial_noise,
                                          contraction_options=cont_opts, optimisation_options=opt_opts)

        en_list = []
        mag_list = []
        D_obs_list = []

        for D_obs in list(range(kp.D_opt))[10::10] + [kp.D_opt]:
            logger.log(f'D_obs={D_obs}')
            D_obs_list.append(D_obs)
            cont_opts['chi_ctm'] = D_obs
            en_list.append(model.energy(gs, **cont_opts))
            mag_list.append(LocalOperator(sx, np.array([[0]]), hermitian=True).expval(gs, **cont_opts))

        print(f'saving results to {outfile}')

        results = dict(kwargs=kwargs, g=g, optimal_energy=gs_energy,
                       D_obs_list=D_obs_list, en_list=en_list, mag_list=mag_list,
                       logfile=str(opt_logger.logfile),
                       gs_tensors=gs.get_tensors())
        with open(outfile, 'wb') as f:
            pickle.dump(results, f)

        for D_obs in list(range(kp.D_opt, 2 * kp.D_opt))[10::10] + [2 * kp.D_opt]:
            logger.log(f'D_obs={D_obs}')
            D_obs_list.append(D_obs)
            cont_opts['chi_ctm'] = D_obs
            en_list.append(model.energy(gs, **cont_opts))
            mag_list.append(LocalOperator(sx, np.array([[0]]), hermitian=True).expval(gs, **cont_opts))

        results = dict(kwargs=kwargs, g=g, optimal_energy=gs_energy,
                       D_obs_list=D_obs_list, en_list=en_list, mag_list=mag_list,
                       logfile=str(opt_logger.logfile),
                       gs_tensors=gs.get_tensors())

        with open(outfile, 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    folder = '~/Desktop/tmp_runs/B_ipeps_gs/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    run_simulation(folder=folder, max_iter=5)
