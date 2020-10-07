from utils.jax_utils.jax_startup import startup

startup()

import jax.numpy as np
import numpy as onp
from pathlib import Path
import pickle
from datetime import datetime

from msc_projects.cluster_tools import KwargsParser
from utils.logging import Logger, get_display_fun
from grad_tn.models.spin_one_half import sx, sy, sz, sp, sm
from grad_tn.models.two_dim_square_lattice.tfim import TFIM, TimeEvolution, x_snapshot, y_snapshot, z_snapshot
from grad_tn.tensornetworks.peps import SingleSiteOperator, correlator_timeslice, Peps

DEFAULTS = dict(g=5.,
                L=5,
                chi=2,
                gs_file=None,
                dt=0.01,
                n_steps=1000,
                quench='X',  # X, Y, Z, +
                task_id=None,
                max_iter=1000,
                initial='u_site',  # old, u_site, approximate
                save_all_peps=False,
                tolerance_grad=1e-7,
                )


def run_simulation(**kwargs):
    kp = KwargsParser(kwargs, DEFAULTS)
    folder = Path(kp.folder).expanduser()
    folder.mkdir(exist_ok=True, parents=True)

    file_str = f'L_{kp.L}_g_{kp.g}_chi_{kp.chi}_dt_{kp.dt}_quench_{kp.quench}'
    if kp.task_id:
        file_str += f'_{kp.task_id}'
    logger = Logger(folder.joinpath(file_str + '.log'), True)
    opt_logger = Logger(folder.joinpath(file_str + '.opt.log'), True)
    outfile = folder.joinpath(file_str + '.pkl')
    kp.log(logger)

    opt_opts = dict(display_fun=get_display_fun(opt_logger), line_search_fn='strong_wolfe', max_iter=kp.max_iter,
                    tolerance_grad=kp.tolerance_grad)
    cont_opts = dict(contraction_method='brute')

    model = TFIM(kp.g, bc='obc', lx=kp.L, ly=kp.L, dtype_hamiltonian=np.float64)
    evolver = TimeEvolution(kp.g, kp.dt, 'obc', real_time=True, lx=kp.L, ly=kp.L, pepo_dtype=np.complex128)

    logger.log(f'Starting with groundstate of g={kp.g} TFIM')

    # Prepare groundstate

    gs = None
    gs_energy = None

    if kp.gs_file:
        logger.log('GS file specified, loading GS from file')
        try:
            with open(kp.gs_file, 'rb') as f:
                res = pickle.load(f)
            gs_tensors = res['gs_tensors']
            gs = Peps(gs_tensors, 'obc')
            gs_energy = res['gs_energy']

            assert np.allclose(kp.g, res['kwargs']['g'])
            assert gs.lx == kp.L
            assert gs.ly == kp.L
        except Exception as e:
            logger.log('Failed to load GS from file. Error: ' + str(e))

    if (gs is None) or (gs_energy is None):
        logger.log('No GS file specified, optimising gs...')
        gs, gs_energy = model.groundstate(kp.chi, (kp.L, kp.L), 'ps', 0.05, cont_opts, opt_opts)

        logger.log('Saving GS to ' + str(folder.joinpath(file_str + '.gs.pkl')))
        results = dict(kwargs=kp.kwargs(), gs=gs, gs_energy=gs_energy, g=kp.g)
        with open(folder.joinpath(file_str + '.gs.pkl'), 'wb') as f:
            pickle.dump(results, f)

    # Prepare quench

    if kp.quench == 'X':  # <Sx(r,t) Sx(center,0)>
        quench_operator = sx
        measure_operator = sx
    elif kp.quench == 'Y':  # <Sy(r,t) Sy(center,0)>
        quench_operator = sy
        measure_operator = sy
    elif kp.quench == 'Z':  # <Sz(r,t) Sz(center,0)>
        quench_operator = sz
        measure_operator = sz
    elif kp.quench == '+':  # <S+(r,t) S-(center,0)>
        quench_operator = sm
        measure_operator = sp
    else:
        raise ValueError(f'Illegal quench code {kp.quench}')

    logger.log(f'Quench: Applying quench operator to center site')
    quenched = SingleSiteOperator(quench_operator, kp.L // 2, kp.L // 2).apply_to_peps(gs)

    # Time evolution

    x_snapshot_data = onp.zeros([kp.n_steps + 1, kp.L, kp.L])
    y_snapshot_data = onp.zeros([kp.n_steps + 1, kp.L, kp.L])
    z_snapshot_data = onp.zeros([kp.n_steps + 1, kp.L, kp.L])
    correlator_data = onp.zeros([kp.n_steps + 1, kp.L, kp.L], dtype=onp.complex)
    t_data = onp.zeros([kp.n_steps + 1])

    state = quenched
    opt_opts['dtype'] = np.complex128
    opt_opts['max_grad_evals_ls'] = 100
    for n in range(kp.n_steps):
        logger.log('Computing Observables')

        t = n * kp.dt
        x_snapshot_data[n, :, :] = x_snapshot(state, cont_opts)
        y_snapshot_data[n, :, :] = y_snapshot(state, cont_opts)
        z_snapshot_data[n, :, :] = z_snapshot(state, cont_opts)
        correlator_data[n, :, :] = correlator_timeslice(gs, state, measure_operator, gs_energy, t, **cont_opts)
        t_data[n] = t

        logger.log(f'Evolving to t={(n + 1) * kp.dt}')
        state = evolver.evolve(state, contraction_options=cont_opts, optimisation_options=opt_opts,
                               random_dev=None, initial=kp.initial)

        # save results (will be overwritten), (in case process dies before it finishes)
        results = dict(kwargs=kp.kwargs(),
                       quench=kp.quench,
                       x_snapshot=x_snapshot_data,
                       y_snapshot=y_snapshot_data,
                       z_snapshot=z_snapshot_data,
                       correlator=correlator_data,
                       t=t_data,
                       state_tensors=state.get_tensors())
        with open(outfile, 'wb') as f:
            pickle.dump(results, f)

        if kp.save_all_peps:
            results = dict(kwargs=kp.kwargs(), t=t, state_tensors=state.get_tensors())
            with open(folder.joinpath(file_str + f'state_t_{t}.pkl'), 'wb') as f:
                pickle.dump(results, f)

    logger.log('Computing Observables')
    t = kp.n_steps * kp.dt
    x_snapshot_data[kp.n_steps, :, :] = x_snapshot(state, cont_opts)
    y_snapshot_data[kp.n_steps, :, :] = y_snapshot(state, cont_opts)
    z_snapshot_data[kp.n_steps, :, :] = z_snapshot(state, cont_opts)
    correlator_data[kp.n_steps, :, :] = correlator_timeslice(gs, state, measure_operator, gs_energy, t, **cont_opts)
    t_data[kp.n_steps] = t

    # save results
    logger.log(f'saving results to {outfile}')
    results = dict(kwargs=kp.kwargs(),
                   quench=kp.quench,
                   x_snapshot=x_snapshot_data,
                   y_snapshot=y_snapshot_data,
                   z_snapshot=z_snapshot_data,
                   correlator=correlator_data,
                   t=t_data,
                   state_tensors=state.get_tensors())
    with open(outfile, 'wb') as f:
        pickle.dump(results, f)

    if kp.save_all_peps:
        results = dict(kwargs=kp.kwargs(), t=t, state_tensors=state.get_tensors())
        with open(folder.joinpath(file_str + f'state_t_{t}.pkl'), 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    folder = '~/Desktop/tmp_runs/C_peps_quench/' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    run_simulation(L=3, max_iter=2, quench='X', chi=2, g=5., dt=0.01, n_steps=5, folder=folder)
