import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pickle
import os
import numpy as onp
from scipy.optimize import curve_fit
from figures.linear_pred import linear_extrap

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 15})

use_dummy = False
fontsize = 18

root = Path(__file__).parent.joinpath(f'raw_data/C_peps_quench/')
folder = root.joinpath('2020_09_23_08_39_00')

g = 5.
L = 5
chi = 2


def load_raw():
    dt = 0.05

    file = folder.joinpath(f'L_{L}_g_{g}_chi_{chi}_dt_{dt}_quench_X.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)
    kwargs = results['kwargs']
    assert g == kwargs['g']
    assert L == kwargs['L']
    assert chi == kwargs['chi']
    assert dt == kwargs['dt']

    z_snap, xx_corr, t = [results[key] for key in ['z_snapshot', 'correlator', 't']]

    # purge zeros
    mask = (t > 0)
    mask[0] = True
    z_snap, xx_corr, t = z_snap[mask], xx_corr[mask], t[mask]

    return xx_corr, dt


def load_ed():
    dt = 0.05
    file = Path(f'~/LRZ_Sync+Share/Physik/msc_thesis/msc-data/I1/L_{L}_g_{g}_X').expanduser() \
        .joinpath(f'L_{L}_g_{g}_dt_{dt}_quench_X.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)
    kwargs = results.pop('kwargs')
    assert (kwargs['L'], kwargs['g'], kwargs['dt']) == (L, g, dt)
    z_snap, xx_corr, t = [results[key] for key in ['z_snapshot', 'xx_correlator', 'times']]

    # purge zeros
    mask = (t > 0)
    mask[0] = True
    z_snap, xx_corr, t = z_snap[mask], xx_corr[mask], t[mask]

    return xx_corr, dt


def get_dsf(C_rt, dt, omega_max=20, n_omega=100, decay=0.5, use_lin_extr=True):
    """
    C_rt : correlation function C(r,t) = <O(t)O(0))>, C_rt[t,x,y]
    dt : time step, C_rt[n,:,:] is t=n*dt
    omega_max : maximal omega
    d_omega : step for omega

    out : dynamical structure facts S(k,w)
    """
    n_steps, L, Ly = C_rt.shape
    assert Ly == L
    N = L ** 2

    if use_lin_extr:
        C_rt = linear_extrap(onp.reshape(C_rt, (n_steps, N)))
        n_steps, _ = C_rt.shape
        C_rt = onp.reshape(C_rt, (n_steps, L, L))

    w = onp.linspace(0, omega_max, num=n_omega + 1, endpoint=True)  # [w]
    t = onp.arange(n_steps) * dt  # [t]

    exp_iwt = onp.exp(1.j * w[:, None] * t[None, :])  # [w,t]
    eta = - onp.log(decay) / ((n_steps * dt) ** 2)
    gaussian = onp.exp(- eta * (t ** 2))  # [t]

    # exp(-eta * t^2)
    sigma_t = onp.sqrt(1 / (2. * eta))
    sigma_w = 1. / sigma_t
    print(f'T={t[-1]}   ,   sigma_t={sigma_t}   ,   sigma_w={sigma_w}')

    # S(r,w) = 1/2π ∫dt C(r,t) = 1/2π 0∫∞ dt 2 Re[C(r,t)] ~ 1/N Σ_n 2 Re[C(r,t_n)] * gaussian(t_n)
    S_rw = 1. / n_steps * onp.einsum('wt, t, txy->wxy', exp_iwt, gaussian, 2 * onp.real(C_rt))  # [w,x,y]
    x = onp.array(range(-L // 2 + 1, L // 2 + 1))
    y = onp.array(range(-L // 2 + 1, L // 2 + 1))
    kx = 2 * onp.pi / L * x
    ky = 2 * onp.pi / L * x

    # S(k,w) = Σ_R exp[-i k*R] S(r,w)
    S_kw = onp.einsum('kx, qy, wxy->wkq',
                      onp.exp(-1.j * kx[:, None] * x[None, :]),
                      onp.exp(-1.j * ky[:, None] * y[None, :]),
                      S_rw)  # [w,kx,ky]

    return S_kw


def process_data():
    which_part = onp.real
    purge_negative = False
    t_max = None

    omega_max = 20
    n_omega = 400
    decay = .1

    dw = omega_max / n_omega

    xx_corr_ed, dt_ed = load_ed()
    xx_corr_my, dt_my = load_raw()

    # use same time cutoff
    xx_corr_ed = xx_corr_ed[:xx_corr_my.shape[0]]
    print(f'T_max before linPred: {len(xx_corr_my) * dt_my}')

    if t_max:
        xx_corr_ed = xx_corr_ed[:int(t_max//dt_ed)]
        xx_corr_my = xx_corr_ed[:int(t_max//dt_my)]

    print('ED')
    dsf_ed = which_part(get_dsf(xx_corr_ed, dt_ed, omega_max=omega_max, n_omega=n_omega, decay=decay))
    print('ADopt')
    dsf_my = which_part(get_dsf(xx_corr_my, dt_my, omega_max=omega_max, n_omega=n_omega, decay=decay))

    if purge_negative:
        dsf_ed[dsf_ed < 0] = 0
        dsf_my[dsf_my < 0] = 0

    # reshape to path through BZ
    dsf_ed = onp.array([dsf_ed[:, 2, 2], dsf_ed[:, 3, 2], dsf_ed[:, 4, 2], dsf_ed[:, 4, 3], dsf_ed[:, 4, 4],
                        dsf_ed[:, 3, 3], dsf_ed[:, 2, 2]]).T
    dsf_my = onp.array([dsf_my[:, 2, 2], dsf_my[:, 3, 2], dsf_my[:, 4, 2], dsf_my[:, 4, 3], dsf_my[:, 4, 4],
                        dsf_my[:, 3, 3], dsf_my[:, 2, 2]]).T
    return which_part(dsf_ed), which_part(dsf_my), dw


def dummy_data():
    raise NotImplementedError





def save_data(data):
    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot(data):
    dsf_ed, dsf_my, dw = data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey='all')

    cmap = 'Reds'  # viridis, seismic, Reds
    neg_vals = False

    if neg_vals:
        _min = min(onp.min(dsf_my), onp.min(dsf_ed))
        _max = max(onp.max(dsf_my), onp.max(dsf_ed))
        vabs = max(abs(_min), abs(_max))
        vabs = onp.ceil(vabs * 10.) / 10.
        vmin, vmax = -vabs, vabs

    else:
        vmin = 0.
        vmax = max(onp.max(dsf_my), onp.max(dsf_ed))
        vmax = onp.ceil(vmax * 10.) / 10.

    for dsf, ax in zip([dsf_my, dsf_ed], [ax1, ax2]):
        pcol = ax.pcolormesh(dsf, linewidth=0., rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap)
        pcol.set_edgecolor('face')
        ax.set_yticklabels([tick * dw for tick in ax.get_yticks()])
        ax.set_xticks([x + 0.5 for x in range(7)])
        # ax.set_xticklabels(['$(0,0)$', '', r'$(\pi, 0)$', '', r'$(\pi, \pi)$', '', '(0, 0)'])
        # ax.set_xlabel('$(k_x, k_y)$')
        ax.set_xticklabels([r'$\Gamma$', '', r'$X$', '', r'$M$', '', r'$\Gamma$'])
        ax.set_xlabel('$\mathbf{k}$', fontsize=fontsize)
        fig.colorbar(pcol, ax=ax)

    ax1.set_ylabel('$\omega$', fontsize=fontsize)

    for t, ax in zip(['ADopt', 'ED'], [ax1, ax2]):
        ax.text(0.05, 0.95, t, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', fill=True, facecolor='white'))

    fig.tight_layout()

    outfile = Path(__file__).parent.joinpath(f'out/{Path(__file__).stem}.pdf')
    fig.savefig(outfile, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    _data = process_data()
    save_data(_data)
    plot(_data)
