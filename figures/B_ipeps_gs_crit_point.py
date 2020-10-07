import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pickle
import os
import numpy as np
from scipy.optimize import curve_fit
from figures.styles import colors_per_chi

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 15})

use_dummy = False

markers_per_chi = {2: 's', 3: 'o', 4: '^', 5: 'v'}
gc_mc_kwargs = dict(color='purple', lw=2., ls='--')
gc_dmrg_kwargs = dict(color='blue', lw=2., ls='-.')

root = Path(__file__).parent.joinpath('raw_data/B_ipeps_gs')
folders = [root.joinpath(f) for f in ['chi_2/D_opt_20/2020_07_08_17_33_21',
                                      'chi_2/D_opt_50/2020_07_08_17_32_16',
                                      'chi_3/D_opt_20/2020_07_08_17_33_28',
                                      'chi_3/D_opt_50/2020_07_08_17_32_56']]


fontsize = 18


def fit_fun(g, gc, beta, m0):
    return np.heaviside(gc - g, 0) * m0 * (np.abs(gc - g) ** beta)


def do_fit(g_data, m_data, chi):
    gc_guess = {2: 3.09, 3: 3.05}[chi]
    mask = np.logical_and(3. < g_data, g_data < 3.1)
    if chi == 3:
        mask[np.logical_and(3.046 < g_data, g_data < 3.0595)] = False
    (gc, beta, m0), pcov = curve_fit(fit_fun, g_data[mask], m_data[mask], p0=(gc_guess, 0.6, 3.))
    # gc_err, beta_err, m0_err = np.sqrt(np.diag(pcov))
    fitted = lambda g: fit_fun(g, gc, beta, m0)
    return gc, beta, fitted


def load_raw():
    data = {}  # data[chi] = (g_data, energy_data, mag_data, D_obs)

    for folder in folders:
        for file in os.listdir(folder):
            if not str(file).endswith('.pkl'):
                continue

            with open(folder.joinpath(file), 'rb') as f:
                results = pickle.load(f)
            g = results['g']
            D_obs_list = results['D_obs_list']
            en_list = results['en_list']
            mag_list = results['mag_list']
            chi = results['kwargs']['chi']
            if chi not in data:
                data[chi] = [[], [], [], None]

            # find highest D_obs
            D_obs = max(D_obs_list)
            n = D_obs_list.index(D_obs)
            data[chi][0].append(g)
            data[chi][1].append(en_list[n])
            data[chi][2].append(mag_list[n])

            if data[chi][3] and data[chi][3] != D_obs:
                print('[WARNING]: inconsistent D_obs')

            data[chi][3] = D_obs

    for chi in data:
        g_data = np.array(data[chi][0])
        idcs = np.argsort(g_data)
        data[chi][0] = g_data[idcs]
        data[chi][1] = np.array(data[chi][1])[idcs]
        data[chi][2] = np.abs(np.array(data[chi][2])[idcs])

    return data


def process_data():
    raw_data = load_raw()  # data[chi] = (g_data, energy_data, mag_data, D_obs)
    data = {}
    for chi in raw_data:
        data[chi] = raw_data[chi]
    data['gc_cluster_mc'] = 3.04438
    data['gc_dmrg'] = 3.046

    for chi in [2, 3]:
        g_data, energy_data, mag_data, D_obs = data[chi]
        gc, beta, fitted = do_fit(g_data, mag_data, chi)
        data[chi, 'gc'] = gc
        data[chi, 'beta'] = beta
        data[chi, 'fitted'] = fitted

    return data


def dummy_data():
    raise NotImplementedError


def save_data(data):
    data = data.copy()
    # cant pickle the lambda `fitted`
    for chi in [2, 3]:
        fitted = data.pop((chi, 'fitted'))
        fit_g_list = np.linspace(3., data[chi, 'gc'], 501)
        fit_m_list = fitted(fit_g_list)
        data[chi, 'fit_g_list'] = fit_g_list
        data[chi, 'fit_m_list'] = fit_m_list

    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot(data):
    chis = [chi for chi in data if type(chi) is int]

    fig, ax = plt.subplots()

    # comparative gc values:
    ax.axvline(data['gc_cluster_mc'], label='$g_c$, cluster MC', **gc_mc_kwargs)
    ax.axvline(data['gc_dmrg'], label='$g_c$, DMRG', **gc_dmrg_kwargs)
    ax.axhline(0, color='gray', lw=.5)

    for chi in chis:
        g_data, en_data, mag_data, D_obs = data[chi]
        ax.plot(g_data, mag_data, markers_per_chi[chi], label=f'$\\chi={chi}$', color=colors_per_chi[chi])
        fit_gs = np.linspace(3., data[chi, 'gc'], 501)
        kwargs = dict(label='Fit') if chi == chis[-1] else {}
        ax.plot(fit_gs, data[chi, 'fitted'](fit_gs), color='red', lw=2, **kwargs)

    ax.set_xlim([3.0, 3.1])
    ax.set_ylim([-.05, .45])
    ax.legend(fontsize=14)
    ax.set_ylabel('$m$', fontsize=fontsize)
    ax.set_xlabel('$g$', fontsize=fontsize)

    # gc inset
    msg = r'Critical point $g_c$\\[2ex]\makebox[2.5cm]{cluster MC:} $3.0444$\\[1ex]\makebox[2.5cm]{DMRG:} $3.0460$\\[1ex]\makebox[2.5cm]{$\chi=2$:} $' \
          + f'{data[2, "gc"]:.4f}' + r'$\\[1ex]\makebox[2.5cm]{$\chi=3$:} $' + f'{data[3, "gc"]:.4f}' + r'$'

    ax.text(0.05, 0.05, msg, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', fill=True, facecolor='white'))
    fig.tight_layout()

    outfile = Path(__file__).parent.joinpath(f'plots/{Path(__file__).stem}.pdf')
    fig.savefig(outfile)
    plt.show()


if __name__ == '__main__':
    _data = process_data()
    save_data(_data)
    plot(_data)
