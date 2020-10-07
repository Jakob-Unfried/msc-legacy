import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pickle
import os
import numpy as np
from scipy.optimize import curve_fit
from figures.styles import colors_per_chi, style_per_chi

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

    return z_snap, xx_corr, t


def load_ed():
    dt = 0.05
    file = Path(f'~/LRZ_Sync+Share/Physik/msc_thesis/msc-data/I1/L_{L}_g_{g}_X').expanduser()\
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

    return z_snap, xx_corr, t


def process_data():
    raw_data = load_raw()
    ed_data = load_ed()
    return raw_data, ed_data


def dummy_data():
    raise NotImplementedError





def save_data(data):
    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot_sz(data):
    my_data, ed_data = data

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
    fig.subplots_adjust(hspace=0)

    xlims = (0, 3.5)
    ylims = (-1.1, 1.1)
    xlabel = '$t$'
    ylabel = r'$\left<S_i^z\right>_t$'
    kwargs = {'my': dict(label='ADopt', marker='o', markersize=2, ls='None', color='black'),
              'ed': dict(label='ED', color='orange', lw=3)}
    ax_data = [(ax1, 2, 2), (ax2, 2, 3), (ax3, 1, 3), (ax4, 2, 4), (ax5, 0, 4)]

    ax5.set_xlabel(xlabel, fontsize=fontsize)
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_ylabel(ylabel, fontsize=fontsize)

    for ax, x, y in ax_data:
        z_snap, xx_corr, t = ed_data
        ax.plot(t, z_snap[:, x, y], **kwargs['ed'])
        z_snap, xx_corr, t = my_data
        ax.plot(t, z_snap[:, x, y], **kwargs['my'])

    # FINALLY
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    outfile = Path(__file__).parent.joinpath(f'plots/{Path(__file__).stem}_sz.pdf')
    fig.savefig(outfile, bbox_inches='tight')
    plt.show()


def plot_corr(data):
    my_data, ed_data = data

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
    fig.subplots_adjust(hspace=0)

    xlims = (0, 3.5)
    ylims = (-1.1, 1.1)
    xlabel = '$t$'
    ylabel = r'$\left<S_i^x(t) S_c^x\right>$'
    kwargs = {'my': {'re': dict(marker='o', markersize=2, ls='None', color='black', label='ADopt, Re'),
                     'im': dict(marker='s', markersize=3, ls='None', color='gray', label='ADopt, Im')},
              'ed': {'re': dict(color='orange', lw=3, label='ED, Re'),
                     'im': dict(color='cyan', lw=2, label='ED, Im')}}
    ax_data = [(ax1, 2, 2), (ax2, 2, 3), (ax3, 1, 3), (ax4, 2, 4), (ax5, 0, 4)]

    ax5.set_xlabel(xlabel, fontsize=fontsize)
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_ylabel(ylabel, fontsize=fontsize)

    for ax, x, y in ax_data:
        z_snap, xx_corr, t = ed_data
        ax.plot(t, np.real(xx_corr[:, x, y]), **kwargs['ed']['re'])
        ax.plot(t, np.imag(xx_corr[:, x, y]), **kwargs['ed']['im'])
        z_snap, xx_corr, t = my_data
        ax.plot(t, np.real(xx_corr[:, x, y]), **kwargs['my']['re'])
        ax.plot(t, np.imag(xx_corr[:, x, y]), **kwargs['my']['im'])

    # FINALLY
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    outfile = Path(__file__).parent.joinpath(f'plots/{Path(__file__).stem}_corr.pdf')
    fig.savefig(outfile, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    _data = process_data()
    save_data(_data)
    plot_sz(_data)
    plot_corr(_data)
