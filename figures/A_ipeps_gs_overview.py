import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import pickle
import os
import numpy as np
from figures.styles import colors_per_chi

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 14})

root = Path(__file__).parent.joinpath('raw_data/B_ipeps_gs')
folders = [root.joinpath(f) for f in ['chi_2/D_opt_20/2020_07_08_17_33_21',
                                      'chi_2/D_opt_50/2020_07_08_17_32_16',
                                      'chi_3/D_opt_20/2020_07_08_17_33_28',
                                      'chi_3/D_opt_50/2020_07_08_17_32_56']]

fontsize = 18

def load_raw():
    data = {}  # data[chi] = (g_data, energy_data, mag_data, D_obt)

    for folder in folders:
        for file in os.listdir(folder):
            if not str(file).endswith('.pkl'):
                continue

            with open(folder.joinpath(file), 'rb') as f:
                results = pickle.load(f)
            g = results['g']
            D_obs_list = results['D_obs_list']
            D_opt = results['kwargs']['D_opt']
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

            data[chi][3] = D_opt

    for chi in data:
        g_data = np.array(data[chi][0])
        idcs = np.argsort(g_data)
        data[chi][0] = g_data[idcs]
        data[chi][1] = np.array(data[chi][1])[idcs]
        data[chi][2] = np.abs(np.array(data[chi][2])[idcs])

    return data


def process_data():
    raw_data = load_raw()
    data = {}
    for chi in raw_data:
        data[chi] = raw_data[chi]
    data['gc_cluster_mc'] = 3.04438
    data['gc_dmrg'] = 3.046
    return data


def save_data(data):
    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot(data):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    chis = [chi for chi in data if type(chi) is int]

    # asymptotic energies
    ax1.axhline(-2., ls='--', color='gray')
    gs = (data[2][0][0], data[2][0][-1])
    ax1.plot(gs, [-g for g in gs], '--', color='gray')

    for chi in reversed(chis):
        g_data, en_data, mag_data, D_opt = data[chi]
        ax1.plot(g_data, en_data, lw=2, color=colors_per_chi[chi],
                 label='$\\chi = {:d}$'.format(chi))
                 # label='$\\chi = {:d}$, $D_{{opt}} = {:d}$'.format(chi, D_opt))
        ax2.plot(g_data, mag_data, lw=2, color=colors_per_chi[chi],
                 label='$\\chi = {:d}$'.format(chi))
                 # label='$\\chi = {:d}$, $D_{{opt}} = {:d}$'.format(chi, D_opt))

    # legend in reverse order
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1])
    ax1.set_ylabel('$E / L^2$', fontsize=fontsize)
    ax1.set_xlabel('$g$', fontsize=fontsize)
    ax1.set_ylim((-5.25, -1.65))

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::-1], labels[::-1])
    ax2.set_ylabel('$m$', fontsize=fontsize)
    ax2.set_xlabel('$g$', fontsize=fontsize)

    outfile = Path(__file__).parent.joinpath(f'plots/{Path(__file__).stem}.pdf')
    fig.savefig(outfile)
    plt.show()


if __name__ == '__main__':
    _data = process_data()
    save_data(_data)
    plot(_data)
