import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as onp
from figures.styles import colors_per_chi
import pickle

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 15})

use_dummy = False
g = 3.5
fontsize = 18


root = Path(__file__).parent.joinpath(f'raw_data/A_peps_gs/g_{g}')
assert root.exists()


def load_raw():
    my_data = {}  # my_data[L, chi] = energy
    isopeps_data = {}  # isopeps_data[L, chi] = energies

    dts = [9.097959895689501364e-01,
           5.518191617571635010e-01,
           3.346952402226447409e-01,
           2.030029248549190535e-01,
           1.231274979358482069e-01,
           7.468060255179591322e-02,
           4.529607513347774783e-02,
           2.747345833310126800e-02,
           1.666349480736345826e-02,
           1.010692049862819963e-02,
           6.130157157696099895e-03]

    # load my data

    for L in [4, 5, 6, 7, 8]:
        folder = root.joinpath(f'L_{L}')
        for chi in [2, 3, 4]:
            log_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.log')
            opt_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.opt.log')
            pkl_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.pkl')

            if not opt_file.exists():
                continue

            energies, grad_norms = _parse_log(opt_file)

            my_data[L, chi] = energies[-1]

    # load isoPEPS data
    for L in [6, 8, 10]:
        for chi in [2, 4, 6]:
            J = 1.
            eta = 6 * chi
            folder = '/Users/jakobunfried/LRZ_Sync+Share/Physik/msc_thesis/analysis/analysis/ising_isopeps'
            file = f'{folder}/rnv_ising_moses_J_{J:.02f}_g_{g:.02f}_L_{L:.0f}_eta_{eta:.0f}_chi_{chi:.0f}'
            isopeps_data[L, chi] = onp.loadtxt(file)

    return my_data, isopeps_data, dts


def process_data():
    my_data, isopeps_data, dts = load_raw()

    # exact energies (L<=5 : ED, L > 5: DMRG
    e0 = {3: -32.40218609609526,
          4: -57.82436977640373,
          5: -90.56087934621652,
          6: -130.6117109064,
          7: -177.9768577337589,
          8: -232.656314635,
          10: -363.95737940}

    diffs_my_data = {(L, chi): (my_data[L, chi] - e0[L]) / (L ** 2) for (L, chi) in my_data}
    diffs_dts_isopeps = {}
    # {(L, chi): isopeps_data[L, chi] - e0[L] / (L ** 2) for (L, chi) in isopeps_data}
    for L, chi in isopeps_data:
        dts, ens = isopeps_data[L, chi]
        diffs_dts_isopeps[L, chi] = dts, ens - e0[L] / (L ** 2)

    return diffs_my_data, diffs_dts_isopeps, dts


def dummy_data():
    raise NotImplementedError





def save_data(data):
    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot(data):
    xlims = (1e-2, 1.)
    ylims = (5e-9, 1e-1)

    diffs_my_data, diffs_dts_isopeps, dts = data

    fig, axs = plt.subplots(2, 3, figsize=(9, 7))

    handles = []
    labels = []

    for ax in axs.flat:
        ax.set_xlabel('$dt$')

    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set_ylabel('$\\Delta E / L^2$')

    for L, ax in zip([4, 5, 6, 7, 8], axs.flat):
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # plot TEBD2
        for chi in [2, 4, 6]:
            if (L, chi) not in diffs_dts_isopeps:
                continue
            _dts, diffs = diffs_dts_isopeps[L, chi]
            style = {2: 'o', 4: 's', 6: '^'}[chi] + '-'
            ax.loglog(_dts, diffs, style, color=colors_per_chi[chi],
                      label=f'TEBD\\textsuperscript{2} , $\\chi = {chi}$')

        # get TEBD handles, labels
        if L == 6:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # plot my data
        for chi in [2, 3, 4]:
            if (L, chi) not in diffs_my_data:
                continue
            style = {2: '-', 3: '--', 4: '-.'}[chi]
            ax.loglog(dts, [diffs_my_data[L, chi]] * len(dts), color=colors_per_chi[chi], ls=style,
                      label=f'ADopt, $\\chi={chi}$', lw=2.)
        if L == 4:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # L textbox
        ax.text(0.95, 0.05, f'$L={L}$', transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', fill=True, facecolor='white'))

    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.7, 0.1, 0.3, 0.4),
               bbox_transform=fig.transFigure)


    # remove axs[1, 2]
    axs[1, 2].clear()
    axs[1, 2].set_axis_off()

    fig.tight_layout()

    outfile = Path(__file__).parent.joinpath(f'plots/{Path(__file__).stem}.pdf')
    fig.savefig(outfile)
    plt.show()


def _parse_log(file):
    with open(file, 'rb') as f:
        lines = [str(x).replace('\'', '').replace('\\n', '').replace('strong wolfe', 'strong_wolfe')[1:] for x in
                 f.readlines()]

    assert len(lines) >= 3

    # find line of initial evaluation
    n_initial = [n for (n, line) in enumerate(lines) if line.startswith('    0')]
    assert len(n_initial) == 1
    n_initial = n_initial[0]

    energies = []
    grad_norms = []
    nums = []

    for line in lines[n_initial:]:
        # skip non-proper lines
        if line.startswith('[WARNING]') or line.startswith('Lack') or line.startswith('Skipping'):
            continue

        # split into columns
        cols = [s for s in line.split(' ') if (len(s) > 0 and s != '\\n')]
        assert len(cols) == 8
        n, en, en_change, grad_norm, step_dist, alpha, num_f, num_df = cols
        energies.append(float(en))
        grad_norms.append(float(grad_norm))
        nums.append(int(n))

    assert nums == list(range(nums[-1] + 1))
    return energies, grad_norms


if __name__ == '__main__':
    _data = process_data()
    save_data(_data)
    plot(_data)
