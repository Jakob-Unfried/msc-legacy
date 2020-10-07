import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as onp
from figures.styles import colors_per_chi, style_per_chi, markers_per_reason
import pickle

width = 8
aspect_ratio = 4./3.
fontsize = 18

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams.update({'font.size': 13})

use_dummy = False
g = 3.5

root = Path(__file__).parent.joinpath(f'raw_data/A_peps_gs/g_{g}')
assert root.exists()


def load_raw():
    my_data = {}  # my_data[L, chi] = energies, grad_norms

    for L in [4, 5, 6, 7, 8]:
        folder = root.joinpath(f'L_{L}')
        for chi in [2, 3, 4]:
            log_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.log')
            opt_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.opt.log')
            pkl_file = folder.joinpath(f'L_{L}_chi_{chi}_g_{g}_brute.pkl')

            if not opt_file.exists():
                continue

            energies, grad_norms = _parse_log(opt_file)

            my_data[L, chi] = onp.array(energies), onp.array(grad_norms)

    return my_data


def process_data():
    raw_data = load_raw()

    # exact energies (L<=5 : ED, L > 5: DMRG
    e0 = {3: -32.40218609609526,
          4: -57.82436977640373,
          5: -90.56087934621652,
          6: -130.6117109064,
          7: -177.9768577337589,
          8: -232.656314635,
          10: -363.95737940}

    data = {}  # data[L][chi] = ΔE/L^2 , |∂E|/L^2

    # 'converged', 'progress', 'ressources', None
    termination_reasons = {
        4: {2: 'progress', 3: None, 4: None},
        5: {2: 'progress', 3: None},
        6: {2: 'progress'},
        7: {2: 'ressources'}
    }

    for L in [4, 5, 6, 7, 8]:
        _dat = {}
        for chi in [2, 3, 4]:
            if (L, chi) not in raw_data:
                continue

            energies, grad_norms = raw_data[L, chi]
            _dat[chi] = (energies - e0[L]) / (L ** 2), grad_norms, termination_reasons[L][chi]
        data[L] = _dat

    return data


def dummy_data():
    raise NotImplementedError





def save_data(data):
    file = Path(__file__).parent.joinpath(f'plotted_data/{Path(__file__).stem}.pkl')
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def plot(data):
    fig, axs = plt.subplots(2, 2, figsize=(width, width / aspect_ratio))
    yticks = [1., 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    xlims = (0, 5000)
    ylims = (1e-8, 1)

    for ax, L in zip(axs.flat, [4, 5, 6, 7]):
        ax.set_xlabel('Iteration Number', fontsize=15)
        ax.set_ylabel('$\\Delta E / L^2$', fontsize=15)
        for chi in [2, 3, 4]:
            if chi not in data[L]:
                continue
            delta_e, _, reason = data[L][chi]
            ax.semilogy(delta_e, ls=style_per_chi[chi], color=colors_per_chi[chi],
                        lw=1.5, label=f'$\\chi = {chi}$')
            if reason:
                ax.semilogy(len(delta_e), delta_e[-1], marker=markers_per_reason[reason], color=colors_per_chi[chi])

        ax.set_yticks(yticks)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.text(0.05, 0.95, f'$L={L}$', transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', fill=False))

    axs[0, 0].legend()
    lines = [Line2D([0, 1], [0, 1], ls='None', marker=markers_per_reason[reason], color='black')
             for reason in ['converged', 'progress', 'ressources']]
    labels = ['Converged', 'Lack of Progress', 'Resources exceeded']
    axs[0, 1].legend(lines, labels, title='Termination reasons', fontsize=12)
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
