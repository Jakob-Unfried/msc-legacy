import numpy as np
import scipy.sparse
import scipy.sparse.linalg

Sx = np.array([[0., 1.],
               [1., 0.]])
Sy = np.array([[0., -1j],
               [1j, 0.]])
Sz = np.array([[1., 0.],
               [0., -1.]])


def gen_pair(row, V, PBC=False):
    """
    assume row is an in order array generate a cyclic pairs
    in the row array given with interaction strength V.
    For example: row = [1, 2, 3, 5]
    will gives [(1, 2, V), (2, 3, V), (3, 5, V), (5, 1, V)]
    """
    if PBC:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row))]
    else:
        return [(row[i], row[(i + 1) % len(row)], V) for i in range(len(row) - 1)]


def build_H_one_body(sites, L, H=None, sx=False, sy=False, sz=False, verbose=False):
    if H is None:
        H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    else:
        pass

    for i, V in sites:
        if verbose:
            print("building", i)

        if sx:
            hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hx

        if sy:
            hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hy

        if sz:
            hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - i)))
            H = H + V * hz

    H = scipy.sparse.csr_matrix(H)
    return H


def build_H_two_body(pairs, L, H=None, sxsx=False,
                     sysy=False, szsz=False, verbose=False):
    if H is None:
        H = scipy.sparse.csr_matrix((2 ** L, 2 ** L))
    else:
        pass

    for i, j, V in pairs:
        if i > j:
            i, j = j, i

        if verbose:
            print("building", i, j)

        if sxsx:
            hx = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (j - i - 1)))
            hx = scipy.sparse.kron(hx, Sx)
            hx = scipy.sparse.kron(hx, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hx

        if sysy:
            hy = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (j - i - 1)))
            hy = scipy.sparse.kron(hy, Sy)
            hy = scipy.sparse.kron(hy, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hy

        if szsz:
            hz = scipy.sparse.kron(scipy.sparse.eye(2 ** (i - 1)), Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (j - i - 1)))
            hz = scipy.sparse.kron(hz, Sz)
            hz = scipy.sparse.kron(hz, scipy.sparse.eye(2 ** (L - j)))
            H = H + V * hz

    H = scipy.sparse.csr_matrix(H)
    return H


def gen_H_2d_Ising(Lx, Ly, J=1., g=0., PBC=False):
    """
    H = -J sigma_z sigma_z - h sigma_x
    """
    lattice = np.zeros((Lx, Ly), dtype=int)
    for i in range(Lx):
        for j in range(Ly):
            lattice[i, j] = int(j * Lx + (i + 1))

    #     for i in range(Lx):
    #         for j in range(0, Ly, 2):
    #             lattice[i, j] = int(j * Lx + (i+1))
    #             lattice[-(i+1), j+1] = int( (j+1) * Lx + (i+1))

    # print(lattice)
    pairs = []
    # NN interaction : J
    for i in range(Lx):
        # print(lattice[i, :])
        pairs = pairs + gen_pair(lattice[i, :], -J, PBC=PBC)

    for j in range(Ly):
        # print(lattice[:, j])
        pairs = pairs + gen_pair(lattice[:, j], -J, PBC=PBC)

    # print('all pairs', pairs)
    # single site operator: h
    L = Lx * Ly
    sx_sites = [(i, -g) for i in range(1, L + 1)]

    H = build_H_two_body(pairs, Lx * Ly, sxsx=True, sysy=False, szsz=False)
    H = build_H_one_body(sx_sites, L, H=H, sx=False, sy=False, sz=True)
    return H
