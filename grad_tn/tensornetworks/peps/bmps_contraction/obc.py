from typing import List

from jax import numpy as np, tree_map, tree_multimap
from jax.numpy import ndarray as array
from autodiff.svd_safe import svd, svd_truncated


# FIXME needs double-checking and testing!

# TODO variational bMPS?


def two_layers(bra_layer, ket_layer, chi_bmps=None, cutoff=0.) -> complex:
    lx, ly = len(bra_layer), len(bra_layer[0])
    x_meet = lx // 2

    if chi_bmps is None:
        # default is double the (vertical = u & d) bond-dimension of the layer.
        # assume uniformity and only consider the u leg of the middle tensor
        chi_bmps = 2 * bra_layer[lx // 2][ly // 2].shape[1] * ket_layer[lx // 2][ly // 2].shape[1]

    # create left and right trivial boundary MPSs
    left_mps = trivial_boundary_mps2(ly, chi_bmps, cutoff)
    right_mps = trivial_boundary_mps2(ly, chi_bmps, cutoff)

    # add trivial bond on left and right boundary
    bra_left = add_trivial_legs_left(bra_layer[0])
    ket_left = add_trivial_legs_left(ket_layer[0])
    bra_right = add_trivial_legs_right(bra_layer[-1])
    ket_right = add_trivial_legs_right(ket_layer[-1])

    left_mps.absorb_col(bra_left, ket_left, True)
    for n in range(1, x_meet):
        left_mps.absorb_col(bra_layer[n], ket_layer[n], True)

    right_mps.absorb_col(bra_right, ket_right, False)
    for n in range(lx - 2, x_meet - 1, -1):
        right_mps.absorb_col(bra_layer[n], ket_layer[n], False)

    res = left_mps.contract_with(right_mps)
    return res


def three_layers(bra_layer, op_layer, ket_layer, chi_bmps=None, cutoff=0.) -> complex:
    lx, ly = len(bra_layer), len(bra_layer[0])
    x_meet = lx // 2

    if chi_bmps is None:
        # default is double the bond-dimension of the layer.
        # assume uniformity and only consider the u leg of the middle tensor
        chi_bmps = 2 * bra_layer[lx // 2][ly // 2].shape[1] * op_layer[lx // 2][ly // 2].shape[2] \
                   * ket_layer[lx // 2][ly // 2].shape[1]

    # create left and right trivial boundary MPSs
    left_mps = trivial_boundary_mps3(ly, chi_bmps, cutoff)
    right_mps = trivial_boundary_mps3(ly, chi_bmps, cutoff)

    # add trivial bond on left and right boundary to match bulk leg-structure
    bra_left = add_trivial_legs_left(bra_layer[0])
    op_left = add_trivial_legs_left(op_layer[0], num_phys_legs=2)
    ket_left = add_trivial_legs_left(ket_layer[0])
    bra_right = add_trivial_legs_right(bra_layer[-1])
    op_right = add_trivial_legs_right(op_layer[-1], num_phys_legs=2)
    ket_right = add_trivial_legs_right(ket_layer[-1])

    left_mps.absorb_col(bra_left, op_left, ket_left, True)
    for n in range(1, x_meet):
        left_mps.absorb_col(bra_layer[n], op_layer[n], ket_layer[n], True)

    right_mps.absorb_col(bra_right, op_right, ket_right, False)
    for n in range(lx - 2, x_meet - 1, -1):
        right_mps.absorb_col(bra_layer[n], op_layer[n], ket_layer[n], False)

    return left_mps.contract_with(right_mps)


class BoundaryMPS2:
    def __init__(self, tensors, chi_bmps, cutoff):
        """

        Parameters
        ----------
        tensors : List of array
            The tensors of the (two-layer) boundary MPS. legs (d,r1,r2,u), at boundary no d or no u
        chi_bmps : int or None
            The maximum bond-dimension of the boundary MPS
        cutoff : float
            The cutoff for singular values. Singular values below cutoff are discarded
        """
        self.tensors = tensors.copy()  # legs: (d,r1,r2,u), at boundary no d / no u
        self.L = len(tensors)
        self.norm = 1.
        self.chi_bmps = chi_bmps
        self.cutoff = cutoff

    # noinspection DuplicatedCode
    def absorb_col(self, bra_col: List[array], ket_col: List[array], left_side: bool):
        if not left_side:
            # transpose l <-> r for all tensors in bra_col, ket_col. Then we can treat it like the left side
            # FUTURE fully implement instead
            bra_col = bra_col.copy()
            ket_col = ket_col.copy()

            bra_col[0] = np.transpose(bra_col[0], [0, 3, 2, 1])  # (p,r,u,l) -> (p,l,u,r)
            ket_col[0] = np.transpose(ket_col[0], [0, 3, 2, 1])  # (p,r,u,l) -> (p,l,u,r)

            # (puldr) -> (purdl)
            # noinspection DuplicatedCode
            bra_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 4, 3, 2]), bra_col[1:-1])
            ket_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 4, 3, 2]), ket_col[1:-1])

            bra_col[-1] = np.transpose(bra_col[-1], [0, 3, 2, 1])  # (p,l,d,r) -> (p,r,d,l)
            ket_col[-1] = np.transpose(ket_col[-1], [0, 3, 2, 1])  # (p,l,d,r) -> (p,r,d,l)

        # sweep up with SVD
        M = np.tensordot(self.tensors[0], ket_col[0], [0, 3])  # (r1,r2,u0) & (p,r1,u1,l1) -> (r2,u0,p,r1,u1)
        # (r2,u0,p,r1,u1) & (p*,r2,u2,l2) -> (u0,r1,u1,r2,u2)
        M = np.tensordot(M, (bra_col[0]), [[2, 0], [0, 3]])
        u0, r1, u1, r2, u2 = M.shape
        # (u0,r1,u1,r2,u2) -> (r1,r2,u0,u1,u2) -> (R,U) -> (R,a) & (a) & (a,U)
        U, S, V = svd(np.reshape(np.transpose(M, [1, 3, 0, 2, 4]), [r1 * r2, u0 * u1 * u2]))
        SV = S[:, None] * V  # (D,U)
        _, a = U.shape
        self.tensors[0] = np.reshape(U, [r1, r2, a])  # (R,a) -> (r1,r2,U)
        for n in range(1, self.L - 1):
            # (d0,r1,r2,u0) & (p,u1,l1,d1,r1) -> (d0,r2,u0,p,u1,d1,r1)
            M = np.tensordot(self.tensors[n], ket_col[n], [1, 2])
            # (d0,r2,u0,p,u1,d1,r1) & (p*,u2,l2,d2,r2) -> (d0,u0,u1,d1,r1,u2,d2,r2)
            M = np.tensordot(M, (bra_col[n]), [[3, 1], [0, 2]])
            d0, u0, u1, d1, r1, u2, d2, r2 = M.shape
            # (d0,u0,u1,d1,r1,u2,d2,r2) -> (ddd,r1,r2,uuu) -> (D,R,U)
            M = np.reshape(np.transpose(M, [0, 3, 6, 4, 7, 1, 2, 5]), [d0 * d1 * d2, r1 * r2, u0 * u1 * u2])
            M = np.tensordot(SV, M, [1, 0])  # (D,U) & (D,R,U) -> (D,R,U)
            d, _, _ = M.shape
            U, S, V = svd(np.reshape(M, [d * r1 * r2, u0 * u1 * u2]))  # (D,R,U) -> (DR,U) -> (DR,a) & (a) & (a,U)
            _, a = U.shape
            self.tensors[n] = np.reshape(U, [d, r1, r2, a])  # (DR,a) -> (D,r1,r2,U)
            SV = S[:, None] * V  # (D,U)

        M = np.tensordot(self.tensors[-1], ket_col[-1], [1, 1])  # (d0,r1,r2) & (p,l1,d1,r1) -> (d0,r2,p,d1,r1)
        # (d0,r2,p,d1,r1) & (p*,l2,d2,r2) -> (d0,d1,r1,d2,r2)
        M = np.tensordot(M, (bra_col[-1]), [[1, 2], [1, 0]])
        d0, d1, r1, d2, r2 = M.shape
        # (d0,d1,r1,d2,r2) -> (d0,d1,d2,r1,r2) -> (D,R)
        M = np.reshape(np.transpose(M, [0, 1, 3, 2, 4]), [d0 * d1 * d2, r1 * r2])
        M = np.dot(SV, M)  # (D,U) & (D,R) -> (D,R)

        # FIXME track norm-change from truncated SVD?
        # sweep down with truncated svd
        U, S, V = svd_truncated(M, cutoff=self.cutoff, chi_max=self.chi_bmps)  # (D,R) -> (D,a) & (a) & (a,R)
        a, _ = V.shape
        self.tensors[-1] = np.reshape(V, [a, r1, r2])  # (d,r1,r2)
        for n in range(self.L - 2, 0, -1):
            M = np.tensordot(self.tensors[n], U * S, [3, 0])  # (d,r1,r2,u) & (d,u) -> (d,r1,r2,u)
            d, r1, r2, u = M.shape
            # (d,r1,r2,u) -> (d,RU) -> (d,a) & (a) & (a,RU)
            U, S, V = svd_truncated(np.reshape(M, [d, r1 * r2 * u]), cutoff=self.cutoff, chi_max=self.chi_bmps)
            a, _ = V.shape
            self.tensors[n] = np.reshape(V, [a, r1, r2, u])  # (a,RU) -> (d,r1,r2,u)
        M = np.tensordot(self.tensors[0], U * S, [2, 0])[None, :, :, :]  # (r1,r2,u) & (d,u) -> (r1,r2,u) -> (d,r1,r2,u)
        d, r1, r2, u = M.shape
        # (d,r1,r2,u) -> (d,RU) -> (d,a) & (a) & (a,RU)
        U, S, V = svd_truncated(np.reshape(M, [d, r1 * r2 * u]), cutoff=self.cutoff, chi_max=self.chi_bmps)
        assert U.shape == (1, 1)
        assert len(S) == 1
        assert V.shape[0] == 1
        self.tensors[0] = np.reshape(V, [1, r1, r2, u])[0, :, :, :] * U[0, 0]
        self.norm *= S[0]

    def contract_with(self, other) -> complex:
        assert self.L == other.L
        tens = np.tensordot(self.tensors[0], other.tensors[0], [[0, 1], [0, 1]])  # (r1,r2,u) & (r1,r2,u) -> (u,u)
        u, u_ = tens.shape
        col = [np.reshape(tens, [u * u_])]
        col += tree_multimap(_contract_with__bulk_contraction2, self.tensors[1:-1], other.tensors[1:-1])
        tens = np.tensordot(self.tensors[-1], other.tensors[-1], [[1, 2], [1, 2]])  # (d,r1,r2) & (d,r1,r2) -> (d,d)
        d, d_ = tens.shape
        col.append(np.reshape(tens, [d * d_]))
        res = np.linalg.multi_dot(col)
        return res * self.norm * other.norm


class BoundaryMPS3:
    def __init__(self, tensors, chi_bmps, cutoff):
        """

        Parameters
        ----------
        tensors : List of array
            The tensors of the (two-layer) boundary MPS. legs (d,r1,r2,r3,u), at boundary no d or no u
        chi_bmps : int or None
            The maximum bond-dimension of the boundary MPS
        cutoff : float
            The cutoff for singular values. Singular values below cutoff are discarded
        """
        self.tensors = tensors.copy()  # legs: (d,r1,r2,r3,u), at boundary no d / no u
        self.L = len(tensors)
        self.norm = 1.
        self.chi_bmps = chi_bmps
        self.cutoff = cutoff

    # noinspection DuplicatedCode
    def absorb_col(self, bra_col: List[array], op_col: List[array], ket_col: List[array], left_side: bool):
        if not left_side:
            # transpose l <-> r for all tensors in bra_col, ket_col, op_col
            # TODO fully implement instead
            bra_col = bra_col.copy()
            op_col = op_col.copy()
            ket_col = ket_col.copy()

            bra_col[0] = np.transpose(bra_col[0], [0, 3, 2, 1])  # (p,r,u,l) -> (p,l,u,r)
            op_col[0] = np.transpose(op_col[0], [0, 1, 4, 3, 2])  # (p,p*,r,u,l) -> (p,p*,l,u,r)
            ket_col[0] = np.transpose(ket_col[0], [0, 3, 2, 1])  # (p,r,u,l) -> (p,l,u,r)

            # (puldr) -> (purdl)
            bra_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 4, 3, 2]), bra_col[1:-1])
            op_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 2, 5, 4, 3]), op_col[1:-1])
            ket_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 4, 3, 2]), ket_col[1:-1])

            bra_col[-1] = np.transpose(bra_col[-1], [0, 3, 2, 1])  # (p,l,d,r) -> (p,r,d,l)
            op_col[-1] = np.transpose(op_col[-1], [0, 1, 4, 3, 2])  # (p,p*,l,d,r) -> (p,p*,r,d,l)
            ket_col[-1] = np.transpose(ket_col[-1], [0, 3, 2, 1])  # (p,l,d,r) -> (p,r,d,l)

        # sweep up with SVD
        M = np.tensordot(self.tensors[0], ket_col[0], [0, 3])  # (r1,r2,r3,u0) & (p,r1,u1,l1) -> (r2,r3,u0,p,r1,u1)
        # (r2,r3,u0,p,r1,u1) & (p,p*,r2,u2,l2) -> (r3,u0,r1,u1,p,r2,u2)
        M = np.tensordot(M, op_col[0], [[0, 3], [4, 1]])
        # (r3,u0,r1,u1,p,r2,u2) & (p*,r3,u3,l3) -> (u0,r1,u1,r2,u2,r3,u3)
        M = np.tensordot(M, (bra_col[0]), [[0, 4], [3, 0]])
        u0, r1, u1, r2, u2, r3, u3 = M.shape
        # (u0,r1,u1,r2,u2,r3,u3) -> (r1,r2,r3,u0,u1,u2,u3) -> (R,U) -> (R,a) & (a) & (a,U)
        U, S, V = svd(np.reshape(np.transpose(M, [1, 3, 5, 0, 2, 4, 6]), [r1 * r2 * r3, u0 * u1 * u2 * u3]))
        SV = S[:, None] * V  # (D,U)
        _, a = U.shape
        self.tensors[0] = np.reshape(U, [r1, r2, r3, a])  # (R,a) -> (r1,r2,r3,U)
        for n in range(1, self.L - 1):
            # (d0,r1,r2,r3,u0) & (p,u1,l1,d1,r1) -> (d0,r2,r3,u0,p,u1,d1,r1)
            M = np.tensordot(self.tensors[n], ket_col[n], [1, 2])
            # (d0,r2,r3,u0,p,u1,d1,r1) & (p,p*,u2,l2,d2,r2) -> (d0,r3,u0,u1,d1,r1,p,u2,d2,r2)
            M = np.tensordot(M, op_col[n], [[4, 1], [1, 3]])
            # (d0,r3,u0,u1,d1,r1,p,u2,d2,r2) & (p*,u3,l3,d3,r3) -> (d0,u0,u1,d1,r1,u2,d2,r2,u3,d3,r3)
            M = np.tensordot(M, (bra_col[n]), [[6, 1], [0, 2]])
            d0, u0, u1, d1, r1, u2, d2, r2, u3, d3, r3 = M.shape
            # (d0,u0,u1,d1,r1,u2,d2,r2,u3,d3,r3) -> (D,R,U)
            M = np.reshape(np.transpose(M, [0, 3, 6, 9, 4, 7, 10, 1, 2, 5, 8]), [d0*d1*d2*d3, r1*r2*r3, u0*u1*u2*u3])
            M = np.tensordot(SV, M, [1, 0])  # (D,U) & (D,R,U) -> (D,R,U)
            # (D,R,U) -> (DR,U) -> (DR,a) & (a) & (a,U)
            d, _, _ = M.shape
            U, S, V = svd(np.reshape(M, [d * r1*r2*r3, u0*u1*u2*u3]))
            _, a = U.shape
            self.tensors[n] = np.reshape(U, [d, r1, r2, r3, a])  # (D,r1,r2,r3,U)
            SV = S[:, None] * V  # (D,U)

        M = np.tensordot(self.tensors[-1], ket_col[-1], [1, 1])  # (d0,r1,r2,r3) & (p,l1,d1,r1) -> (d0,r2,r3,p,d1,r1)
        # (d0,r2,r3,p,d1,r1) & (p,p*,l2,d2,r2) -> (d0,r3,d1,r1,p,d2,r2)
        M = np.tensordot(M, op_col[-1], [[1, 3], [2, 1]])
        # (d0,r3,d1,r1,p,d2,r2) & (p*,l3,d3,r3) -> (d0,d1,r1,d2,r2,d3,r3)
        M = np.tensordot(M, (bra_col[-1]), [[1, 4], [1, 0]])
        d0, d1, r1, d2, r2, d3, r3 = M.shape
        # (d0,d1,r1,d2,r2,d3,r3) -> (d0,d1,d2,d3,r1,r2,r3) -> (D,R)
        M = np.reshape(np.transpose(M, [0, 1, 3, 5, 2, 4, 6]), [d0*d1*d2*d3, r1*r2*r3])
        M = np.dot(SV, M)  # (D,U) & (D,R) -> (D,R)

        # FIXME track norm-change from truncated SVD?
        # sweep down with truncated svd
        U, S, V = svd_truncated(M, cutoff=self.cutoff, chi_max=self.chi_bmps)  # (D,R) -> (D,a) & (a) & (a,R)
        a, _ = V.shape
        self.tensors[-1] = np.reshape(V, [a, r1, r2, r3])  # (d,r1,r2,r3)
        for n in range(self.L - 2, 0, -1):
            M = np.tensordot(self.tensors[n], U * S, [4, 0])  # (d,r1,r2,r3,u) & (d,u) -> (d,r1,r2,r3,u)
            d, r1, r2, r3, u = M.shape
            # (d,r1,r2,r3,u) -> (d,RU) -> (d,a) & (a) & (a,RU)
            U, S, V = svd_truncated(np.reshape(M, [d, r1 * r2 * r3 * u]), cutoff=self.cutoff, chi_max=self.chi_bmps)
            a, _ = V.shape
            self.tensors[n] = np.reshape(V, [a, r1, r2, r3, u])  # (a,RU) -> (d,r1,r2,r3,u)
        # (r1,r2,r3,u) & (d,u) -> (r1,r2,r3,u) -> (d,r1,r2,r3,u)
        M = np.tensordot(self.tensors[0], U * S, [3, 0])[None, :, :, :, :]
        d, r1, r2, r3, u = M.shape
        # (d,r1,r2,r3,u) -> (d,RU) -> (d,a) & (a) & (a,RU)
        U, S, V = svd_truncated(np.reshape(M, [d, r1 * r2 * r3 * u]), cutoff=self.cutoff, chi_max=self.chi_bmps)
        assert U.shape == (1, 1)
        assert len(S) == 1
        assert V.shape[0] == 1
        self.tensors[0] = np.reshape(V, [1, r1, r2, r3, u])[0, :, :, :] * U[0, 0]
        self.norm *= S[0]

    def contract_with(self, other) -> complex:
        # self is left
        assert self.L == other.L
        # (r1,r2,r3,u) & (r1,r2,r3,u) -> (u,u)
        tens = np.tensordot(self.tensors[0], other.tensors[0], [[0, 1, 2], [0, 1, 2]])
        u, u_ = tens.shape
        col = [np.reshape(tens, [u * u_])]
        col += tree_multimap(_contract_with__bulk_contraction3, self.tensors[1:-1], other.tensors[1:-1])
        # (d,r1,r2,r3) & (d,r1,r2,r3) -> (d,d)
        tens = np.tensordot(self.tensors[-1], other.tensors[-1], [[1, 2, 3], [1, 2, 3]])
        d, d_ = tens.shape
        col.append(np.reshape(tens, [d * d_]))
        res = np.linalg.multi_dot(col)
        return res * self.norm * other.norm


def trivial_boundary_mps2(l, chi_bmps, cutoff):
    tensors = [np.reshape(1., 3 * [1])] + [np.reshape(1., 4 * [1])] * (l - 2) + [np.reshape(1., 3 * [1])]
    return BoundaryMPS2(tensors, chi_bmps, cutoff)


def trivial_boundary_mps3(l, chi_bmps, cutoff):
    tensors = [np.reshape(1., 4 * [1])] + [np.reshape(1., 5 * [1])] * (l - 2) + [np.reshape(1., 4 * [1])]
    return BoundaryMPS3(tensors, chi_bmps, cutoff)


# noinspection DuplicatedCode
def add_trivial_legs_left(col, num_phys_legs=1):
    new_col = [np.nan for _ in range(len(col))]
    if num_phys_legs == 1:
        new_col[0] = col[0][:, :, :, None]  # (p,r,u) -> (p,r,u,l)
        # (p,d,r,u) -> (p,u,d,r) -> (p,u,l,d,r)
        new_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 3, 1, 2])[:, :, None, :, :], col[1:-1])
        new_col[-1] = col[-1][:, None, :, :]  # (p,d,r) -> (p,l,d,r)
        return new_col
    elif num_phys_legs == 2:
        new_col[0] = col[0][:, :, :, :, None]  # (p,p*,r,u) -> (p,p*,r,u,l)
        # (p,p*,d,r,u) -> (p,p*,u,d,r) -> (p,p*,u,l,d,r)
        new_col[1:-1] = tree_map(lambda arr: np.transpose(arr, [0, 1, 4, 2, 3])[:, :, :, None, :, :], col[1:-1])
        new_col[-1] = col[-1][:, :, None, :, :]  # (p,p*,d,r) -> (p,p*,l,d,r)
        return new_col
    else:
        raise NotImplementedError


def add_trivial_legs_right(col, num_phys_legs=1):
    new_col = [np.nan for _ in range(len(col))]
    if num_phys_legs == 1:
        new_col[0] = col[0][:, None, :, :]  # (p,u,l) -> (p,r,u,l)
        # (p,u,l,d) -> (p,u,l,d,r)
        new_col[1:-1] = tree_map(lambda arr: arr[:, :, :, :, None], col[1:-1])
        new_col[-1] = col[-1][:, :, :, None]  # (p,l,d) -> (p,l,d,r)
        return new_col
    elif num_phys_legs == 2:
        new_col[0] = col[0][:, :, None, :, :]  # (p,p*,u,l) -> (p,p*,r,u,l)
        # (p,p*,u,l,d) -> (p,p*,u,l,d,r)
        new_col[1:-1] = tree_map(lambda arr: arr[:, :, :, :, :, None], col[1:-1])
        new_col[-1] = col[-1][:, :, :, :, None]  # (p,p*,l,d) -> (p,p*,l,d,r)
        return new_col
    else:
        raise NotImplementedError


def _contract_with__bulk_contraction2(arr1, arr2):
    tens = np.tensordot(arr1, arr2, [[1, 2], [1, 2]])  # (d,r1,r2,u) & (d,r1,r2,u) -> (d,u,d,u)
    d, u, d_, u_ = tens.shape
    return np.reshape(np.transpose(tens, [0, 2, 1, 3]), [d * d_, u * u_])


def _contract_with__bulk_contraction3(arr1, arr2):
    tens = np.tensordot(arr1, arr2, [[1, 2, 3], [1, 2, 3]])  # (d,r1,r2,r3,u) & (d,r1,r2,r3,u) -> (d,u,d,u)
    d, u, d_, u_ = tens.shape
    return np.reshape(np.transpose(tens, [0, 2, 1, 3]), [d * d_, u * u_])
