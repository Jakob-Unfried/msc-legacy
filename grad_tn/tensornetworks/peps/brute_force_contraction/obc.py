import numpy as onp
import grad_tn.tensornetworks.peps.brute_force_contraction._obc_instructions as instructions
from utils.jax_utils.ncon_jax import ncon


def two_layers(bra_tensors, ket_tensors):
    lx, ly = len(bra_tensors), len(bra_tensors[0])
    assert all(len(bra_tensors[x]) == ly for x in range(lx))
    assert len(ket_tensors) == lx
    assert all(len(ket_tensors[x]) == ly for x in range(lx))

    tensors = [ket_tensors[x][y] for y in range(ly) for x in range(lx)] \
              + [bra_tensors[x][y] for y in range(ly) for x in range(lx)]

    connects, order = _get_instructions2(lx, ly)
    return ncon(tensors, connects, order)


def three_layers(bra_tensors, op_tensors, ket_tensors):
    lx, ly = len(bra_tensors), len(bra_tensors[0])
    assert all(len(bra_tensors[x]) == ly for x in range(lx))
    assert len(op_tensors) == lx
    assert all(len(op_tensors[x]) == ly for x in range(lx))
    assert len(ket_tensors) == lx
    assert all(len(ket_tensors[x]) == ly for x in range(lx))

    tensors = [ket_tensors[x][y] for y in range(ly) for x in range(lx)] \
              + [op_tensors[x][y] for y in range(ly) for x in range(lx)] \
              + [bra_tensors[x][y] for y in range(ly) for x in range(lx)]

    connects, order = _get_instructions3(lx, ly)
    return ncon(tensors, connects, order)


def _get_instructions2(lx, ly):
    if (lx, ly) in instructions.connects2:
        return instructions.connects2[lx, ly], instructions.order2[lx, ly]

    N = lx * ly
    M = 2 * N - lx - ly

    # 1,...,M : ket layer
    # M+1,...,2M : bra layer
    # 2M+1,...,3M+N : p legs

    # functions that number the bonds ( in layers )
    def _d(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if y == 0:
            return None

        return (1 - lx) + x + (2 * lx - 1) * y + M * n

    def _l(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if x == 0:
            return None

        return x + (2 * lx - 1) * y + M * n

    def _r(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if x == lx - 1:
            return None

        return 1 + x + (2 * lx - 1) * y + M * n

    def _u(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if y == ly - 1:
            return None

        return lx + x + (2 * lx - 1) * y + M * n

    def _p(x, y):
        # leg between op and bra (p leg of op)
        return 2 * M + 1 + x + lx * y

    # tensors: [tens[l, x, y] for l in [ket, op, bra] for y in range(ly) for x in range(lx)]
    #           = [ ket[0,0], ket[1,0], ket[2,0], ..., op[...], ..., bra[...], ...]

    def connects_ket_tens(x, y):
        if x == 0:
            if y == 0:  # bottom left
                return [_p(x, y), _r(x, y, 0), _u(x, y, 0)]  # (p,r,u)
            elif y == ly - 1:  # top left
                return [_p(x, y), _d(x, y, 0), _r(x, y, 0)]  # (p,d,r)
            else:  # left
                return [_p(x, y), _d(x, y, 0), _r(x, y, 0), _u(x, y, 0)]  # (p,d,r,u)

        elif x == lx - 1:
            if y == 0:  # bottom right
                return [_p(x, y), _u(x, y, 0), _l(x, y, 0)]  # (p,u,l)
            elif y == ly - 1:  # top right
                return [_p(x, y), _l(x, y, 0), _d(x, y, 0)]  # (p,l,d)
            else:  # right
                return [_p(x, y), _u(x, y, 0), _l(x, y, 0), _d(x, y, 0)]  # (p,u,l,d)

        else:
            if y == 0:  # bottom
                return [_p(x, y), _r(x, y, 0), _u(x, y, 0), _l(x, y, 0)]  # (p,r,u,l)
            elif y == ly - 1:  # top
                return [_p(x, y), _l(x, y, 0), _d(x, y, 0), _r(x, y, 0)]  # (p,l,d,r)
            else:  # bulk
                return [_p(x, y), _u(x, y, 0), _l(x, y, 0), _d(x, y, 0), _r(x, y, 0)]  # (p,u,l,d,r)

    def connects_bra_tens(x, y):
        if x == 0:
            if y == 0:  # bottom left
                return [_p(x, y), _r(x, y, 1), _u(x, y, 1)]  # (p,r,u)
            elif y == ly - 1:  # top left
                return [_p(x, y), _d(x, y, 1), _r(x, y, 1)]  # (p,d,r)
            else:  # left
                return [_p(x, y), _d(x, y, 1), _r(x, y, 1), _u(x, y, 1)]  # (p,d,r,u)

        elif x == lx - 1:
            if y == 0:  # bottom right
                return [_p(x, y), _u(x, y, 1), _l(x, y, 1)]  # (p,u,l)
            elif y == ly - 1:  # top right
                return [_p(x, y), _l(x, y, 1), _d(x, y, 1)]  # (p,l,d)
            else:  # right
                return [_p(x, y), _u(x, y, 1), _l(x, y, 1), _d(x, y, 1)]  # (p,u,l,d)

        else:
            if y == 0:  # bottom
                return [_p(x, y), _r(x, y, 1), _u(x, y, 1), _l(x, y, 1)]  # (p,r,u,l)
            elif y == ly - 1:  # top
                return [_p(x, y), _l(x, y, 1), _d(x, y, 1), _r(x, y, 1)]  # (p,l,d,r)
            else:  # bulk
                return [_p(x, y), _u(x, y, 1), _l(x, y, 1), _d(x, y, 1), _r(x, y, 1)]  # (p,u,l,d,r)

    connects = [connects_ket_tens(x, y) for y in range(ly) for x in range(lx)] \
               + [connects_bra_tens(x, y) for y in range(ly) for x in range(lx)]

    order = []

    def order_helper(x, y, *funs):
        # used when adding connects of the tensors at (x,y) to order
        # funs are all functions, e.g. _l and _r that should be appended at this time.
        # returns fun(x, y, n) for all fun in funs and n in range(2) as well as _p(x,y)
        # in optimal order
        res = []
        res.extend([fun(x, y, 0) for fun in funs])
        res.append(_p(x, y))
        res.extend([fun(x, y, 1) for fun in funs])
        return res

    # BOTTOM LEFT QUARTER
    # bottom row
    order.append(_p(0, 0))
    for x in range(1, lx // 2):
        order.extend(order_helper(x, 0, _l))
    # other rows
    for y in range(1, ly // 2):
        # x = 0
        order.extend(order_helper(0, y, _d))
        for x in range(1, lx // 2):
            order.extend(order_helper(x, y, _l, _d))

    # BOTTOM RIGHT QUARTER
    # bottom row
    order.extend([_p(lx // 2, 0)])
    for x in range(lx // 2 + 1, lx):
        order.extend(order_helper(x, 0, _l))
    # other rows
    for y in range(1, ly // 2):
        # x = lx // 2
        order.extend(order_helper(lx // 2, y, _d))
        for x in range(lx // 2 + 1, lx):
            order.extend(order_helper(x, y, _l, _d))

    # BOTTOM HALF
    order.extend([_l(lx // 2, y, n) for n in [0, 1] for y in range(ly // 2)])

    # TOP LEFT QUARTER
    # bottom row
    order.extend([_p(0, ly // 2)])
    for x in range(1, lx // 2):
        order.extend(order_helper(x, ly // 2, _l))
    # other rows
    for y in range(ly // 2 + 1, ly):
        # x = 0
        order.extend(order_helper(0, y, _d))
        for x in range(1, lx // 2):
            order.extend(order_helper(x, y, _l, _d))

    # THREE QUARTERS
    order.extend([_d(x, ly // 2, n) for n in [0, 1] for x in range(lx // 2)])

    # FILL LAST QUARTER
    for x in range(lx // 2, lx):
        for y in range(ly // 2, ly):
            order.extend(order_helper(x, y, _l, _d))

    # checks:
    n_layers = 2
    n_labels = n_layers * (2 * lx * ly - lx - ly) + (n_layers - 1) * lx * ly
    connects_flat = onp.array([elem for sublist in connects for elem in sublist])
    order_arr = onp.array(order)
    assert onp.all(0 < connects_flat) and onp.all(connects_flat <= n_labels)
    assert onp.all(0 < order_arr) and onp.all(order_arr <= n_labels)
    for n in range(1, n_labels + 1):
        assert onp.sum(connects_flat == n) == 2
        assert onp.sum(order_arr == n) == 1

    return connects, order


def _get_instructions3(lx, ly):
    if (lx, ly) in instructions.connects3:
        return instructions.connects3[lx, ly], instructions.order3[lx, ly]

    N = lx * ly
    M = 2 * N - lx - ly

    # 1,...,M : ket layer
    # M+1,...,2M : op layer
    # 2M+2,...,3M : bra layer
    # 3M+1,...,3M+N : ket - op
    # 3M+N+1,...,3M+2N : op - bra

    # functions that number the bonds ( in layers )
    def _d(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if y == 0:
            return None

        return (1 - lx) + x + (2 * lx - 1) * y + M * n

    def _l(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if x == 0:
            return None

        return x + (2 * lx - 1) * y + M * n

    def _r(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if x == lx - 1:
            return None

        return 1 + x + (2 * lx - 1) * y + M * n

    def _u(x, y, n):
        if (x < 0) or (x >= lx) or (y < 0) or (y >= ly):
            raise RuntimeError

        if y == ly - 1:
            return None

        return lx + x + (2 * lx - 1) * y + M * n

    def _p_star(x, y):
        # leg between ket and op (p* leg of op)
        return 3 * M + 1 + x + lx * y

    def _p(x, y):
        # leg between op and bra (p leg of op)
        return 3 * M + N + 1 + x + lx * y

    # tensors: [tens[l, x, y] for l in [ket, op, bra] for y in range(ly) for x in range(lx)]
    #           = [ ket[0,0], ket[1,0], ket[2,0], ..., op[...], ..., bra[...], ...]

    def connects_ket_tens(x, y):
        if x == 0:
            if y == 0:  # bottom left
                return [_p_star(x, y), _r(x, y, 0), _u(x, y, 0)]  # (p,r,u)
            elif y == ly - 1:  # top left
                return [_p_star(x, y), _d(x, y, 0), _r(x, y, 0)]  # (p,d,r)
            else:  # left
                return [_p_star(x, y), _d(x, y, 0), _r(x, y, 0), _u(x, y, 0)]  # (p,d,r,u)

        elif x == lx - 1:
            if y == 0:  # bottom right
                return [_p_star(x, y), _u(x, y, 0), _l(x, y, 0)]  # (p,u,l)
            elif y == ly - 1:  # top right
                return [_p_star(x, y), _l(x, y, 0), _d(x, y, 0)]  # (p,l,d)
            else:  # right
                return [_p_star(x, y), _u(x, y, 0), _l(x, y, 0), _d(x, y, 0)]  # (p,u,l,d)

        else:
            if y == 0:  # bottom
                return [_p_star(x, y), _r(x, y, 0), _u(x, y, 0), _l(x, y, 0)]  # (p,r,u,l)
            elif y == ly - 1:  # top
                return [_p_star(x, y), _l(x, y, 0), _d(x, y, 0), _r(x, y, 0)]  # (p,l,d,r)
            else:  # bulk
                return [_p_star(x, y), _u(x, y, 0), _l(x, y, 0), _d(x, y, 0), _r(x, y, 0)]  # (p,u,l,d,r)

    def connects_op_tens(x, y):
        if x == 0:
            if y == 0:  # bottom left
                return [_p(x, y), _p_star(x, y), _r(x, y, 1), _u(x, y, 1)]  # (p,p*,r,u)
            elif y == ly - 1:  # top left
                return [_p(x, y), _p_star(x, y), _d(x, y, 1), _r(x, y, 1)]  # (p,p*,d,r)
            else:  # left
                return [_p(x, y), _p_star(x, y), _d(x, y, 1), _r(x, y, 1), _u(x, y, 1)]  # (p,p*,d,r,u)

        elif x == lx - 1:
            if y == 0:  # bottom right
                return [_p(x, y), _p_star(x, y), _u(x, y, 1), _l(x, y, 1)]  # (p,p*,u,l)
            elif y == ly - 1:  # top right
                return [_p(x, y), _p_star(x, y), _l(x, y, 1), _d(x, y, 1)]  # (p,p*,l,d)
            else:  # right
                return [_p(x, y), _p_star(x, y), _u(x, y, 1), _l(x, y, 1), _d(x, y, 1)]  # (p,p*,u,l,d)

        else:
            if y == 0:  # bottom
                return [_p(x, y), _p_star(x, y), _r(x, y, 1), _u(x, y, 1), _l(x, y, 1)]  # (p,p*,r,u,l)
            elif y == ly - 1:  # top
                return [_p(x, y), _p_star(x, y), _l(x, y, 1), _d(x, y, 1), _r(x, y, 1)]  # (p,p*,l,d,r)
            else:  # bulk
                return [_p(x, y), _p_star(x, y), _u(x, y, 1), _l(x, y, 1), _d(x, y, 1), _r(x, y, 1)]  # (p,p*,u,l,d,r)

    def connects_bra_tens(x, y):
        if x == 0:
            if y == 0:  # bottom left
                return [_p(x, y), _r(x, y, 2), _u(x, y, 2)]  # (p,r,u)
            elif y == ly - 1:  # top left
                return [_p(x, y), _d(x, y, 2), _r(x, y, 2)]  # (p,d,r)
            else:  # left
                return [_p(x, y), _d(x, y, 2), _r(x, y, 2), _u(x, y, 2)]  # (p,d,r,u)

        elif x == lx - 1:
            if y == 0:  # bottom right
                return [_p(x, y), _u(x, y, 2), _l(x, y, 2)]  # (p,u,l)
            elif y == ly - 1:  # top right
                return [_p(x, y), _l(x, y, 2), _d(x, y, 2)]  # (p,l,d)
            else:  # right
                return [_p(x, y), _u(x, y, 2), _l(x, y, 2), _d(x, y, 2)]  # (p,u,l,d)

        else:
            if y == 0:  # bottom
                return [_p(x, y), _r(x, y, 2), _u(x, y, 2), _l(x, y, 2)]  # (p,r,u,l)
            elif y == ly - 1:  # top
                return [_p(x, y), _l(x, y, 2), _d(x, y, 2), _r(x, y, 2)]  # (p,l,d,r)
            else:  # bulk
                return [_p(x, y), _u(x, y, 2), _l(x, y, 2), _d(x, y, 2), _r(x, y, 2)]  # (p,u,l,d,r)

    connects = [connects_ket_tens(x, y) for y in range(ly) for x in range(lx)] \
               + [connects_op_tens(x, y) for y in range(ly) for x in range(lx)] \
               + [connects_bra_tens(x, y) for y in range(ly) for x in range(lx)]

    order = []

    def order_helper(x, y, *funs):
        # used when adding connects of the tensors at (x,y) to order
        # funs are all functions, e.g. _l and _r that should be appended at this time.
        # returns fun(x, y, n) for all fun in funs and n in range(3) as well as _p(x,y) and _p_star(x,y)
        # in optimal order
        res = []
        res.extend([fun(x, y, 0) for fun in funs])
        res.append(_p_star(x, y))
        res.extend([fun(x, y, 1) for fun in funs])
        res.append(_p(x, y))
        res.extend([fun(x, y, 2) for fun in funs])
        return res

    # BOTTOM LEFT QUARTER
    # bottom row
    order.extend([_p(0, 0), _p_star(0, 0)])
    for x in range(1, lx // 2):
        order.extend(order_helper(x, 0, _l))
    # other rows
    for y in range(1, ly // 2):
        # x = 0
        order.extend(order_helper(0, y, _d))
        for x in range(1, lx // 2):
            order.extend(order_helper(x, y, _l, _d))

    # BOTTOM RIGHT QUARTER
    # bottom row
    order.extend([_p(lx // 2, 0), _p_star(lx // 2, 0)])
    for x in range(lx // 2 + 1, lx):
        order.extend(order_helper(x, 0, _l))
    # other rows
    for y in range(1, ly // 2):
        # x = lx // 2
        order.extend(order_helper(lx // 2, y, _d))
        for x in range(lx // 2 + 1, lx):
            order.extend(order_helper(x, y, _l, _d))

    # BOTTOM HALF (connect previous quarters)
    order.extend([_l(lx // 2, y, n) for n in range(3) for y in range(ly // 2)])

    # TOP LEFT QUARTER
    # bottom row
    order.extend([_p(0, ly // 2), _p_star(0, ly // 2)])
    for x in range(1, lx // 2):
        order.extend(order_helper(x, ly // 2, _l))
    # other rows
    for y in range(ly // 2 + 1, ly):
        # x = 0
        order.extend(order_helper(0, y, _d))
        for x in range(1, lx // 2):
            order.extend(order_helper(x, y, _l, _d))

    # THREE QUARTERS
    order.extend([_d(x, ly // 2, n) for n in range(3) for x in range(lx // 2)])

    # FILL LAST QUARTER
    for x in range(lx // 2, lx):
        for y in range(ly // 2, ly):
            order.extend(order_helper(x, y, _l, _d))

    # checks:
    n_layers = 3
    n_labels = n_layers * (2 * lx * ly - lx - ly) + (n_layers - 1) * lx * ly
    connects_flat = onp.array([elem for sublist in connects for elem in sublist])
    order_arr = onp.array(order)
    assert onp.all(0 < connects_flat) and onp.all(connects_flat <= n_labels)
    assert onp.all(0 < order_arr) and onp.all(order_arr <= n_labels)
    for n in range(1, n_labels + 1):
        assert onp.sum(connects_flat == n) == 2
        assert onp.sum(order_arr == n) == 1

    return connects, order


if __name__ == '__main__':
    # only used to generate _obc_instructions and hard-code them
    l_range = range(3, 11)

    print('connects3')
    for _lx in l_range:
        for _ly in l_range:
            _connects, _order = _get_instructions3(_lx, _ly)
            print(f'{_lx, _ly}: {_connects},')

    print('order3')
    for _lx in l_range:
        for _ly in l_range:
            _connects, _order = _get_instructions3(_lx, _ly)
            print(f'{_lx, _ly}: {_order},')

    print('connects2')
    for _lx in l_range:
        for _ly in l_range:
            _connects, _order = _get_instructions2(_lx, _ly)
            print(f'{_lx, _ly}: {_connects},')

    print('order2')
    for _lx in l_range:
        for _ly in l_range:
            _connects, _order = _get_instructions2(_lx, _ly)
            print(f'{_lx, _ly}: {_order},')
