"""
explicitly contracting a closed tensor-network
with straight-forward / naive VJP
"""

from jax import partial, custom_vjp, tree_map
from utils.jax_utils.ncon_jax import ncon


@partial(custom_vjp, nondiff_argnums=(1, 2, 3))
def contract_closed_network(tensors, connects, contraction_order=None, check_network=True):
    if not all([all([con > 0 for con in tens_cons]) for tens_cons in connects]):
        raise ValueError('Not a closed network')

    return ncon(tensors, connects, contraction_order, check_network)


def _contract_closed_network_fwd(tensors, connects, contraction_order, check_network):
    val = contract_closed_network(tensors, connects, contraction_order, check_network)
    res = (tensors,)
    return val, res


def _contract_closed_network_bwd(connects, contraction_order, check_network, res, val_bar):
    tensors, = res
    tensors_bar = tree_map(lambda n: val_bar * _get_env(tensors, connects, contraction_order, check_network, n),
                           list(range(len(tensors))))
    # equivalent [_get_env(tensors, connects, contraction_order, check_network, n) for n in range(len(tensors))]
    return tensors_bar,


def _get_env(tensors, connects, contraction_order, check_network, n):
    assert 0 <= n < len(tensors)
    # remove tensors[n]
    tensors = tensors[:n] + tensors[n + 1:]

    # adjust connects. deep copy to avoid list-mutation problems
    connects = [sublist[:] for sublist in connects]
    # replace labels of legs that would be contracted with tensors[n] by open labels
    cons_to_be_removed = connects.pop(n)
    for n, con in enumerate(cons_to_be_removed):
        connects = [[-(n + 1) if c == con else c for c in sublist] for sublist in connects]

    # remove labels from order
    if contraction_order:
        contraction_order = [i for i in contraction_order if (i not in cons_to_be_removed)]

    env = ncon(tensors, connects, contraction_order, check_network)

    return env


contract_closed_network.defvjp(_contract_closed_network_fwd, _contract_closed_network_bwd)
