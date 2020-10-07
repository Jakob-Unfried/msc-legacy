"""
Pure python versions of lax control-flow, intended for debugging
"""


def while_loop(cond_fun, body_fun, init_carry):

    carry = init_carry
    while cond_fun(carry):
        carry = body_fun(carry)
    return carry


def fori_loop(lower, upper, body_fun, init_carry):

    carry = init_carry
    for i in range(lower, upper):
        carry = body_fun(i, carry)
    return carry
