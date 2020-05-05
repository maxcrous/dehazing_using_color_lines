""" Due to the lack of support of the numpy error_model for Numba jitclasses,
    This document overrides the jitclass baseclass and passes the numpy error
    model to all jit functions.
"""
from numba import float64, boolean
from numba.experimental.jitclass import base


def override(*args, **kwargs):
    from numba import njit
    return njit(*args, **kwargs)

base.njit = override(error_model='numpy')
from numba.experimental import jitclass


spec = [('point', float64[:]),
        ('direction', float64[:]),
        ('patch', float64[:, :, :]),
        ('transmission', float64),
        ('support_matrix', boolean[:, :])]

jitclass = jitclass
