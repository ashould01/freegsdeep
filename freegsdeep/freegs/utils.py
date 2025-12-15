import torch
from freegsdeep.utilstyping import *
    
def romberg(y: Tensor, dx=1.0, dim=-1) -> float:
    def _tupleset(t: Tuple, d: int, v: Any) -> Tuple:
        l = list(t)
        l[d] = v
        return tuple(l)
    nd = len(y.shape)
    Nsamps = y.shape[dim]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError(f"Number of samples must be 2^k + 1; got at least {n} > {Ninterv}")
    R = {}
    h = Ninterv * dx
    slice_all = (slice(None), ) * nd
    slice0 = _tupleset(slice_all, dim, 0)
    slicem1 = _tupleset(slice_all, dim, -1)
    R[(0, 0)] = (y[slice0] + y[slicem1]) * h / 2.0
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k + 1):
        start >>= 1
        slice_R = _tupleset(slice_R, dim, slice(start, stop, step))
        step >>= 1
        R[(i, 0)] = 0.5 * (R[(i - 1, 0)] + h * y[slice_R].sum(dim))
        for j in range(1, i + 1):
            prev = R[(i, j - 1)]
            R[(i, j)] = prev + (prev - R[(i - 1, j - 1)]) / \
                ((1 << (2 * j)) - 1)
        h /= 2.0
    
    return R[(k, k)]