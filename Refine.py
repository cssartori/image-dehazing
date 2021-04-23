"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np

from numba import jit, njit, prange
from typing import Callable
try:
    from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
except (ModuleNotFoundError, ImportError):
    from numba.errors import NumbaDeprecationWarning, NumbaWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
try:
    import cupy as cp #pylint: disable= import-error
    hasGPU = True
except Exception: #pylint: disable= broad-except
    hasGPU = False


@jit(parallel=True, fastmath= True)
def guided_filter(imageArray:np.ndarray, p:np.ndarray, r:int= 40, eps:float= 1e-3) -> np.ndarray:
    """
    Filter refinement under the guidance of an image. O(N) implementation.
    According to the reference paper http://research.microsoft.com/en-us/um/people/kahe/eccv10/

    Parameters
    -----------
    imageArray: np.ndarray
        an H*W RGB image used as guidance.

    p: np.ndarray
        the H*W filter to be guided

    r: int (in pixels, default=40)
        the radius of the guided filter

    eps: float (default=1e-3)
        the epsilon parameter

    Return
    -----------
    The guided filter p'.
    """
    #H: height, W: width, C:colors
    H, W, C = imageArray.shape
    #S is a matrix with the sizes of each local patch (window wk)
    S = _boxFilter(_matrixOp(np.ones, (H, W)), r)
    if hasGPU:
        # Run this op on the GPU
        imageArray = cp.asarray(imageArray)
        p = cp.asarray(p)
    zeroes = _matrixOp(np.empty, (C, H, W))
    #the mean value of each channel in imageArray
    mean_i = zeroes.copy()
    for c in prange(0, C): #pylint: disable= not-an-iterable
        mean_i[c] = _boxFilter(imageArray[:,:, c], r)/S
    #the mean of the guided filter p
    mean_p = _boxFilter(p, r)/S
    #the correlation of (imageArray, p) corr_ip in each channel
    mean_ip = zeroes.copy()
    for c in prange(0, C): #pylint: disable= not-an-iterable
        mean_ip[c] = _boxFilter(imageArray[:,:, c]*p, r)/S

    #covariance of (imageArray, p) cov_ip in each channel
    cov_ip = zeroes.copy()
    for c in prange(0, C): #pylint: disable= not-an-iterable
        cov_ip[c] = mean_ip[c] - mean_i[c]*mean_p
    #variance of imageArray in each local patch (window wk), used to build the matrix sigma_k in eq.(14)
    #The variance in each window is a 3x3 symmetric matrix with variance as its values:
    #           rr, rg, rb
    #   sigma = rg, gg, gb
    #           rb, gb, bb
    var_i = _matrixOp(np.empty,(C, C, H, W))
    # Channel-on-channel variance
    # Do in a parallelizable loop
    for i in prange(0, 3): #pylint: disable= not-an-iterable
        for j in prange(i, 3): #pylint: disable= not-an-iterable
            # The i, j combination represents the channels affected
            # eg, (0, 0) for (red, red)
            var_i[i, j] = _boxFilter(imageArray[:,:, i]*imageArray[:,:, j], r)/S - mean_i[i]*mean_i[j]
    if hasGPU:
        cov_ip = cp.asnumpy(cov_ip)
        var_i = cp.asnumpy(var_i)
    # We break this into an external call so we can use Numba's
    # optimized njit implementation
    a = doGridLoop((H, W, C), cov_ip, var_i, eps)

    if hasGPU:
        a = cp.asarray(a)
    b = mean_p - a[:,:, 0]*mean_i[0,:,:] - a[:,:, 1]*mean_i[1,:,:] - a[:,:, 2]*mean_i[2,:,:] #eq.(15)

    # the filter p'  eq.(16)
    pp = (
        _boxFilter(a[:,:, 0], r)*imageArray[:,:, 0]
        + _boxFilter(a[:,:, 1], r)*imageArray[:,:, 1]
        + _boxFilter(a[:,:, 2], r)*imageArray[:,:, 2]
        + _boxFilter(b, r)
        ) / S
    if hasGPU:
        pp = cp.asnumpy(pp)
    return pp

@njit(cache= True) # Row dependencies means can't be parallel
def yCumSum(a:np.ndarray) -> np.ndarray:
    """
    Numba based computation of y-direction
    cumulative sum. Can't be parallel!
    """
    out = np.empty_like(a)
    out[0, :] = a[0, :]
    for i in prange(1, a.shape[0]): #pylint: disable= not-an-iterable
        out[i, :] = a[i, :] + out[i - 1, :]
    return out

@njit(parallel= True, cache= True)
def xCumSum(a:np.ndarray) -> np.ndarray:
    """
    Numba-based parallel computation
    of X-direction cumulative sum
    """
    out = np.empty_like(a)
    for i in prange(a.shape[0]): #pylint: disable= not-an-iterable
        out[i, :] = np.cumsum(a[i, :])
    return out

@jit
def _boxFilter(m:np.ndarray, r:int, gpu:bool= hasGPU) -> np.ndarray:
    if gpu:
        m = cp.asnumpy(m)
    out = __boxfilter__(m, r)
    if gpu:
        return cp.asarray(out)
    return out


@njit(cache= True)
def __boxfilter__(m:np.ndarray, r:int):
    """
    Fast box filtering implementation, O(1) time.

    Parameters
    ----------
    m:  a 2-D matrix data normalized to [0.0, 1.0]
    r:  radius of the window considered

    Return
    -----------
    The filtered matrix m'.
    """
    #H: height, W: width
    H, W = m.shape
    #the output matrix m'
    mp = np.empty(m.shape)

    #cumulative sum over y axis
    ySum = yCumSum(m) #np.cumsum(m, axis=0)
    #copy the accumulated values of the windows in y
    mp[0:r+1,: ] = ySum[r:(2*r)+1,: ]
    #differences in y axis
    mp[r+1:H-r,: ] = ySum[(2*r)+1:,: ] - ySum[ :H-(2*r)-1,: ]
    mp[(-r):,: ] = ySum[-1, :].repeat(r).reshape((-1, r)).T - ySum[H-(2*r)-1:H-r-1,: ]

    #cumulative sum over x axis
    xSum = xCumSum(mp) #np.cumsum(mp, axis=1)
    #copy the accumulated values of the windows in x
    mp[:, 0:r+1] = xSum[:, r:(2*r)+1]
    #difference over x axis
    mp[:, r+1:W-r] = xSum[:, (2*r)+1: ] - xSum[:, :W-(2*r)-1]
    # A few Numba njit fixes:
    # 1. Can't use np.newaxis / NOne on a slice (xSum[:, -1][:, np.newaxis]), must use reshape instead
    # 2. Can't do on bare slice, have to copy first (https://github.com/numba/numba/issues/4917)
    # np.save("numba_0.49.1_crash_py3.6.10_numpy1.18.4+mkl_win10x64.npy", xSum)
    mp[:, -r: ] = xSum[:, -1].copy().reshape(-1, 1).repeat(r).reshape((-1, r)) - xSum[:, W-(2*r)-1:W-r-1]
    return mp

@njit(parallel=True, cache= True)
def doGridLoop(fillShape:tuple, cov_ip:np.ndarray, var_i:np.ndarray, eps:float) -> np.ndarray:
    """
    Loop over the grid
    """
    H, W, C = fillShape
    # CuPY's internal loop here is VERY slow
    # So we'll do this in Numpy
    a = np.empty((H, W, C))
    #coordGrid = createCoordinateGrid(imageArray.shape).reshape(-1).tolist()
    for i in prange(0, H): #pylint: disable= not-an-iterable
        for j in prange(0, W): #pylint: disable= not-an-iterable
            sigma = np.array([
                                [
                                    var_i[0, 0, i, j],
                                    var_i[0, 1, i, j],
                                    var_i[0, 2, i, j]
                                ],
                                [
                                    var_i[0, 1, i, j],
                                    var_i[1, 1, i, j],
                                    var_i[1, 2, i, j]
                                ],
                                [
                                    var_i[0, 2, i, j],
                                    var_i[1, 2, i, j],
                                    var_i[2, 2, i, j]
                                ]
                        ])
            #covariance of (imageArray, p) in pixel (i,j) for the 3 channels
            cov_ip_ij = np.array([ cov_ip[0, i, j], cov_ip[1, i, j], cov_ip[2, i, j]])
            inverted = np.linalg.inv(sigma + eps * np.identity(3))
            a[i, j] = np.dot(cov_ip_ij, inverted) #eq.(14)
    return a


from inspect import getmembers, isfunction
fnPoolNp = [o for o in getmembers(np) if isfunction(o[1])]
try:
    fnPool = [o for o in getmembers(cp) if isfunction(o[1])]
except Exception: #pylint: disable= broad-except
    fnPool = list()

def _matrixOp(fn:Callable, *args, **kwargs):
    if not hasGPU:
        return fn(*args, **kwargs)
    if isinstance(fn, str):
        lookupFunction = fn
    else:
        lookupFunction = fn.__name__
    result = 'err'
    for fnName, fnExec in fnPool:
        if fnName == lookupFunction:
            try:
                result = fnExec(*args, **kwargs)
                break
            except Exception: #pylint: disable= broad-except
                for fnNameNp, fnExecNp in fnPoolNp:
                    if fnNameNp == lookupFunction:
                        result = fnExecNp(*args, **kwargs)
                        break
    if isinstance(result, str):
        if not isinstance(fn, str):
            # Return as-called
            return fn(*args, **kwargs)
        raise RuntimeError(f"Function `{fn}` not found in Numpy or Cupy namespace")
    return result
