"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np

from numba import jit, njit, prange
try:
    from numba.core.errors import NumbaDeprecationWarning, NumbaWarning
except ModuleNotFoundError:
    from numba.errors import NumbaDeprecationWarning, NumbaWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
try:
    import cupy as cp
    hasGPU = True
except Exception:
    hasGPU = False


@jit(parallel=True, fastmath= True)
def guided_filter(imageArray, p, r=40, eps=1e-3):
    """
    Filter refinement under the guidance of an image. O(N) implementation.
    According to the reference paper http://research.microsoft.com/en-us/um/people/kahe/eccv10/

    Parameters
    -----------
    imageArray:    an H*W RGB image used as guidance.
    p:        the H*W filter to be guided
    r:        the radius of the guided filter (in pixels, default=40)
    eps:      the epsilon parameter (default=1e-3)

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
    for c in prange(0, C):
        mean_i[c] = _boxFilter(imageArray[:,:, c], r)/S
    #the mean of the guided filter p
    mean_p = _boxFilter(p, r)/S
    #the correlation of (imageArray, p) corr_ip in each channel
    mean_ip = zeroes.copy()
    for c in prange(0, C):
        mean_ip[c] = _boxFilter(imageArray[:,:, c]*p, r)/S

    #covariance of (imageArray, p) cov_ip in each channel
    cov_ip = zeroes.copy()
    for c in prange(0, C):
        cov_ip[c] = mean_ip[c] - mean_i[c]*mean_p
    #variance of imageArray in each local patch (window wk), used to build the matrix sigma_k in eq.(14)
    #The variance in each window is a 3x3 symmetric matrix with variance as its values:
    #           rr, rg, rb
    #   sigma = rg, gg, gb
    #           rb, gb, bb
    var_i = _matrixOp(np.empty,(C, C, H, W))
    # Channel-on-channel variance
    # Do in a parallelizable loop
    for i in prange(0, 3):
        for j in prange(i, 3):
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

    #the filter p'  eq.(16)
    pp = ( _boxFilter(a[:,:, 0], r)*imageArray[:,:, 0]
            +_boxFilter(a[:,:, 1], r)*imageArray[:,:, 1]
            +_boxFilter(a[:,:, 2], r)*imageArray[:,:, 2]
            +_boxFilter(b, r) )/S
    if hasGPU:
        pp = cp.asnumpy(pp)
    return pp

@njit(parallel= True)
def yCumSum(a):

    out = np.empty_like(a)
    out[0, :] = a[0, :]
    for i in prange(1, a.shape[0]):
        out[i, :] = a[i, :] + out[i - 1, :]
    return out

@njit(parallel= True)
def xCumSum(a):
    out = np.empty_like(a)
    for i in prange(a.shape[0]):
        out[i, :] = np.cumsum(a[i, :])
    return out

@jit
def _boxFilter(m, r, gpu= hasGPU):
    if gpu:
        m = cp.asnumpy(m)
    out = __boxfilter__(m, r)
    if gpu:
        return cp.asarray(out)
    return out

@jit(parallel= True, fastmath= True)
def __boxfilter__(m, r):
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
    ysum = np.cumsum(m, axis=0)
    #copy the accumulated values of the windows in y
    mp[0:r+1,: ] = ysum[r:(2*r)+1,: ]
    #differences in y axis
    mp[r+1:H-r,: ] = ysum[(2*r)+1:,: ] - ysum[ :H-(2*r)-1,: ]
    mp[(-r):,: ] = np.tile(ysum[-1,: ], (r, 1)) - ysum[H-(2*r)-1:H-r-1,: ]

    #cumulative sum over x axis
    xsum = np.cumsum(mp, axis=1)
    #copy the accumulated values of the windows in x
    mp[:, 0:r+1] = xsum[:, r:(2*r)+1]
    #difference over x axis
    mp[:, r+1:W-r] = xsum[:, (2*r)+1: ] - xsum[:, :W-(2*r)-1]
    mp[:, -r: ] = np.tile(xsum[:, -1][:, None], (1, r)) - xsum[:, W-(2*r)-1:W-r-1]
    return mp

@njit(parallel=True)
def doGridLoop(fillShape:tuple, cov_ip:np.ndarray, var_i:np.ndarray, eps:float) -> np.ndarray:
    H, W, C = fillShape
    # CuPY's internal loop here is VERY slow
    # So we'll do this in Numpy
    a = np.empty((H, W, C))
    #coordGrid = createCoordinateGrid(imageArray.shape).reshape(-1).tolist()
    for i in prange(0, H):
        for j in prange(0, W):
            sigma = np.array([
                                [var_i[0, 0, i, j], var_i[0, 1, i, j], var_i[0, 2, i, j]],
                                [var_i[0, 1, i, j], var_i[1, 1, i, j], var_i[1, 2, i, j]],
                                [var_i[0, 2, i, j], var_i[1, 2, i, j], var_i[2, 2, i, j]]
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
except Exception:
    fnPool = list()

def _matrixOp(fn, *args, **kwargs):
    if not hasGPU:
        return fn(*args, **kwargs)
    if isinstance(fn, str):
        lookupFn = fn
    else:
        lookupFunction = fn.__name__
    result = 'err'
    for fnName, fnExec in fnPool:
        if fnName == lookupFunction:
            try:
                result = fnExec(*args, **kwargs)
                break
            except Exception:
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
