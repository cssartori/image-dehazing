"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
from numba import njit, prange

@njit(parallel= True, fastmath= True)
def recover(imageArray, atm, t, tmin=0.1):
    """
    Radiance recovery. According to section (4.3) and equation (16) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imageArray:    an H*W RGB hazed image
    atm:      the atmospheric light in imageArray
    t:        the transmission in imageArray
    tmin:     the minimum value that transmission can take (default=0.1)

    Return
    -----------
    The imaged recovered and dehazed, j (a H*W RGB matrix).
    """

    #the output dehazed image
    j = np.empty(imageArray.shape)

    #equation (16)
    for c in prange(0, imageArray.shape[2]):
        j[:,:, c] = ((imageArray[:,:, c]-atm[c])/np.maximum(t[:,:], tmin))+atm[c]

    return j/np.amax(j)
