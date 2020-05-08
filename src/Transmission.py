"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
try:
    import DarkChannel
except ModuleNotFoundError:
    from . import DarkChannel
from numba import jit, prange

@jit(parallel= True, fastmath= True)
def estimate(imageArray, A, w=0.95):
    """
    Transmission estimation. According to section (4.1) equation (11) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imageArray:    an H*W RGB hazed image
    A:        the atmospheric light of imageArray
    w:        the omega weight parameter, the amount of haze to be removed (default=0.95)

    Return
    -----------
    The transmission estimated in imageArray, t (a H*W matrix).
    """
    #the normalized haze image
    nimg = np.empty(imageArray.shape)

    #calculate the normalized haze image
    for c in prange(0, imageArray.shape[2]):
        nimg[:,:, c] = imageArray[:,:, c]/A[c]

    #estimate the dark channel of the normalized haze image
    njdark = DarkChannel.estimate(nimg)

    #calculates the transmisson t
    t = 1-w*njdark+0.25

    return t
