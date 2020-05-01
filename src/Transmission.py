"""
Fast Single Image Haze Removal Using Dark Channel Prior
Final Project of "INF01050 - Computational Photography" class, 2016, at UFRGS.

Carlo S. Sartori
"""

import numpy
from . import DarkChannel
from numba import jit

@jit
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
    nimg = numpy.zeros(imageArray.shape)

    #calculate the normalized haze image
    for c in range(0, imageArray.shape[2]):
        nimg[:,:, c] = imageArray[:,:, c]/A[c]

    #estimate the dark channel of the normalized haze image
    njdark = DarkChannel.estimate(nimg)

    #calculates the transmisson t
    t = 1-w*njdark+0.25

    return t
