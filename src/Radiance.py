"""
Fast Single Image Haze Removal Using Dark Channel Prior
Final Project of "INF01050 - Computational Photography" class, 2016, at UFRGS.

Carlo S. Sartori
"""

import numpy
from numba import njit

@njit
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
    j = numpy.zeros(imageArray.shape)

    #equation (16)
    for c in range(0, imageArray.shape[2]):
        j[:,:, c] = ((imageArray[:,:, c]-atm[c])/numpy.maximum(t[:,:], tmin))+atm[c]

    return j/numpy.amax(j)
