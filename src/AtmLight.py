"""
Fast Single Image Haze Removal Using Dark Channel Prior
Final Project of "INF01050 - Computational Photography" class, 2016, at UFRGS.

Carlo S. Sartori
"""

import numpy
from numba import jit

@jit
def estimate(imageArray, jdark, px=1e-3):
    """
    Automatic atmospheric light estimation. According to section (4.4) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imageArray:    an H*W RGB hazed image
    jdark:    the dark channel of imageArray
    px:       the percentage of brigther pixels to be considered (default=1e-3, i.e. 0.1%)

    Return
    -----------
    The atmosphere light estimated in imageArray, A (a RGB vector).
    """

    #reshape both matrix to get it in 1-D array shape
    imgavec = numpy.resize(imageArray, (imageArray.shape[0]*imageArray.shape[1], imageArray.shape[2]))
    jdarkvec = numpy.reshape(jdark, jdark.size)

    #the number of pixels to be considered
    numpx = numpy.int(jdark.size * px)

    #index sort the jdark channel in descending order
    isjd = numpy.argsort(-jdarkvec)

    asum = numpy.array([0.0, 0.0, 0.0])
    for i in range(0, numpx):
        asum[:] += imgavec[isjd[i],:]

    A = numpy.array([0.0, 0.0, 0.0])
    A[:] = asum[:]/numpx

    #returns the calculated airlight A
    return A
