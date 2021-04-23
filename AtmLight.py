"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
from numba import jit

@jit
def estimate(imageArray:np.ndarray, jDark:np.ndarray, px:float= 1e-3) -> np.ndarray:
    """
    Automatic atmospheric light estimation. According to section (4.4) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------

    imageArray: np.ndarray
        an H*W RGB hazed image

    jDark: np.ndarray
        the dark channel of imageArray

    px: float (default=1e-3, i.e. 0.1%)
        the percentage of brighter pixels to be considered

    Return
    -----------
    The atmosphere light estimated in imageArray, A (a RGB vector).
    """
    #reshape both matrix to get it in 1-D array shape
    imgAVec = np.resize(imageArray, (imageArray.shape[0]*imageArray.shape[1], imageArray.shape[2]))
    jDarkVec = np.reshape(jDark, jDark.size)
    #the number of pixels to be considered
    numPixels = np.int(jDark.size * px)
    #index sort the jDark channel in descending order
    isJD = np.argsort(-jDarkVec)
    arraySum = np.array([0.0, 0.0, 0.0])
    for i in range(0, numPixels):
        arraySum[:] += imgAVec[isJD[i],:]
    arr:np.ndarray = np.array([0.0, 0.0, 0.0])
    arr[:] = arraySum[:]/numPixels
    #returns the calculated airlight arr
    return arr
