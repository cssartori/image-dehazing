"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
from numba import jit, njit, prange

@jit
def estimate(imageArray, ps=15):
    """
    Dark Channel estimation. According to equation (5) in the reference paper
    http://research.microsoft.com/en-us/um/people/kahe/cvpr09/

    Parameters
    -----------
    imageArray:   an H*W RGB  hazed image
    ps:      the patch size (a patch P(x) has size (ps x ps) and is centered at pixel x)

    Return
    -----------
    The dark channel estimated in imageArray, jDark (a matrix H*W).
    """
    offset = ps // 2
    #Padding of the image to have windows of ps x ps size centered at each image pixel
    imPad = np.pad(imageArray, [(offset, offset), (offset, offset), (0, 0)], 'edge')

    return getJDark(offset, np.empty(imageArray.shape[:2]), imPad)

@njit(parallel=True, cache= True)
def getJDark(offset:int, jDark:tuple, paddedImage:np.ndarray) -> np.ndarray:
    """
    Get the dark channel
    """
    #jDark is the Dark channel to be found
    for i in prange(offset, (jDark.shape[0]+offset)): #pylint: disable= not-an-iterable
        for j in prange(offset, (jDark.shape[1]+offset)): #pylint: disable= not-an-iterable
            #creates the patch P(x) of size ps x ps centered at x
            patch = paddedImage[i-offset:i+1+offset, j-offset:j+1+offset]
            #selects the minimum value in this patch and set as the dark channel of pixel x
            jDark[i-offset, j-offset] = patch.min()

    return jDark
