"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
try:
    import DarkChannel
except (ModuleNotFoundError, ImportError):
    # pylint: disable= relative-beyond-top-level
    from . import DarkChannel
from numba import jit, prange

@jit(parallel= True, fastmath= True)
def estimate(imageArray:np.ndarray, lightArray:np.ndarray, w:float= 0.95) -> np.ndarray:
    """
    Transmission estimation. According to section (4.1) equation (11) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------

    imageArray: np.ndarray
        an H*W RGB hazed image

    lightArray: np.ndarray
        the atmospheric light of imageArray

    w: float (default=0.95)
        the omega weight parameter, the amount of haze to be removed

    Return
    -----------
    The transmission estimated in imageArray, t (a H*W matrix).
    """
    #the normalized haze image
    nimg = np.empty(imageArray.shape)
    #calculate the normalized haze image
    for c in prange(0, imageArray.shape[2]): #pylint: disable= not-an-iterable
        nimg[:,:, c] = imageArray[:,:, c]/lightArray[c]
    #estimate the dark channel of the normalized haze image
    njdark = DarkChannel.estimate(nimg)
    #calculates the transmission t
    t = 1 - w * njdark+0.25
    return t
