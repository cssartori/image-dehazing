#!python3
"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

Main References
[1] Single Image Haze Removal Using Dark Channel Prior (2009)
    http://kaiminghe.com/cvpr09/index.html

[2] Guided Image Filtering (2010)
    http://kaiminghe.com/eccv10/index.html

@author Philip Kahn
@date 20200501
"""

import numpy as np
from typing import Optional, Tuple, Union
try:
    import DarkChannel
    import AtmLight
    import Transmission
    import Refine
    import Radiance
    from timeit_local import timeit
except (ModuleNotFoundError, ImportError):
    # pylint: disable= relative-beyond-top-level
    from . import DarkChannel
    from . import AtmLight
    from . import Transmission
    from . import Refine
    from . import Radiance
    from .timeit_local import timeit



def dehaze(imageArray:np.ndarray, a:Optional[np.ndarray]= None, t:Optional[np.ndarray]= None, rt:Optional[np.ndarray]= None, tmin:float= 0.1, ps:int= 15, w:float= 0.95, px:float= 1e-3, r:int= 40, eps:float= 1e-3, m:bool= False, returnLight:bool= False) -> Union[np.ndarray, Tuple[np.ndarray, float]]: #pylint: disable= unused-argument
    """
    Application of the dehazing algorithm, as described in section (4) of the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------

    imageArray: np.ndarray
        an H*W RGB hazed image

    a: np.ndarray (default= None, will be calculated internally)
        the atmospheric light RGB array of imageArray

    t: np.ndarray (default= None, will be calculated internally)
        the transmission matrix H*W of imageArray

    rt: np.ndarray (default= None, will be calculated internally)
        the raw transmission matrix H*W of imageArray, to be refined

    tmin: float (default=0.1)
        the minimum value the transmission can take

    ps: int (default=15)
        the patch size for dark channel estimation

    w: float (default=0.95)
        the omega weight, amount of haze to be kept

    px: float (default=1e-3, i.e. 0.1%)
        the percentage of brightest pixels to be considered when estimating atmospheric light

    r: int (default=40)
        the radius of the guided filter in pixels

    eps: float (default=1e-3)
        the epsilon parameter for guided filter

    m:
        print out messages along processing

    returnLight: bool (default= False)
        Return the light sum along with the array

    Return
    -----------
    The dehazed image version of imageArray, dehazed (a H*W RGB matrix).
    """
    def doNothing(*args, **kwargs): #pylint: disable= unused-argument
        return
    if m:
        timeDisp = print
    else:
        timeDisp = doNothing
    with timeit("\tDark channel estimated in", logFn= timeDisp):
        jDark = DarkChannel.estimate(imageArray, ps)
    #return jDark
    #if no atmospheric given
    if a is None:
        with timeit("\tAtmospheric light estimated in", logFn= timeDisp):
            a = AtmLight.estimate(imageArray, jDark)
    #if no raw transmission and complete transmission given
    if rt is None and t is None:
        with timeit("\tTransmission estimated in", logFn= timeDisp):
            rt = Transmission.estimate(imageArray, a, w)
        #threshold of raw transmission
        rt = np.maximum(rt, tmin)
    #if no complete transmission given, refine the raw using guided filter
    if t is None:
        with timeit("\tRefinement filter run in", logFn= timeDisp):
            t = Refine.guided_filter(imageArray, rt)
    #recover the scene radiance
    with timeit("\tRadiance recovery in", logFn= timeDisp):
        dehazed = Radiance.recover(imageArray, a, t, tmin)
    if returnLight:
        return dehazed, np.sum(a)
    return dehazed
