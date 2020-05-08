"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""
try:
    import DarkChannel
    import AtmLight
    import Transmission
    import Refine
    import Radiance
except ModuleNotFoundError:
    from . import DarkChannel
    from . import AtmLight
    from . import Transmission
    from . import Refine
    from . import Radiance
import numpy as np

"""
Main References
[1] Single Image Haze Removal Using Dark Channel Prior (2009)
    http://kaiminghe.com/cvpr09/index.html

[2] Guided Image Filtering (2010)
    http://kaiminghe.com/eccv10/index.html

"""
try:
    from timeit_local import timeit
except ModuleNotFoundError:
    from .timeit_local import timeit

def dehaze(imageArray, a=None, t=None, rt=None, tmin=0.1, ps=15, w=0.95, px=1e-3, r=40, eps=1e-3, m=False, returnLight= False):
    """
    Application of the dehazing algorithm, as described in section (4) of the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imageArray:    an H*W RGB hazed image
    a:        the atmospheric light RGB array of imageArray (default=None, will be calculated internally)
    t:        the transmission matrix H*W of imageArray (default=None, will be calculated internally)
    rt:       the raw transmission matrix H*W of imageArray, to be refined (default=None, will be calculated internally)
    tmin:     the minimum value the transmission can take (default=0.1)
    ps:       the patch size for dark channel estimation (default=15)
    w:        the omega weight, amount of haze to be kept (default=0.95)
    px:       the percentage of brightest pixels to be considered when estimating atmospheric light (default=1e-3, i.e. 0.1%)
    r:        the radius of the guided filter in pixels (default=40)
    eps:      the epsilon parameter for guided filter (default=1e-3)
    m:        print out messages along processing

    Return
    -----------
    The dehazed image version of imageArray, dehazed (a H*W RGB matrix).
    """
    def doNothing():
        return
    if m:
        timeDisp = print
    else:
        timeDisp = doNothing
    with timeit("\tDark channel estimated in", logFn= timeDisp):
        jdark = DarkChannel.estimate(imageArray, ps)
    #return jdark
    #if no atmospheric given
    if a == None:
        with timeit("\tAtmospheric light estimated in", logFn= timeDisp):
            a = AtmLight.estimate(imageArray, jdark)

    #if no raw transmission and complete transmission given
    if rt == None and t == None:
        with timeit("\tTransmission estimated in", logFn= timeDisp):
            rt = Transmission.estimate(imageArray, a, w)
        #threshold of raw transmission
        rt = np.maximum(rt, tmin)


    #if no complete transmission given, refine the raw using guided filter
    if t == None:
        with timeit("\tRefinement filter run in", logFn= timeDisp):
            t = Refine.guided_filter(imageArray, rt)

    #recover the scene radiance
    with timeit("\tRadiance recovery in", logFn= timeDisp):
        dehazed = Radiance.recover(imageArray, a, t, tmin)

    if returnLight:
        return dehazed, np.sum(a)
    return dehazed
