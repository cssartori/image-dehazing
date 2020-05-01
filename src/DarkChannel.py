"""
Fast Single Image Haze Removal Using Dark Channel Prior
Final Project of "INF01050 - Computational Photography" class, 2016, at UFRGS.

Carlo S. Sartori
"""

import numpy
from numba import jit

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
    The dark channel estimated in imageArray, jdark (a matrix H*W).
    """

    #Padding of the image to have windows of ps x ps size centered at each image pixel
    impad = numpy.pad(imageArray, [(ps//2, ps//2), (ps//2, ps//2), (0, 0)], 'edge')

    #Jdark is the Dark channel to be found
    jdark = numpy.zeros((imageArray.shape[0], imageArray.shape[1]))

    for i in range(ps//2, (imageArray.shape[0]+ps//2)):
        for j in range(ps//2, (imageArray.shape[1]+ps//2)):
            #creates the patch P(x) of size ps x ps centered at x
            patch = impad[i-ps//2:i+1+ps//2, j-ps//2:j+1+ps//2]
            #selects the minimum value in this patch and set as the dark channel of pixel x
            jdark[i-ps//2, j-ps//2] = patch.min()

    return jdark
