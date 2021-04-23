"""
Fast Single Image Haze Removal Using Dark Channel Prior
Original by https://github.com/cssartori

@author Philip Kahn
@date 20200501
"""

import numpy as np
from skimage.io import imread, imsave, imshow

#A class to hold an Image
class AImage(object):
    """
    Class to hold an Image.
    """
    #img_array is the matrix of pixels in img
    __img_array__ = None
    #filename is the path to the image in img
    __filename__ = None

    def __init__(self, img=None, filename=None):
        self.__filename__ = filename
        self.__img_array__ = img

    def height(self) -> int:
        """
        Get the image height
        """
        return self.__img_array__.shape[0]

    def width(self) -> int:
        """
        Get the image width
        """
        return self.__img_array__.shape[1]

    def colors(self) -> int:
        """
        Get the number of color channels
        """
        return self.__img_array__.shape[2]

    def filename(self) -> str:
        """
        Get the working file name
        """
        return self.__filename__

    def __getitem__(self, index):
        """
        Get a slice of the image
        """
        return self.__img_array__[index]

    def __setitem__(self, index, value):
        """
        Set a slice of the image
        """
        self.__img_array__[index] = value

    def array(self):
        """
        Get the image array
        """
        return self.__img_array__

    def show(self):
        """
        Show the internal image
        """
        imshow(self.__img_array__)

    @staticmethod
    def fromarray(array):
        """
        Returns a standardized image
        """
        #check if the array is float type
        if array.dtype != np.float64:
            #cast to float with values from 0 to 1
            array = np.divide(np.asfarray(array), 255.0)
        sImage = AImage(array)
        return sImage

    @staticmethod
    def open(filename) -> np.ndarray:
        """
        Get an image array from an image file
        """
        try:
            array = imread(filename)
            img = AImage.load(array, filename)
        except (IOError, PermissionError, FileNotFoundError):
            raise IOError(f"Couldn't access file {filename}")
        return img

    @staticmethod
    def load(array:np.ndarray, filename:str= None):
        """
        Get a convenient AImage from an ndarray
        """
        #check if the array is float type
        if array.dtype != np.float64:
            #cast to float with values from 0 to 1
            array = np.divide(np.asfarray(array), 255).astype(np.float64)
        return AImage(array, filename)

    @staticmethod
    def save(im, filename):
        """
        Save an image
        """
        if isinstance(im, AImage):
            imsave(filename, im.array())
            sImage = AImage(im.array(), filename)
        elif isinstance(im, np.ndarray):
            imsave(filename, im)
            sImage = AImage(im, filename)
        else:
            raise TypeError('im parameter should be either a np.ndarray or AImage.AImage')
        return sImage
