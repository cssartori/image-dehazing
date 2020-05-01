"""
Fast Single Image Haze Removal Using Dark Channel Prior
Final Project of "INF01050 - Computational Photography" class, 2016, at UFRGS.

Carlo S. Sartori
"""
import argparse
import sys
import numpy

def dehazeImage(img:numpy.ndarray, outputImgFile:str,  a= None, t= None, rt= None, tmin:float= 0.1, ps:int= 15, w:float= 0.99, px:float= 1e-3, r:int= 40, eps:float= 1e-3, verbose:bool= False):
    from .AImage import AImage
    from .Dehaze import dehaze
    #tries to open the input image
    try:
        img = AImage.open(img)
        if verbose:
            print(f"Image '{img}' opened.")
    except (IOError, FileNotFoundError):
        raise FileNotFoundError(f"File '{img}' cannot be found.")
    #Dehaze the input image
    oImg = dehaze(img.array(), a, t, rt, tmin, ps, w, px, r, eps, verbose)
    from skimage import exposure
    oImg2 = exposure.adjust_gamma(oImg, gamma= 1.1)
    oImg3 = exposure.adjust_sigmoid(oImg2, gain= 5.5)
    oImg3 = (oImg3 * 255).astype(numpy.uint8)
    #save the image to file
    _ = AImage.save(oImg3, outputImgFile)
    if verbose:
        print(f"Image '{outputImgFile}' saved.")
    return img

def dehazeDirectory(directory, outputDirectory= None, extensions= ["png", "jpg", "jpeg"], a= None, t= None, rt= None, tmin:float= 0.1, ps:int= 15, w:float= 0.99, px:float= 1e-3, r:int= 40, eps:float= 1e-3, verbose:bool= False):
    import glob
    import os
    if outputDirectory is None:
        outputDirectory = directory
    fileSet = list()
    for extension in extensions:
        files = glob.glob(os.path.join(directory, "**", f"*.{extension}"), recursive= True)
        fileSet += list(files)
    fileSet = frozenset(fileSet)
    print(f"Going to convert {len(fileSet)} images")
    for i, filename in enumerate(fileSet, 1):
        filenameParts = os.path.basename(filename).split(".")
        fileBase = filenameParts[:-1]
        outFile = os.path.join(outputDirectory, ".".join(fileBase + ["dehazed", filenameParts[-1]]))
        _ = dehazeImage(filename, outFile, a, t, rt, tmin, ps, w, px, r, eps, verbose)
        print(f"Done {i} of {len(fileSet)}")


#Prepare the arguments the program shall receive
def __prepareargs__():
    parser = argparse.ArgumentParser(description='Fast Single Image Haze Removal Using Dark Channel Prior.')
    parser.add_argument('-i', nargs=1, type=str, help='input image path', required=True)
    parser.add_argument('-o', nargs=1, type=str, help='output image path', required=True)
    parser.add_argument('-a', nargs=1, type=str, help='atm. light file path (default=None)', required=False)
    parser.add_argument('-t', nargs=1, type=str, help='transmission file path (default=None)', required=False)
    parser.add_argument('-rt', nargs=1, type=str, help='raw transmission file path (default=None)', required=False)
    parser.add_argument('-tmin', nargs=1, type=float, help='minimum transmission allowed (default=0.1)', required=False)
    parser.add_argument('-ps', nargs=1, type=int, help='patch size (default=15)', required=False)
    parser.add_argument('-w', nargs=1, type=float, help='omega weight (default=0.99)', required=False)
    parser.add_argument('-px', nargs=1, type=float, help='percentage of pixels for the atm. light (default=1e-3)', required=False)
    parser.add_argument('-r', nargs=1, type=int, help='pixel radius of guided filter (default=40)', required=False)
    parser.add_argument('-eps', nargs=1, type=float, help='epsilon of guided filter(default=1e-3)', required=False)
    parser.add_argument('-m', action='store_const', help='print out messages along processing', const=True, default=False, required=False)

    return parser

#Parse the input arguments and returns a dictionary with them
def __getargs__(parser):
    args = vars(parser.parse_args())
    return args


#The main module, in case this program was called as the main program
if __name__ == '__main__':
    #receive and prepare the arguments
    parser = __prepareargs__()
    args = __getargs__(parser)

    #get required parameters
    input_img_file = args['i'][0].strip()
    output_img_file = args['o'][0].strip()

    #default values of optional parameters
    a = None
    t = None
    rt = None
    tmin=0.1
    ps=15
    w=0.99
    px=1e-3
    r=40
    eps=1e-3
    m = bool(args['m'])

    #check for optional parameters
    if args['a'] is not None:
        a = numpy.loadtxt(args['a'][0])
    if args['t'] is not None:
        t = numpy.loadtxt(args['t'][0])
    if args['rt'] is not None:
        rt = numpy.loadtxt(args['rt'][0])
    if args['tmin'] is not None:
        tmin = args['tmin'][0]
    if args['ps'] is not None:
        ps = args['ps'][0]
    if args['w'] is not None:
        w = args['w'][0]
    if args['px'] is not None:
        px = args['px'][0]
    if args['r'] is not None:
        r = args['r'][0]
    if args['eps'] is not None:
        eps = args['eps'][0]

    _ = dehazeImage(input_img_file, output_img_file, a, t, rt, tmin, ps, w, px, r, eps, m)
    print('Dehazing finished!')
