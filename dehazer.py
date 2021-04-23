"""
Fast Single Image Haze Removal Using Dark Channel Prior

Original by https://github.com/cssartori

@author Philip Kahn
@date 20210423
"""
import argparse
import os
import datetime as dt
import glob
import textwrap
import gc
import warnings
import csv
import time
import pandas as pd
from skimage import exposure, io
import numpy as np
from typing import Union, Optional
from PIL import Image
from imagehash import phash, colorhash
try:
    from AImage import AImage
    from Dehaze import dehaze
except (ModuleNotFoundError, ImportError):
    # pylint: disable= relative-beyond-top-level
    from .AImage import AImage
    from .Dehaze import dehaze

#pylint: disable= dangerous-default-value

def dehazeImage(img:Union[str, np.ndarray], outputImgFile:Optional[str]= None,  a:Optional[np.ndarray]= None, t:Optional[np.ndarray]= None, rt:Optional[np.ndarray]= None, tmin:float= 0.1, ps:int= 15, w:float= 0.99, px:float= 1e-3, r:int= 40, eps:float= 1e-3, verbose:bool= False, report:bool= False, checkSections:bool= False) -> np.ndarray: #pylint:disable= redefined-outer-name
    """
    Dehaze an image

    Parameters
    =======================

    img: str, np.ndarray
        A file path or numpy array corresponding to an image

    outputImgFile: str (default= None)
        When not none, the file to save the output image to

    a: np.ndarray (default= None)
        Atmospheric light array (computed if None)

    t: np.ndarray (default= None)
        Transmission array (computed if None)

    rt: np.ndarray (default= None)
        Raw transmission array (computed if None)

    tmin: float (default= 0.1)
        Minimum transmission allowed

    ps: int (default= 15)
        Patch size

    w: float (default= 0.99)
        Omega weight

    px: float (default= 1e-3)
        Percentage of pixels for the atmospheric light

    r: int (default= 40)
        Pixel radius for the guided filter

    eps: float (default= 1e-3)
        Epsilon of the guided filter

    verbose: bool (default= False)

    report: bool (default= False)
        If True, returns tuple (img:np.ndarray, stats:list-of-dicts)
        with stats containing statistics for the image and optionally
        sections

    checkSections: bool (default= False)
        Also run stats on horizontal slices of the image

    Returns
    ================================

    np.ndarray : dehazed image

    If report is True, returns (np.ndarray, list)
    """
    startTime = dt.datetime.now()
    # Image loading
    saveImage = isinstance(outputImgFile, str)
    if not saveImage:
        outputImgFile = None
    if saveImage and not os.path.exists(os.path.dirname(outputImgFile)):
        raise ValueError(f"Output directory `{os.path.dirname(outputImgFile)}` does not exist")
    if isinstance(img, str):
        # tries to open the input image
        try:
            inFilename = os.path.basename(img)
            img = AImage.open(img)
            if verbose:
                print(f"Image `{inFilename}` opened.")
        except PermissionError:
            raise PermissionError(f"Permission denied reading `{os.path.abspath(img)}`")
        except (IOError, FileNotFoundError):
            raise FileNotFoundError(f"File `{os.path.abspath(img)}`` cannot be found.")
    elif isinstance(img, np.ndarray):
        inFilename = None
        img = AImage.load(img)
        if verbose:
            print("Loaded image from ndarray")
    else:
        raise TypeError("Invalid image type")
    # Dehaze the input image
    oImgO, totalLight = dehaze(img.array(), a, t, rt, tmin, ps, w, px, r, eps, verbose, returnLight= True)
    # Fix the pixel ranges that are returned from the dehazer, if need be
    if np.min(oImgO) <= -0.1:
        # Some images have insane range, eg, -3
        oImgR = (oImgO - np.clip(np.min(oImgO), -255, 0))
        oImg = oImgR / np.max(oImgR)
    else:
        oImg = oImgO.copy()
    oImg = np.clip(exposure.rescale_intensity(oImg, in_range= (np.min(img.array()), np.max(img.array()))), 0, 255)
    # Compare to original, if sufficiently dehazed do exposure correction
    originalHash = phash(Image.fromarray((255 * img.array()).astype(np.uint8)))
    originalHashC = colorhash(Image.fromarray((255 * img.array()).astype(np.uint8)))
    newHash = phash(Image.fromarray((255 * oImg).astype(np.uint8)))
    newHashC = colorhash(Image.fromarray((255 * oImg).astype(np.uint8)))
    percepHashDiff = abs(newHash - originalHash)
    colorHashDiff = abs(newHashC - originalHashC)
    # Check the differences between input and output
    if verbose:
        # aerial: 10
        # RED: 4
        print(inFilename)
        print("Perceptual hash:", percepHashDiff)
        print("Color hash:", colorHashDiff)
    if percepHashDiff >= 20 and colorHashDiff >= 5:
        if outputImgFile is None:
            warnings.warn("There may be an issue with the dehazed image")
        else:
            warnings.warn(f"There may be an issue with the dehazed image `{outputImgFile}`")
    # Generate a final exposure-corrected image
    if percepHashDiff > 1 and (colorHashDiff > 2 or percepHashDiff >= 4 or totalLight >= 2.75):
        needed = True
        try:
            gamma = np.clip(1.1, 1, 1.2) # Brightness
            gain = np.clip(5.4, 5, 5.7) # Contrast #pylint: disable= unused-variable
            try:
                oImg2 = exposure.adjust_gamma(np.clip(oImg, 0, 255), gamma= gamma)
            except ValueError:
                oImg2 = np.clip(oImg.copy(), 0, 255)
            # oImg3 = exposure.adjust_sigmoid(oImg2, gain= gain)
            oImg3 = (oImg2 * 255).astype(np.uint8)
        except ValueError as e:
            print(f"Did not need to dehaze; nonsensical result for hash difference {percepHashDiff} & {colorHashDiff}: {e}")
            oImg3 = (255 * img.array()).astype(np.uint8)
    else:
        needed = False
        print(f"Dehazing made no or trivial perceptual changes to the data in `{inFilename}` (hash difference {percepHashDiff} & {colorHashDiff})")
        oImg3 = (255 * img.array()).astype(np.uint8)
    #save the image to file
    if saveImage:
        _ = AImage.save(oImg3, outputImgFile)
        if verbose:
            print(f"Image '{outputImgFile}' saved.")
    otherStats = list()
    if checkSections:
        # Review horizontal sections of a photo
        # The goal of this is for the case where you only
        # care about haze in a subsection of an image and,
        # therefore, don't want to manipulate the image
        # unless haze exists in this "bad" location
        otherStatsDict = {}
        h, w = img.array().shape[:2]
        refImg = (255 * img.array()).astype(np.uint8)
        sections = {
            "topQuarter": ((0, h//4), (0, w)),
            "middleQuarter": ((h//4, h//2), (0, w)),
            "bottomHalf": ((h//2, h), (0, w)),
        }
        for corner, slices in sections.items():
            if verbose:
                print(f"\tDehazing corner {corner}...")
            if outputImgFile is not None:
                oParts = outputImgFile.split(".")
                ext = oParts.pop()
                oParts.append(f"section_{corner}")
                oParts.append(ext)
                quadOut = ".".join(oParts)
            else:
                quadOut = None
            h0, h1 = slices[0]
            w0, w1 = slices[1]
            sectionOHash = phash(Image.fromarray(refImg[h0:h1, w0:w1]))
            sectionNewHash = phash(Image.fromarray(oImg3[h0:h1, w0:w1]))
            sectionCOHash = colorhash(Image.fromarray(refImg[h0:h1, w0:w1]))
            sectionCNewHash = colorhash(Image.fromarray(oImg3[h0:h1, w0:w1]))
            sHashDiff = abs(sectionNewHash - sectionOHash)
            sHashDiff2 = abs(sectionCNewHash - sectionCOHash)
            needed = sHashDiff > 1 and (sHashDiff2 > 2 or sHashDiff >= 4) # or totalLight >= 2.75)
            qs = {
                "perceptualHashDifference": sHashDiff,
                "colorHashDifference": sHashDiff2,
                "totalLight": "",
                "needed": needed,
                "needMeasure": {
                    "perceptualBasic": sHashDiff > 1,
                    "perceptualStrong": sHashDiff >= 4,
                    "colorShift": sHashDiff2 > 2,
                    "atmosphericLight": False
                },
                "runTimeSeconds": "-",
                "style": f"section_{corner}",
                "topHalfBad": None,
                "topQuarterBad": None,
                "wholeImageBad": None,
                "wholeImageGood": None
            }
            otherStatsDict[corner] = qs
            if quadOut is not None:
                io.imsave(quadOut, oImg3[h0:h1, w0:w1])
                print(f"\twrote subimage `{quadOut}`")
    if report:
        stats = {
            "perceptualHashDifference": percepHashDiff,
            "colorHashDifference": colorHashDiff,
            "totalLight": totalLight,
            "needed": needed,
            "needMeasure": {
                "perceptualBasic": percepHashDiff > 1,
                "perceptualStrong": percepHashDiff >= 4,
                "colorShift": colorHashDiff > 2,
                "atmosphericLight": totalLight >= 2.75
            },
            "runTimeSeconds": np.around((dt.datetime.now() - startTime).total_seconds(), 3),
            "style": "fullPhoto"
        }
        if checkSections:
            # if there's haze in the bottom half, the whole frame is bad.  If there's frame in the next quarter up, the top half is bad.  if there's haze in the top quarter, the top quarter is bad.  otherwise the whole frame is good.
            stats["topQuarterBad"] = otherStatsDict["topQuarter"]["needed"]
            stats["topHalfBad"] = otherStatsDict["middleQuarter"]["needed"] or stats["topQuarterBad"]
            stats["wholeImageBad"] = otherStatsDict["bottomHalf"]["needed"] or stats["needed"]
            stats["wholeImageGood"] = not (stats["topQuarterBad"] or stats["topHalfBad"] or stats["wholeImageBad"])
            # Aggregate it into a list
            for _, statSet in otherStatsDict.items():
                otherStats += [statSet]
        return oImg3, [stats] + otherStats
    return oImg3

def dehazeDirectory(directory:str, outputDirectory:Optional[str]= None, extensions= ["png", "jpg", "jpeg"], verbose:bool= False, recursiveSearch:bool= True, report:bool= False, checkSections:bool= False, **kwargs) -> pd.DataFrame:
    """
    Dehaze a directory of images

    Parameters
    =======================

    directory: str

    outputDirectory: str (default= None)
        Where to save the outputs. If None, they'll be saved in the read
        directory. If a str, but the directory does not exist, an attempt
        will be made to create it.

    extensions: list (default= ["png", "jpg", "jpeg"])
        List of extensions of images to check.

    verbose: bool (default= False)

    recursiveSearch: bool (default= True)
        Deeply and recursively search directories

    report: bool (default= False)
        If True, returns a Pandas DataFrame of the report.

    checkSections: bool (default= False)

    Also accepts dehazing kwarg parameters to pass to dehazeImage().

    Returns
    ================================

    If report is True, returns a Pandas DataFrame of the report.
    Otherwise, returns None
    """
    if outputDirectory is None:
        outputDirectory = directory
    else:
        if not os.path.isdir(outputDirectory):
            os.makedirs(outputDirectory)
    fileSet = list()
    extensions = frozenset(extensions)
    for extension in extensions:
        if recursiveSearch:
            files = glob.glob(os.path.join(directory, "**", f"*.{extension}"), recursive= True)
        else:
            files = glob.glob(os.path.join(directory, f"*.{extension}"))
        fileSet += list(files)
    fileSet = frozenset(fileSet)
    reportFile = os.path.join(outputDirectory, "report.csv")
    if report and os.path.exists(reportFile):
        os.unlink(reportFile)
    print(f"Going to convert {len(fileSet)} images")
    def badness(thruthy):
        if thruthy is None:
            return ""
        return "Bad" if thruthy else "OK"
    def goodness(thruthy):
        if thruthy is None:
            return ""
        return "Good" if thruthy else "Bad"
    for iCount, filename in enumerate(fileSet, 1):
        filenameParts = os.path.basename(filename).split(".")
        fileBase = filenameParts[:-1]
        outFile = os.path.join(outputDirectory, ".".join(fileBase + ["dehazed", filenameParts[-1]]))
        results = dehazeImage(filename, outFile, verbose= verbose, report= report, checkSections= checkSections, **kwargs)
        if report:
            # Handle output reporting
            loggedReport= False
            if verbose:
                print("About to report")
            statsSet= results[1]
            # statsSet will always be length 1 for
            # checkSections= False, but we always return
            # a list for consistent handling
            for stats in statsSet:
                # Create a score for "needed"
                howMuchNeeded = 0
                if stats["needMeasure"]["perceptualBasic"]:
                    for _, boolVal in stats["needMeasure"].items():
                        howMuchNeeded += int(boolVal)
                writeHeader = not os.path.exists(reportFile)
                if verbose and not loggedReport:
                    print(f"Going to write to `{reportFile}` (with headers? {writeHeader})")
                    loggedReport = True
                commonPrefix = os.path.commonprefix([os.path.abspath(os.path.normpath(directory)), os.path.abspath(os.path.normpath(os.path.dirname(filename)))])
                fileDir = os.path.abspath(os.path.normpath(os.path.dirname(filename))).replace(commonPrefix, "")
                if len(fileDir) == 0:
                    fileDir = "./"
                row = {
                    "Search Directory": os.path.abspath(os.path.normpath(directory)),
                    "File Directory": fileDir,
                    "Filename": os.path.basename(filename),
                    "Processing Time (s)": stats["runTimeSeconds"],
                    "Style": stats["style"],
                    "Hazy?": "Hazy" if stats["needed"] else "Clear",
                    "Perceptual Hash Difference": stats["perceptualHashDifference"],
                    "Color Hash Difference": stats["colorHashDifference"],
                    "Total Light": stats["totalLight"],
                    "Needed Details": stats["needMeasure"],
                    "Top Quarter Status": badness(stats["topQuarterBad"]),
                    "Top Half Status": badness(stats["topHalfBad"]),
                    "Bottom Half ('Whole Image') status": badness(stats["wholeImageBad"]),
                    "Whole Image": goodness(stats["wholeImageGood"]),
                    "isBadTopQuarter": stats["topQuarterBad"] if stats["topQuarterBad"] is not None else "",
                    "isBadTopHalf": stats["topHalfBad"] if stats["topHalfBad"] is not None else "",
                    "isBadBottomHalfOrWholeImage": stats["wholeImageBad"] if stats["wholeImageBad"] is not None else "",
                    "isValidImage": stats["wholeImageGood"] if stats["wholeImageGood"] is not None else "",
                    "dehazingNeeded": stats["needed"],

                }
                ############
                # We're writing this to report as we go.
                # Since the most common viewer is Excel, and
                # Microsoft grabs files and never lets go, we
                # don't want to crap out as soon as someone views
                # a file while we're trying to write. So, we'll
                # attempt an append for 10 minutes before giving
                # up the ghost, and warn the user lots in the
                # terminal.
                ############
                hasWritten = False
                tries = 0
                tryLimit = 60
                waitDelaySeconds = 10
                while not hasWritten:
                    try:
                        with open(reportFile, "a", newline= "") as fh:
                            writer = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator= "\n")
                            if writeHeader:
                                writer.writerow([col for col, value in row.items()])
                            writer.writerow([str(value) for col, value in row.items()])
                        hasWritten = True
                    except (IOError, PermissionError):
                        if tries > tryLimit:
                            raise
                        print(f"Could not open `{reportFile}` for writing; if you use Excel, please close the file there. The process will abort in {(tryLimit - tries + 1) * waitDelaySeconds} seconds...")
                        time.sleep(waitDelaySeconds)
                        tries += 1
        print(f"Done {iCount} of {len(fileSet)}")
    if report:
        return pd.read_csv(reportFile)


def dehazeFolderOfDirectories(parentDirectory:str, outputDir:str= "dehazedFrames", recursiveSearch:bool= False, report:bool= True, checkSections:bool= False, **kwargs):
    """
    Convenience wrapper for dehazeDirectorySet for all folders
    in a parent directory
    """
    if not parentDirectory.endswith("*"):
        parentDirectory = os.path.join(parentDirectory, "*")
    return dehazeDirectorySet([x for x in glob.glob(os.path.join(parentDirectory, "*"))], outputDir, recursiveSearch, report, checkSections, **kwargs)

def dehazeDirectorySet(directorySet:Union[list, tuple, set, frozenset], outputDir:str= "dehazedFrames", recursiveSearch:bool= False, report:bool= True, checkSections:bool= False, **kwargs):
    """
    Run dehazeDirectory on a set of directories.

    Does some changes of defaults appropriate to directory
    searches, and handles reporting.
    """
    parentDirectory = os.path.commonprefix(directorySet)
    if not os.path.exists(parentDirectory):
        # In case there's partial filename overlap
        parentDirectory = os.path.dirname(parentDirectory)
    metaReport = os.path.join(parentDirectory, "metaReport.csv")
    if os.path.exists(metaReport):
        os.unlink(metaReport)
    notice = f"""\
    ========================================================================
    *** Notice: The total report will be dumped to `{parentDirectory}` ***
    Per-directory reports will be updated on a per-file basis in their respective output subdirectories (`**/{outputDir}`).
    The total report will be updated upon each directory's completion.
    The total report output file will be `{os.path.abspath(metaReport)}`
    ========================================================================\n\n
    """
    if report:
        print(textwrap.dedent(notice))
    for folder in directorySet:
        passOutput = os.path.join(folder, outputDir)
        print(f"Starting folder {folder}...")
        reportOutO = dehazeDirectory(folder, passOutput, recursiveSearch= recursiveSearch, report= report, checkSections= checkSections, **kwargs)
        if report:
            ############
            # We're writing this to report as we go.
            # Since the most common viewer is Excel, and
            # Microsoft grabs files and never lets go, we
            # don't want to crap out as soon as someone views
            # a file while we're trying to write. So, we'll
            # attempt an append for 10 minutes before giving
            # up the ghost, and warn the user lots in the
            # terminal.
            ############
            hasWritten = False
            tries = 0
            tryLimit = 60
            waitDelaySeconds = 10
            while not hasWritten:
                try:
                    if os.path.exists(metaReport):
                        baseReport = pd.read_csv(metaReport)
                        reportOut = pd.concat([baseReport, reportOutO], ignore_index= True)
                    else:
                        reportOut = reportOutO.copy()
                    reportOut.to_csv(metaReport, quoting= csv.QUOTE_ALL, line_terminator= "\n", index= False)
                    hasWritten = True
                except (IOError, PermissionError):
                    if tries > tryLimit:
                        raise
                    print(f"Could not open `{metaReport}`; if you use Excel, please close the file there. The process will abort in {(tryLimit - tries + 1) * waitDelaySeconds} seconds...")
                    time.sleep(waitDelaySeconds)
                    tries += 1
        print(f"\tDone with folder {folder}")
        gc.collect()

# Prepare the arguments for when this is called on the command line
def __prepareargs__():
    parser = argparse.ArgumentParser(description='Fast Single Image Haze Removal Using Dark Channel Prior.') #pylint: disable= redefined-outer-name
    parser.add_argument('-i', nargs=1, type=str, help='input image path', required=True)
    parser.add_argument('-o', nargs=1, type=str, help='output image path', required=True)
    parser.add_argument('-a', nargs=1, type=str, help='atm. light file path (default=None)', required=False)
    parser.add_argument('-c', nargs=1, type=int, help='timer loop', required=False)
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
def __getargs__(parser): #pylint: disable= redefined-outer-name
    args = vars(parser.parse_args()) #pylint: disable= redefined-outer-name
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
    timer = False
    ps=15
    w=0.99
    px=1e-3
    r=40
    eps=1e-3
    m = bool(args['m'])

    #check for optional parameters
    if args['a'] is not None:
        a = np.loadtxt(args['a'][0])
    if args['t'] is not None:
        t = np.loadtxt(args['t'][0])
    if args['rt'] is not None:
        rt = np.loadtxt(args['rt'][0])
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
    if args["c"] is not None:
        timer = bool(args["c"][0])

    if not timer:
        _ = dehazeImage(input_img_file, output_img_file, a= a, t= t, rt= rt, tmin= tmin, ps= ps, w= w, px= px, r= r, eps= eps, verbose= m)
    else:
        for i in range(10):
            print(f"Iteration {i}")
            _ = dehazeImage(input_img_file, output_img_file, a= a, t= t, rt= rt, tmin= tmin, ps= ps, w= w, px= px, r= r, eps= eps, verbose= m)
    print('Dehazing finished!')
