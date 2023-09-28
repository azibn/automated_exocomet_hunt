#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.

import os
import multiprocessing
import sys
import traceback
import argparse
import glob
import warnings
import numpy as np
from astropy.io import fits
from analysis_tools_cython import folders_in

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

parser.add_argument("-nice", help="set niceness", dest="nice", default=8, type=int)


# Get directories from command line arguments.
args = parser.parse_args()

# set niceness
os.nice(args.nice)

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))

## Prepare multithreading.

try:
    multiprocessing.set_start_method("fork")  # default for >=3.8 is spawn
except RuntimeError:  # there might be a timeout sometimes.
    pass

m = multiprocessing.Manager()
lock = m.Lock()

# pipeline_options = {"xrp":".pkl","spoc":".fits"}


def import_file(f_path):
    print(f_path)
    hdul = fits.open(f_path)
    data = hdul[0].header
    ticid = data['TIC_ID']
    tmag = data['TMAG']
    print(ticid, tmag)



if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=args.threads)

    for path in paths:
        print("starting")

        if not os.path.isdir(path):
            if os.path.isfile(path):
                import_file(path)

                sys.exit()

        # if we are in the lowest subdirectory, perform glob this way.

        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # works for both Kepler and TESS fits files.
            fits = glob.glob(os.path.join(path, "*lc.fits"))

            pool.map(import_file, fits)

        fits = glob.glob(os.path.join(path, "**/*.fits"))

        pool.map(import_file, fits)
