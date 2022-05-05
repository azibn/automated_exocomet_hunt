#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os

# import pandas as pd
os.nice(8)
os.environ["OMP_NUM_THREADS"] = "1"
from analysis_tools_cython import *
import multiprocessing
import sys
import traceback
import argparse
import tqdm
import data
import glob
import loaders
import warnings

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

parser.add_argument("-o", default=f"output.txt", dest="of", help="output file")
parser.add_argument(
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)
parser.add_argument(
    "-f",
    help='select flux. "PDCSAP_FLUX is default. XRP lightcurve options are "corrected flux", "PCA flux" or "raw flux"',
    dest="f",
    default="PDCSAP_FLUX",
)
parser.add_argument(
    "-c",
    help="select sigma clipping threshold for XRP lightcurves.",
    dest="c",
    default=3,
    type=int,
)
parser.add_argument("-ls", help="Lomb-Scargle power", default=0.08, dest="ls")
parser.add_argument("-p", help="enable plotting", action="store_true")

# Get directories from command line arguments.
args = parser.parse_args()

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))

## Prepare multithreading.
multiprocessing.set_start_method("fork")  # default for >=3.8 is spawn
m = multiprocessing.Manager()
lock = m.Lock()


def mission_lightcurves(f_path):
    try:
        f = os.path.basename(f_path)
        print(f)
        table = import_lightcurve(f_path, flux=args.f, drop_bad_points=args.q)
        result_str = processing(table, f_path, make_plots=args.p, power=args.ls)
        try:
            os.makedirs("output_log")  # make directory plot if it doesn't exist
        except FileExistsError:
            pass
        lock.acquire()
        with open(os.path.join("output_log/", args.of), "a") as out_file:
            out_file.write(result_str + "\n")
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()


def xrp_lightcurves(f_path):
    try:
        f = os.path.basename(f_path)
        print(f_path)
        table = import_XRPlightcurve(f_path, sector=sector_test, clip=args.c)[0]
        table = table["time", args.f, "quality"]
        result_str = processing(table, f_path, make_plots=args.p, power=args.ls)
        lock.acquire()
        try:
            os.makedirs("output_log_xrp")  # make directory plot if it doesn't exist
        except FileExistsError:
            pass
        with open(os.path.join("output_log_xrp/", args.of), "a") as out_file:
            out_file.write(result_str + "\n")
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()


def single_file(f_path):
    try:
        f = os.path.basename(f_path)
        print(f_path)
        if (os.path.split(args.fits_file[0])[1].startswith("kplr")) or (
            os.path.split(args.fits_file[0])[1].startswith("hlsp_tess")
            and os.path.split(args.fits_file[0])[1].endswith("fits")
        ):  # or os.path.split(args.fits_file[0])[1].startswith("tess") and os.path.split(args.fits_file[0])[1].endswith("fits")):
            table = import_lightcurve(args.fits_file[0])
            t, flux, quality, real = clean_data(table)
        else:
            table = import_XRPlightcurve(f_path, sector=sector_test, clip=args.c)[0]
            table = table["time", args.f, "quality"]
        result_str = processing(table, f_path, make_plots=args.p)
        # lock.acquire()
        # with open(args.of,'a') as out_file:
        #    out_file.write(result_str+'\n')
        # lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":

    sector_test = int(input("Sector? "))  # args.path[0].split("_")[
    # -2
    # ]   # this is the case for XRP lightcurves... This is not required for mission lightcurves so it is ok to not consider them.
    pool = multiprocessing.Pool(processes=args.threads)

    for path in paths:
        if not os.path.isdir(path):
            # result_str = single_file(path)

            print(path, "not a directory, skipping.", file=sys.stderr)
            continue

        # if we are in the lowest subdirectory, perform glob this way.
        if not list(folders_in(path)):
            print("this is the lowest subdirectory")

            # this should work for both Kepler and TESS fits files.
            fits_files = glob.glob(os.path.join(path, "*lc.fits"))
            pkl_files = glob.glob(os.path.join(path, "*.pkl"))

            pool.map(mission_lightcurves, fits_files)
            pool.map(xrp_lightcurves, pkl_files)

        else:
            print("globbing subdirectories")

            # Start at Sector directory, glob goes through `target/000x/000x/xxxx/**/*lc.fits`
            fits_files = glob.glob(os.path.join(path, "target/**/**/**/**/*lc.fits"))

            # These are test SPOC files that I have in my home CSC directory
            # test_fits = glob.glob(os.path.join(path,'**/**/*lc.fits'))

            # Starts at sector directory. globs files in one subdirectory level below
            pkl_files = glob.glob(os.path.join(path, "**/*.pkl"))

            pool.map(xrp_lightcurves, pkl_files)
            pool.map(mission_lightcurves, fits_files)
