#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os

# os.nice(8)
os.environ["OMP_NUM_THREADS"] = "1"
from analysis_tools_cython import *
import multiprocessing
import sys
import time
import traceback
import argparse
import tqdm
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
    "-f",
    help='select flux. "corrected flux is default. XRP lightcurve options are "corrected flux", "PCA flux" or "raw flux". For other lightcurves, options are "PDCSAP_FLUX"',
    dest="f",
    default="corrected flux",
)
parser.add_argument(
    "-c",
    help="set sigma clipping threshold for MAD cuts.",
    dest="c",
    default=3,
    type=int,
)

parser.add_argument("-step", help="enable twostep. used in conjuction with -m fourier", dest="step", action="store_true")
parser.add_argument("-p", help="enable plotting", action="store_true")
parser.add_argument(
    "-metadata",
    help="metadata file",
    dest="metadata",
    action="store_true",
)

parser.add_argument("-n", help="set niceness", dest="n", default=8, type=int)
parser.add_argument(
    "-m",
    help="set smoothing method. Default is None.",
    dest="m",
    default=None,
    type=str,
)

# Get directories from command line arguments.
args = parser.parse_args()

# set niceness
os.nice(args.n)

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

#pipeline_options = {"xrp":".pkl","spoc":".fits"}

def run_lc(f_path, get_metadata=args.metadata):
    try:
        f = os.path.basename(f_path)
        print(f_path)
        if f_path.endswith(".pkl"):
            table, lc_info = import_XRPlightcurve(
                f_path,
                sector=sector,
                clip=args.c,
            )
            table = table["time", args.f, "quality"]
        else:
            table, lc_info = import_lightcurve(f_path, flux=args.f)
        lc_info = " ".join([str(i) for i in lc_info])
        result_str = processing(
            table, f_path, method=args.m, make_plots=args.p, twostep=args.step
        )
        try:
            os.makedirs("output_log")  
            os.makedirs("lc_metadata")
        except FileExistsError:
            pass
        #try:
        #    os.makedirs("lc_metadata")
        #except FileExistsError:
        #    pass
        lock.acquire()
        with open(os.path.join("output_log/", args.of), "a") as out_file:
            out_file.write(result_str + "\n")
        if get_metadata:
            with open(
                os.path.join("lc_metadata/", f"metadata_sector_{sector}"),"a") as out_file2:
                out_file2.write(f + " " + lc_info + " " + f"{sector}" + "\n")
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    print(f"Script started at {time.ctime()}.")
    start_time = time.time()
    try:
        if "sector" in args.path[0]:
            sector = int(os.path.split(args.path[0])[0].split("sector")[1].split("_")[1])
            print(sector)
        else:
            sector = int(input("Sector? "))
    except IndexError:
            sector = int(input("Sector? "))

    pool = multiprocessing.Pool(processes=args.threads)

    for path in paths:
        if not os.path.isdir(path):
            if os.path.isfile(path):
                run_lc(path,args.metadata)
            #else:
            #    print(path, "not a directory or valid file, skipping.", file=sys.stderr)
            #continue

        # if we are in the lowest subdirectory, perform glob this way.
        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # works for both Kepler and TESS fits files.
            fits_files = glob.glob(os.path.join(path, "*lc.fits"))
            pkl_files = glob.glob(os.path.join(path, "*.pkl"))

            pool.map(run_lc, pkl_files)
            pool.map(run_lc, fits_files)

        else:
            print("globbing subdirectories")

            # Start at Sector directory, glob goes through `target/000x/000x/xxxx/**/*lc.fits`
            fits_files = glob.glob(os.path.join(path, "target/**/**/**/**/*lc.fits"))
            pkl_files = glob.glob(os.path.join(path, "**/*.pkl"))

            print("running the search...")

            pool.map(run_lc, pkl_files)
            pool.map(run_lc, fits_files)
    end_time = time.time()
    total_time = end_time - start_time
    if total_time <= 60:
        print(f"Completed in {(total_time)/60} minutes")
    elif total_time <= 60*60 and total_time > 60:
        print(f"Completed in {(total_time)/(3600)} hours")
    else:
        print(f"Completed in {(total_time)} seconds")

## trying a piece of code here. please work
