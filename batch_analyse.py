#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os

# os.nice(8)
os.environ["OMP_NUM_THREADS"] = "1"
from analysis_tools_cython import *
import multiprocessing
import sys
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
    "-q", help="Keep only points with SAP_QUALITY=0", action="store_true"
)
parser.add_argument(
    "-f",
    help='select flux. "corrected flux is default. XRP lightcurve options are "corrected flux", "PCA flux" or "raw flux". For other lightcurves, options are "PDCSAP_FLUX"',
    dest="f",
    default="corrected flux",
)
parser.add_argument(
    "-c",
    help="select sigma clipping threshold for XRP lightcurves.",
    dest="c",
    default=3,
    type=int,
)

parser.add_argument("-step", help="default twostep", dest="step", action="store_true")
parser.add_argument("-p", help="enable plotting", action="store_true")

parser.add_argument("-n", help="set niceness", dest="n", default=8, type=int)
parser.add_argument(
    "-m",
    help='set smoothing method. Default is lowess.',
    dest="m",
    default="lowess",
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


def run_lc(f_path):
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
            table, lc_info = import_lightcurve(
                f_path, flux=args.f
            )
        lc_info = " ".join([str(i) for i in lc_info])
        result_str = processing(
            table, f_path, method=args.m, make_plots=args.p, twostep=args.step
        )
        try:
            os.makedirs("output_log")  # make directory plot if it doesn't exist
            os.makedirs("output_log_xrp")
        except FileExistsError:
            pass
        try:
            os.makedirs("lightcurve_info")
        except FileExistsError:
            pass
        lock.acquire()
        if f_path.endswith(".pkl"):
            with open(os.path.join("output_log_xrp/", args.of), "a") as out_file:
                out_file.write(result_str + "\n")

            with open(
                os.path.join(
                    "lightcurve_info/", f"xrp_lightcurve_info_sector_{sector}"
                ),
                "a",
            ) as out_file_2:
                out_file_2.write(lc_info + "\n")
        else:
            with open(os.path.join("output_log/", args.of), "a") as out_file:
                out_file.write(result_str + "\n")

            with open(
                os.path.join("lightcurve_info/", f"lightcurve_info_sector_{sector}"),
                "a",
            ) as out_file_2:
                out_file_2.write(lc_info + "\n")

        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":

    if "sector" in args.path[0]:
        sector = int(os.path.split(args.path[0])[0].split("sector")[1].split("_")[1])
        print(sector)

    else:
        sector = int(input("Sector? "))

    pool = multiprocessing.Pool(processes=args.threads)

    for path in paths:
        if not os.path.isdir(path):
            if os.path.isfile(path):
                print(path)
            else:
                print(path, "not a directory, skipping.", file=sys.stderr)
            continue

        # if we are in the lowest subdirectory, perform glob this way.
        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # this should work for both Kepler and TESS fits files.
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

## trying a piece of code here. please work
