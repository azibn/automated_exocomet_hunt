#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
## Change directory of batch analyse to one above, so that outputs can be put in above dir and not in `scripts`.
import os
import multiprocessing
import sys
import traceback
import argparse
import glob
import warnings
import numpy as np
from analysis_tools_cython import (
    import_XRPlightcurve,
    import_lightcurve,
    processing,
    folders_in,
)
from tqdm import tqdm
import time

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

parser.add_argument("-o", default=f"output.txt", dest="of", help="output file")

parser.add_argument(
    "-f",
    help='select flux. "CORR_FLUX is default (`eleanor-lite`). XRP lightcurve options are "corrected flux", "PCA flux" or "raw flux". For other lightcurves, options are "PDCSAP_FLUX"',
    dest="f",
    default="CORR_FLUX",
)
parser.add_argument(
    "-c",
    help="set sigma clipping threshold for MAD cuts.",
    dest="c",
    default=3,
    type=int,
)

parser.add_argument(
    "-step",
    help="enable twostep. used in conjuction with -m fourier",
    dest="step",
    action="store_true",
)
parser.add_argument("-p", help="enable plotting", action="store_true", dest="p")

parser.add_argument(
    "-metadata",
    help="save metadata of the lightcurves as a .txt file",
    dest="metadata",
    action="store_true",
)

parser.add_argument("-nice", help="set niceness", dest="nice", default=8, type=int)


parser.add_argument(
    "-m",
    help="set smoothing method. Default is None.",
    dest="m",
    default=None,
    type=str,
)

parser.add_argument("-n", help="does not save output file", action="store_true")

parser.add_argument(
    "-pipeline",
    help="pipeline choice. Default is `eleanor-lite`. Other options are `spoc` and `xrp`",
    default="eleanor-lite",
    type=str
)


parser.add_argument(
    "-q",
    help="drop bad quality data. To keep bad quality data, call this argument. Default is True",
    action="store_false",
)

parser.add_argument(
    "-som_cutouts",
    help="extract lightcurves cutouts for SOM clustering. Default is False.",
    action="store_true",
    dest="som",
)



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


def run_lc(f_path):
    """
     Function: Processes the lightcurves

     Args:
         f_path (str): Path to the lightcurve file.
         get_metadata (bool, optional): Flag indicating whether to retrieve metadata. Defaults to args.metadata.
         return_arraydata (bool, optional): Flag indicating whether to return array data. Defaults to args.return_arraydata.

     Raises:
         KeyboardInterrupt: Raised when the process is terminated early.
         SystemExit: Raised when the process is terminated by a system exit.

    Lightcurve results are saved in `output_log/{file.txt}`, unless specified with the no save argument `-n`.

    """

    try:
        f = os.path.basename(f_path)
        print(f_path)
        if f_path.endswith(".pkl"):

            table, lc_info = import_XRPlightcurve(
                f_path, sector=sector, clip=args.c, drop_bad_points=args.q
            )
            table = table[table.colnames[:5]]

        else:
            table, lc_info = import_lightcurve(f_path, flux=args.f, pipeline=args.pipeline)
            table = table[table.colnames[:5]]
        result_str, save_data = processing(
            table,
            f_path,
            lc_info,
            method=args.m,
            make_plots=args.p,
            twostep=args.step,
            som_cutouts=args.som,
        )

        if args.n:
            # print result_str in terminal
            print(result_str)
            return
        
        # make directory for output file.
        os.makedirs("outputs", exist_ok=True)

        lc_info = " ".join([str(i) for i in lc_info])

        lock.acquire()
        with open(os.path.join("outputs", args.of), "a") as out_file:
            out_file.write(result_str + "\n")


        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting", file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file " + f_path, file=sys.stderr)
        traceback.print_exc()
    except:
        print(f"An error occurred: {e}")
    # Exit the script with an error code
        sys.exit(1)

# def find_fits_files(path: str) -> List[str]:
#     """
#     Recursively searches for all .fits files in the specified directory and its subdirectories.
#     Returns a list of file paths.
#     """
#     fits_files = []
#     for entry in tqdm(os.scandir(path)):
#         if entry.is_file() and entry.name[-5:] == ".fits":
#             fits_files.append(entry.path)
#         elif entry.is_dir():
#             fits_files.extend(find_fits_files(entry.path))
#     return fits_files





if __name__ == "__main__":
    if ("sector" in args.path[0]) & (args.path[0].endswith('.pkl')):
        sector = int(os.path.split(args.path[0])[0].split("sector")[1].split("_")[1])
        print(f"sector {sector}")
    else:
        sector = input("Sector? ")

    pool = multiprocessing.Pool(processes=args.threads)
    
    for path in paths:
        if not os.path.isdir(path):
            if os.path.isfile(path) & path.endswith(".fits"):
                run_lc(path)

                sys.exit()
            elif path.endswith(".txt"):
                "processes a list of TIC IDs"
                print("processing txt file")
                with open(path, "r") as file:
                    files = [line.strip() for line in file.readlines()]
                pool.map(run_lc, files)
                sys.exit()


        # if we are in the lowest subdirectory, perform glob this way.

        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # works for both Kepler and TESS fits files.
            fits = glob.glob(os.path.join(path, "*lc.fits"))
            pkl = glob.glob(os.path.join(path, "*.pkl"))

            pool.map(run_lc, fits)
            pool.map(run_lc, pkl)



        else:
            print("globbing subdirectories")
            # Start at Sector directory, glob goes through `target/000x/000x/xxxx/**/*lc.fits`

            fits = glob.glob(os.path.join(path, "**/*lc.fits"),recursive=True)
            pkl = glob.glob(os.path.join(path, "**/*.pkl"),recursive=True)

            pool.map(run_lc, fits)
            pool.map(run_lc, pkl)

    pool.close()
    pool.join()