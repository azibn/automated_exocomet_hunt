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


parser = argparse.ArgumentParser(description="Get metadata from target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument("-o", default=f"output.txt", dest="of", help="output file")
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

def get_metadata(f_path):
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
        _, lc_info = import_lightcurve(f_path)

        os.makedirs("metadata", exist_ok=True)

        lc_info = " ".join([str(i) for i in lc_info])

        lock.acquire()
        with open(os.path.join("metadata",args.of), "a") as file:
            file.write(lc_info + "\n")
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
                get_metadata(path)

                sys.exit()
            elif path.endswith(".txt"):
                "processes a list of TIC IDs"
                print("processing txt file")
                with open(path, "r") as file:
                    files = [line.strip() for line in file.readlines()]
                pool.map(get_metadata, files)
                sys.exit()


        # if we are in the lowest subdirectory, perform glob this way.

        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # works for both Kepler and TESS fits files.
            fits = glob.glob(os.path.join(path, "*lc.fits"))
            pkl = glob.glob(os.path.join(path, "*.pkl"))

            pool.map(get_metadata, fits)
            pool.map(get_metadata, pkl)



        else:
            print("globbing subdirectories")
            # Start at Sector directory, glob goes through `target/000x/000x/xxxx/**/*lc.fits`

            fits = glob.glob(os.path.join(path, "**/*lc.fits"),recursive=True)
            pkl = glob.glob(os.path.join(path, "**/*.pkl"),recursive=True)

            pool.map(run_lc, fits)
            pool.map(run_lc, pkl)

    pool.close()
    pool.join()