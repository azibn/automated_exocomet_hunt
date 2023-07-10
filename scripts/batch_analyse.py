#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
## Change directory of batch analyse to one above, so that outputs can be put in above dir and not in `scripts`.
import os
os.chdir('../')
import multiprocessing
import sys
import traceback
import argparse
import glob
import warnings
import numpy as np
from scripts.analysis_tools_cython import (
    import_XRPlightcurve,
    import_lightcurve,
    processing,
    folders_in,
)
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

parser.add_argument(
    "-step",
    help="enable twostep. used in conjuction with -m fourier",
    dest="step",
    action="store_true",
)
parser.add_argument("-p", help="enable plotting", action="store_true")
parser.add_argument(
    "-metadata",
    help="save metadata of the lightcurves as a .txt file",
    dest="metadata",
    action="store_true",
)

parser.add_argument("-nice", help="set niceness", dest="nice", default=8, type=int)


parser.add_argument(
    "-return_arraydata",
    help="save the full processed lightcurve as an .npz file",
    dest="return_arraydata",
    action="store_true",
)

parser.add_argument(
    "-m",
    help="set smoothing method. Default is None.",
    dest="m",
    default=None,
    type=str,
)

parser.add_argument("-n", help="does not save output file", action="store_true")
parser.add_argument(
    "-q",
    help="drop bad quality data. To keep bad quality data, call this argument. Default is True",
    action="store_false",
)

parser.add_argument('-prep_som', help='extract lightcurves cutouts for SOM clustering. Default is False.', action='store_true',dest='som')

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


def run_lc(f_path, get_metadata=args.metadata, return_arraydata=args.return_arraydata):
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
            table = table["time", args.f, "quality", "flux error"]

        else:
            table, lc_info = import_lightcurve(f_path, flux=args.f)
            table = table["TIME", args.f, "QUALITY","PDCSAP_FLUX_ERR"]
        result_str, save_data = processing(
            table, f_path, lc_info, method=args.m, make_plots=args.p, twostep=args.step, som_cutouts=args.som
        )

        try:
            os.makedirs("../output_log")
            # os.makedirs("lc_metadata")
        except FileExistsError:
            pass
        try:
            os.makedirs("lc_metadata")
        except FileExistsError:
            pass

        if return_arraydata:
            if f_path.endswith(".pkl"):
                obj_id = lc_info[1]
            else:
                obj_id = lc_info[1]
            try:
                os.makedirs(f"/storage/astro2/phrdhx/tesslcs/lc_arraydata/")
            except FileExistsError:
                pass
            try:
                os.makedirs(
                    f"/storage/astro2/phrdhx/tesslcs/lc_arraydata/sector_{sector}"
                )
            except FileExistsError:
                pass
            # except NameError:
            #     sector = input("sector:")
            #     os.makedirs(
            #         f"/storage/astro2/phrdhx/tesslcs/lc_arraydata/sector_{sector}"
            #     )
            np.savez(
                f"/storage/astro2/phrdhx/tesslcs/lc_arraydata/sector_{sector}/tesslc_{obj_id}.npz",
                obj_id=obj_id,
                time=save_data[0],
                flux=save_data[1],
                trend_flux=save_data[2],
                quality=save_data[3],
            )  # ,agg_flux=flux_aggressive_filter,agg_trend_flux=trend_flux_aggressive_filter,quality=quality)

        if args.n:
            return

        lc_info = " ".join([str(i) for i in lc_info])

        lock.acquire()
        with open(os.path.join("../output_log/", args.of), "a") as out_file:
            out_file.write(result_str + "\n")
        if get_metadata:
            with open(
                os.path.join("lc_metadata/", f"metadata_sector_{sector}.txt"), "a"
            ) as out_file2:
                out_file2.write(f + " " + lc_info + " " + f"{sector}" + "\n")

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
        print(f"sector {sector}")
    else:
        sector = int(input("sector: "))

    pool = multiprocessing.Pool(processes=args.threads)

    for path in paths:
        if not os.path.isdir(path):
            if os.path.isfile(path):
                run_lc(path, args.metadata)

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
            fits = glob.glob(os.path.join(path, "target/**/**/**/**/*lc.fits"))
            pkl = glob.glob(os.path.join(path, "**/*.pkl"))

            print("running the search...")

            pool.map(run_lc, fits)
            pool.map(run_lc, pkl)
            
