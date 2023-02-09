import os
import multiprocessing
import argparse
import glob
from analysis_tools_cython import import_XRPlightcurve, folders_in

parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

parser.add_argument("-o", default=f"output.txt", dest="of", help="output file")


parser.add_argument("-n", help="set niceness", dest="n", default=8, type=int)

args = parser.parse_args()

# set niceness
os.nice(args.n)

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))

m = multiprocessing.Manager()
lock = m.Lock()


def get_metadata(f_path):
    f = os.path.basename(f_path)
    print(f)
    if f_path.endswith(".pkl"):
        _, data = import_XRPlightcurve(f_path, sector)
    lock.acquire()
    with open(
        os.path.join("lightcurve_info/", f"xrp_lightcurve_info_sector_{sector}"),
        "a",
    ) as out_file_2:
        out_file_2.write(data + "\n")
    lock.release()


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=args.threads)
    for path in paths:
        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the search...")

            # this should work for both Kepler and TESS fits files.
            fits_files = glob.glob(os.path.join(path, "*lc.fits"))
            pkl_files = glob.glob(os.path.join(path, "*.pkl"))

            pool.map(run_lc, pkl_files)
            pool.map(run_lc, fits_files)
