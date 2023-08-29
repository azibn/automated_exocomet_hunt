import os
import sys
import argparse
import glob
from analysis_tools_cython import import_XRPlightcurve
import multiprocessing

parser = argparse.ArgumentParser(
    description="A script to delete files that are fainter than our cutoff magnitude in the 2_min_cadences_folder"
)
parser.add_argument(help="target directory", default=".", nargs="+", dest="path")
parser.add_argument("-s", help="TESS sector", dest="s", type=int)

args = parser.parse_args()

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))


def read_lightcurve(f, sector=args.s):
    data, lc_info = import_XRPlightcurve(f, sector)
    print("TIC" + str(lc_info[0]))
    if lc_info[3] > 13:
        os.remove(f)
        print(f"file {lc_info[0]} deleted, magnitude {lc_info[3]}")


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count() - 1
    for path in paths:
        # print(path)
        # read_lightcurve(path,args.s)
        pool = multiprocessing.Pool(processes=num_processes)
        files = glob.glob(os.path.join(path, "*.pkl"))
        pool.map(read_lightcurve, files)
