import os
import argparse
import glob
import multiprocessing
import pandas as pd
from astropy.io import fits

parser = argparse.ArgumentParser(
    description="A script to delete files that are fainter than our cutoff magnitude."
)
parser.add_argument(help="target directory", default=".", nargs="+", dest="path")

args = parser.parse_args()

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))


lookup = pd.read_csv("project_lookup.csv")


def read_lightcurve(f, lookup=lookup):
    try:
        hdul = fits.open(f)
        ticid = hdul[0].header["TICID"]
        hdul.close()
    except FileNotFoundError:
        print("Import failed: file not found")
        return

    failed_tics = []
    try:
        mag = lookup[lookup["TIC"] == ticid].reset_index(drop=True).loc[0].Magnitude
    except IndexError:
        failed_tics.append(ticid)
        print(f"Import failed: TIC {ticid} not found in lookup table.")

    if mag > 13:
        os.remove(f)
        print(
            f"file {ticid} deleted, magnitude {mag}. Subdirectory {os.path.dirname(f)} deleted."
        )
        os.rmdir(os.path.dirname(f))
    else:
        print("TIC" + ticid)


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count() - 1

    for path in paths:
        pool = multiprocessing.Pool(processes=num_processes)
        files = glob.glob(os.path.join(path, "**/**/**/**/*.fits"))
        pool.map(read_lightcurve, files)
