import os
import argparse
import glob
import multiprocessing
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import time
import sys

parser = argparse.ArgumentParser(
    description="A script to delete files that are fainter than our cutoff magnitude."
)


args = parser.parse_args()

paths = []

def read_lightcurve(file):
    #print(f)
    hdul = fits.open(file)
    ticid = hdul[0].header["TIC_ID"]
    hdul.close()
    
    mag = data[data.TIC_ID == ticid]["Magnitude"].values[0]

    if mag > 13:
        os.remove(file)
        print(
            f"file {ticid} deleted, magnitude {mag}. Subdirectory {os.path.dirname(file)} deleted."
        )
        os.rmdir(os.path.dirname(file))
    else:
        print(f"TIC {ticid}, magnitude {mag}")

if __name__ == "__main__":
    print("reading data")
    data = pd.read_csv("s9.csv")
    mag = data['Magnitude']
    files = data['new_path'].to_list()
    print("file uploaded")
    num_processes = multiprocessing.cpu_count() - 1
    
    pool = multiprocessing.Pool(processes=num_processes)
    print("number of threads used: " + str(num_processes))

    pool.map(read_lightcurve, files)
        #read_lightcurve(i)

    pool.close()
    pool.join()

