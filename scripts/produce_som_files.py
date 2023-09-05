import pandas as pd
import numpy as np
import glob
import json
import os
from analysis_tools_cython import processing, import_XRPlightcurve
import argparse
import multiprocessing  
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")

parser.add_argument(
    "-t", help="number of threads to use", default=40, dest="threads", type=int
)

parser.add_argument(
    "-m",
    help="set smoothing method. Default is None.",
    dest="m",
    default='median',
    type=str,
)

parser.add_argument('-nice', help='set niceness', dest='nice', default=19, type=int')


args = parser.parse_args()


os.nice(args.nice)

# Define a function to process a single file
def process_file(filepath):
    try:
        print(filepath)
        lc, lc_info = import_XRPlightcurve(filepath,sector= int(filepath.split('sector')[1].split('_')[1]))
        lc = lc['time','corrected flux','quality','flux error']
        results, _ = processing(lc,lc_info=lc_info,method=args.m,som_cutouts=True)
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error while processing {filepath}: {str(e)}"

if __name__ == '__main__':
    # Load the DataFrame with filepaths
    df = pd.read_csv('som_candidates.csv')  # Adjust the file format and path accordingly
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=args.threads)
    
    # Use the `map` function to apply the `process_file` function to each filepath in parallel
    processed_data = pool.map(process_file, df['abs_path'])
    
    # Close the pool of worker processes
    pool.close()
    pool.join()
    
    # Now, `processed_data` contains the processed data for each file
    # You can work with this data as needed


