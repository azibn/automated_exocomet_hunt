import pandas as pd
import numpy as np
import glob
import json
import os
from analysis_tools_cython import processing, import_XRPlightcurve, import_lightcurve
import argparse
import multiprocessing  
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser(description="Get cutouts of lightcurves across multiple sectors.")




parser.add_argument(
    "-t", help="number of threads to use", default=40, dest="threads", type=int
)

parser.add_argument(
    "-m",
    help="set smoothing method. Default is median.",
    dest="m",
    default='median',
    type=str,
)

parser.add_argument('-save_dir_name',help='name of directory to save files to.', default='som_cutouts', dest='som_cutouts_directory_name')

parser.add_argument('-nice', help='set niceness', dest='nice', default=8, type=int)
parser.add_argument('-snr', '--snr-threshold', help= 'SNR threshold to use for selecting lightcurves', dest='snr', default=5, type=float)



args = parser.parse_args()


os.nice(args.nice)

# Define a function to process a single file
def process_file(filepath):
    try:
        print(filepath)
        lc, lc_info = import_XRPlightcurve(filepath,sector= int(filepath.split('sector')[1].split('_')[1]))
        lc = lc['time','corrected flux','quality','flux error']
        results, _ = processing(lc,lc_info=lc_info,method=args.m,som_cutouts=True,som_cutouts_directory_name=args.som_cutouts_directory_name)
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error while processing {filepath}: {str(e)}"


def process_file2(filepath):
    try:
        print(filepath)
        lc, lc_info = import_lightcurve(filepath)
        lc = lc[lc.colnames[:5]]
        results, _ = processing(lc,lc_info=lc_info,method=args.m,som_cutouts=True,som_cutouts_directory_name=args.som_cutouts_directory_name)
    except FileNotFoundError:
        return f"File not found: {filepath}"
        pass
    except Exception as e:
        return f"Error while processing {filepath}: {str(e)}"
        pass

if __name__ == '__main__':
    # Load the DataFrame with filepaths
    print("reading in dataframe.")
    df = pd.read_csv('eleanor-lite-v2_candidates.csv') #,sep=" ",skiprows=1,header=None)
    print("dataframe imported.")
    # ## commented because this is the version where the anomalies are removed.
    # #with open("colnames.json", "r", encoding="utf-8") as f:
    # #    check = f.read()
    # #    columns = json.loads(check)
    # #    columns = columns["column_names"]
    # #df.columns = columns
    # print(df)
    # print("dataframe read. Now applying filters and cuts")
    # df = df[df.transit_prob == 'maybeTransit']

    # #df['abs_path'] = df['path'].str.replace('/tmp/tess/', '/storage/astro2/phrdhx/tesslcs/')

    # df = df[(df.asym_score <= 3)].reset_index(drop=True)
    # df = df[df.snr <= 0]
    # data_new = df[abs(df.snr) >= args.snr].reset_index(drop=True)
    # data_new.duration = data_new.duration.astype(float)
    # data_new.drop(data_new[data_new['duration'] <= 0.4].index,inplace=True)
    # #data_new['abs_depth'] = abs(data_new.depth)

    # # getting rid of anything that is more than 0.1% transit depth
    # data_new.drop(data_new[data_new['depth'] > 0].index,inplace=True)
    # data_new.abs_depth = data_new.abs_depth.astype(float)
    # data_new.drop(data_new[(data_new['abs_depth'] >= 0.01)].index,inplace=True)

    # # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=args.threads)
    print("starting multiprocessing.")
    # Use the `map` function to apply the `process_file` function to each filepath in parallel
    processed_data = pool.map(process_file2, df['abs_path'])
    
    # Close the pool of worker processes
    pool.close()
    pool.join()
    
    # Now, `processed_data` contains the processed data for each file
    # You can work with this data as needed
    print(f"process finished, saved to {args.som_cutouts_directory_name}")

