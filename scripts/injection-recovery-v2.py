import pandas as pd
import argparse
import os
import multiprocessing
import numpy as np
import random
import json
from tqdm import tqdm
from astropy.table import Table
from analysis_tools_cython import *
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

"""This script is an udpated injection-recovery with the eleanor-lite pipeline. It also takes into account the post-search phase and adds the distribution cuts into the mix."""


parser = argparse.ArgumentParser(description='Injecting recovery for lightcurves')
parser.add_argument('--directory', type=str, help='The directory of the lightcurves',dest='directory')
parser.add_argument('--mag_lower', type=float, default=3, help='The lower magnitude limit',dest='mag_lower')
parser.add_argument('--mag_upper', type=float, default=13, help='The upper magnitude limit',dest='mag_upper')
parser.add_argument('--sector', type=int, default=None, help='TESS Sector',dest='sector')
parser.add_argument('--number', type=int, default=50000, help='Sample size',dest='number')
parser.add_argument('-o','--output',help='output file name.',default='output.txt',dest='output')

parser.add_argument(
    "--systematic-bins",
    action="store_true",
    help="Use systematic depth bins instead of random sampling",
    dest="systematic_bins"
)

parser.add_argument(
    "-n-per-bin",
    help="Number of injections per magnitude-depth bin",
    dest="n_per_bin",
    default=20,
    type=int,
)

args = parser.parse_args()

if args.directory is None:
    print('loading dataframe')
    dataset = pd.read_parquet('extreme-df.parquet')
    print('dataframe loaded')


os.makedirs('injection-recovery-plots-test/',exist_ok=True)
os.makedirs(f'injection-recovery-plots-test/tmag-{args.mag_lower}-{args.mag_upper}/',exist_ok=True)




def sample(dataset, number=None, mag_lower=None, mag_upper=None, sector=None):
    mask = (dataset.Tmag > mag_lower) & (dataset.Tmag < mag_upper)
    if sector is not None:
        mask &= (dataset.Sector == sector)
    data = dataset[mask]
    
    if number is None:
        return data.filepath2.values.tolist()
    return data.sample(n=number, replace=True).filepath.values.tolist() 




def skewed_transit_model(x, A, t0, sigma=3.28541476e-01, alpha=1.43857307e+00):
    """
    Injecting lightcurve using skewed Gaussian function. The default is the Beta Pic parameters.

    Parameters:
        flux: the lightcurve flux
        x: Input data points.
        A: Amplitude of the Gaussian.
        t0: injected time
        sigma: Standard deviation of the Gaussian. # find the Beta Pic
        alpha: Skewness parameter (find the one for Beta Pic)

    Returns:
        y: The value of the skewed Gaussian at each input data point x.

    Notes: These are the Beta Pic parameters:
        - A = 8.84860547e-04, t0 = 1.48614591e+03, sigma = 3.28541476e-01, alpha = 1.43857307e+00
    """

   

    return 1 - (A * skewnorm.pdf(x,alpha,loc=t0, scale=sigma))


def transit_model(time, depth, injected_time):
    return 1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)

def run_injection(file, depth=None):
    """
    Run injection recovery on a single file
    
    Parameters:
    -----------
    file : str
        Path to lightcurve file
    depth : float, optional
        If provided, uses this specific depth. If None, samples randomly.
    """
    lc, lc_info = import_lightcurve(file)
    #lc = lc.to_pandas()

    
    # Injecting the transit in the lightcurve
    if depth is None:
        # Original random sampling
        depth = 10 ** np.random.uniform(-4, -2, 1)[0]

    # ## time range - duration of lightcurve
    # time_range = lc["TIME"][
    #         lc["TIME"].between(
    #             lc["TIME"].min() + 1, lc["TIME"].max() - 1, inclusive=False
    #         )
    #     ].reset_index(drop=True)
    
    # _, injected_time = random.choice(list(enumerate(time_range)))
    injected_time = 1485.5

    # Create comet model
    injected_flux = transit_model(lc['TIME'], depth, injected_time)

    # Injecting into lightcurve
    lc['FLUX'] = lc['PCA_FLUX'] * injected_flux
    time_array = np.array(lc['TIME'])

    injected_idx = np.argmin(np.abs(time_array - injected_time))
    recovered_range = time_array[injected_idx - 4 : injected_idx + 4]

    # recovered_range = lc.TIME[
    #         lc.TIME.loc[lc.TIME == injected_time].index[0]
    #         - 6 : lc.TIME.loc[lc.TIME == injected_time].index[0]
    #         + 7].reset_index(drop=True)

    #lc = Table.from_pandas(lc)
    lc = lc[['TIME','FLUX','QUALITY','FLUX_ERR']]
    results, data_arrays = processing(lc,lc_info=lc_info,method='median')

    #time = float(results[4])

    return lc_info[0], injected_time, depth, results.split(), recovered_range

def process_file_systematic(args):
    """Wrapper for systematic bin processing"""
    file, depth = args
    return process_file(file, depth)


def process_file(file, depth=None):
    """Modified to accept optional depth parameter"""
    tic_id, injected_time, depth, results, recovered_range = run_injection(file, depth)
    recovered_range = recovered_range.tolist()
    return {'TIC_ID': tic_id, 'injected_time': injected_time, 'injected_depth': depth,'results':results,'recovered_range':recovered_range}


def assign_recovery_conditions(df):
    print(df.columns)
    df['recovered'] = ((df['time'].astype(float) > df['recovered_range'].str[0]) & 
                  (df['time'].astype(float) < df['recovered_range'].str[-1])).astype(int)
    return df


if __name__ == '__main__':
    if args.directory:
        files = []
        # Add desc to tqdm for finding files
        for root, _, filenames in tqdm(os.walk(args.directory), desc="Finding files"):
            for filename in filenames:
                files.append(os.path.join(root, filename))  # Removed print since tqdm shows progress
        
        files = random.choices(files, k=args.number)
        # with open('files-tmag-13-14.txt', 'w') as f:
        #     for file in files:
        #         f.write(file + '\n')
    else:
        files = sample(dataset, number=args.number, mag_lower=args.mag_lower, mag_upper=args.mag_upper, sector=args.sector)

    df_results = []
    if args.systematic_bins:
        depth_edges = np.logspace(-4, -2, 11)
        depth_bins = list(zip(depth_edges[:-1], depth_edges[1:]))
        
        processing_list = []
        # Add tqdm for bin creation
        for depth_min, depth_max in tqdm(depth_bins, desc="Creating depth bins"):
            for _ in range(args.n_per_bin):
                file = random.choice(files)
                depth = np.random.uniform(depth_min, depth_max)
                processing_list.append((file, depth))
        
        for item in tqdm(processing_list, desc="Processing files"):
            result = process_file_systematic(item)
            df_results.append(result)
    else:
        for file in tqdm(files, desc="Processing files"):
            result = process_file(file)
            df_results.append(result)

    print("Creating DataFrames...")
    df = pd.DataFrame(df_results)
    results_df = pd.DataFrame()
    

    # Add tqdm for result processing
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing results"):
        result_series = pd.Series(row['results'])
        results_df = pd.concat([results_df, result_series.to_frame().T], ignore_index=True)

    # Load column names
    with open('colnames.json', 'r', encoding='utf-8') as f:
        columns = json.loads(f.read())['column_names']
        
    
    print("Loaded columns:", columns)

    results_df.columns = columns
    df = df.drop('results', axis=1) 
    df = pd.concat([df, results_df], axis=1) 

    df = assign_recovery_conditions(df)

    print(df.recovered.value_counts())

    output_dir = 'injection-recovery-plots-test/'
    os.makedirs(output_dir, exist_ok=True)
    print("Shape of final DataFrame:", df.shape)
    print("Attempting to save to:", os.path.join(output_dir, args.output))
    df.to_csv(args.output, index=False)
    print("Save completed")