"""Injection-recovery for the eleanor-lite TESS pipeline.

Runs either random-depth sampling or systematic depth-binned sampling over a
magnitude-filtered set of lightcurves, then writes the post-search results CSV
with a time-based recovery flag.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis_tools_cython import *


parser = argparse.ArgumentParser(description='Injecting recovery for lightcurves')
parser.add_argument('--directory', type=str, help='The directory of the lightcurves', dest='directory')
parser.add_argument('--mag_lower', type=float, default=3, help='The lower magnitude limit', dest='mag_lower')
parser.add_argument('--mag_upper', type=float, default=13, help='The upper magnitude limit', dest='mag_upper')
parser.add_argument('--sector', type=int, default=None, help='TESS Sector', dest='sector')
parser.add_argument('--number', type=int, default=50000, help='Sample size', dest='number')
parser.add_argument('-o', '--output', help='output file name.', default='output.txt', dest='output')
parser.add_argument(
    "--systematic-bins",
    action="store_true",
    help="Use systematic depth bins instead of random sampling",
    dest="systematic_bins",
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
    dataset = pd.read_parquet('data/extreme-df.parquet')
    print('dataframe loaded')


def sample(dataset, number=None, mag_lower=None, mag_upper=None, sector=None):
    mask = (dataset.Tmag > mag_lower) & (dataset.Tmag < mag_upper)
    if sector is not None:
        mask &= (dataset.Sector == sector)
    data = dataset[mask]

    if number is None:
        return data.filepath.values.tolist()
    return data.sample(n=number, replace=True).filepath.values.tolist()


def transit_model(time, depth, injected_time):
    return 1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)


def run_injection(file, depth=None):
    """Run injection recovery on a single file.

    If `depth` is None, samples log-uniform in [1e-4, 1e-2].
    """
    lc, lc_info = import_lightcurve(file)

    if depth is None:
        depth = 10 ** np.random.uniform(-4, -2, 1)[0]

    injected_time = 1485.5

    injected_flux = transit_model(lc['TIME'], depth, injected_time)
    lc['FLUX'] = lc['PCA_FLUX'] * injected_flux

    time_array = np.array(lc['TIME'])
    injected_idx = np.argmin(np.abs(time_array - injected_time))
    recovered_range = time_array[injected_idx - 4 : injected_idx + 4]

    lc = lc[['TIME', 'FLUX', 'QUALITY', 'FLUX_ERR']]
    results, _ = processing(lc, lc_info=lc_info, method='median')

    return lc_info[0], injected_time, depth, results.split(), recovered_range


def process_file(file, depth=None):
    tic_id, injected_time, depth, results, recovered_range = run_injection(file, depth)
    return {
        'TIC_ID': tic_id,
        'injected_time': injected_time,
        'injected_depth': depth,
        'results': results,
        'recovered_range': recovered_range.tolist(),
    }


def assign_recovery_conditions(df):
    print(df.columns)
    df['recovered'] = (
        (df['time'].astype(float) > df['recovered_range'].str[0])
        & (df['time'].astype(float) < df['recovered_range'].str[-1])
    ).astype(int)
    return df


if __name__ == '__main__':
    if args.directory:
        files = []
        for root, _, filenames in tqdm(os.walk(args.directory), desc="Finding files"):
            for filename in filenames:
                files.append(os.path.join(root, filename))
        files = random.choices(files, k=args.number)
    else:
        files = sample(
            dataset,
            number=args.number,
            mag_lower=args.mag_lower,
            mag_upper=args.mag_upper,
            sector=args.sector,
        )

    df_results = []
    if args.systematic_bins:
        depth_edges = np.logspace(-4, -2, 11)
        depth_bins = list(zip(depth_edges[:-1], depth_edges[1:]))

        processing_list = []
        for depth_min, depth_max in tqdm(depth_bins, desc="Creating depth bins"):
            for _ in range(args.n_per_bin):
                file = random.choice(files)
                depth = np.random.uniform(depth_min, depth_max)
                processing_list.append((file, depth))

        for file, depth in tqdm(processing_list, desc="Processing files"):
            df_results.append(process_file(file, depth))
    else:
        for file in tqdm(files, desc="Processing files"):
            df_results.append(process_file(file))

    print("Creating DataFrames...")
    df = pd.DataFrame(df_results)
    results_df = pd.DataFrame()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing results"):
        result_series = pd.Series(row['results'])
        results_df = pd.concat([results_df, result_series.to_frame().T], ignore_index=True)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colnames.json'), 'r', encoding='utf-8') as f:
        columns = json.loads(f.read())['column_names']

    print("Loaded columns:", columns)
    results_df.columns = columns
    df = df.drop('results', axis=1)
    df = pd.concat([df, results_df], axis=1)

    df = assign_recovery_conditions(df)

    print(df.recovered.value_counts())

    print("Shape of final DataFrame:", df.shape)
    print("Attempting to save to:", args.output)
    df.to_csv(args.output, index=False)
    print("Save completed")
