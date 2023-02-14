import argparse
import os
import multiprocessing
import math
import random
import numpy as np
import pandas as pd
from astropy.table import Table
from analysis_tools_cython import *

os.environ["OMP_NUM_THREADS"] = "1"


parser = argparse.ArgumentParser(description="Injection Testing")

parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")
parser.add_argument(
    "-s", help="sector", dest="sector", default=6, type=int
)  # ,default=f'{args.path[0].split('sector')[1].split('')}')
parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

# parser.add_argument("-o", default=f"injected_output.txt", dest="of", help="output file")
parser.add_argument(
    "-number",
    help="How many lightcurves to randomise. default is None",
    dest="number",
    default=4000,
    type=int,
)
parser.add_argument(
    "-percentage",
    help="Percentage of directory to randomise. input as decimal (i.e: 10% == 0.1)",
    dest="percentage",
    default=0.2,
)
parser.add_argument("-sector", help="lightcurve sector", dest="sector")
parser.add_argument("-save_csv", dest="save_csv", action="store_true")
parser.add_argument("-mag_lower", dest="mag_lower", help="lower magnitude", type=int)
parser.add_argument("-mag_higher", dest="mag_higher", help="higher magnitude", type=int)


args = parser.parse_args()

paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))

try:
    multiprocessing.set_start_method("fork")  # default for >=3.8 is spawn
except RuntimeError:  # there might be a timeout sometimes.
    pass

m = multiprocessing.Manager()
lock = m.Lock()

## import lookup data
lookup = pd.read_csv(f"/storage/astro2/phrdhx/tesslcs/sector{args.sector}lookup.csv")


def select_lightcurves(path):
    """returns a sample of random lightcures in the directory. Can be randomised by either number of ligthcurves or percentage of lightcurve files in the directory."""
    if args.number:
        return random.sample(os.listdir(path), args.number)
    if args.percentage:
        return random.sample(
            os.listdir(path), int(len(os.listdir(path)) * args.percentage)
        )


def new_select_lightcurves(mag_lower, mag_higher):
    return (
        lookup.Filename[
            (lookup.Magnitude >= mag_lower) & (lookup.Magnitude <= mag_higher)
        ]
        .sample(args.number)
        .values
    )


def select_lightcurves_multiple_directories(path):
    """this function works with files where lightcurves are categorised into magnitude directories.
    Selects random number of lightcurves in each subdirectory without the need to go into each directory."""
    a = []
    for i in os.listdir(path):
        try:
            if os.path.isdir(i):
                print(i)
                a.append(random.sample(os.listdir(os.path.join(path, i)), 1))
                print(a)
        except Exception as e:
            print(i, e)
            continue
    return [j for i in a for j in i]


def inject_lightcurve(flux, time, depth, injected_time):
    return flux * (
        1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)
    )


def run_injection(path, save_csv=args.save_csv):
    """returns lightcurves as a dataframe"""

    ## grab the random lightcurves
    # lc_paths = select_lightcurves(path)
    lc_paths = new_select_lightcurves(args.mag_lower, args.mag_higher)
    injected_depths = []
    injected_times = []
    mags = []
    recovered_depth = []
    results_for_binning = []
    recovered_or_not = []
    filename = []
    for i in lc_paths:
        print(i)
        filename.append(i)
        data, lc_info = import_XRPlightcurve(
            os.path.join(path, i), sector=args.sector, return_type="pandas"
        )

        depth = 10 ** np.random.uniform(-4, -2, 1)[0]
        injected_depths.append(depth)
        mags.append(lc_info[3])
        time_range = data["time"][
            data["time"].between(
                data["time"].min() + 1, data["time"].max() - 1, inclusive=False
            )
        ].reset_index(
            drop=True
        )  # resets index so consistency is kept when working with indices
        _ , injected_time = random.choice(list(enumerate(time_range)))
        injected_times.append(injected_time)

        ## comet model
        data["injected_dip_flux"] = data["corrected flux"] * (
            1
            - comet_curve(
                data["time"], depth, injected_time, 3.02715600e-01, 3.40346173e-01
            )
        )

        data = Table.from_pandas(data)
        data = data[["time", "injected_dip_flux", "quality", "flux error"]]
        results, data_arrays = processing(data, path, lc_info, method="median")
        try:
            os.makedirs("injection_recovery_data_arrays/")
        except FileExistsError:
            pass
        try:
            np.savez(
                f"injection_recovery_data_arrays/lc_info[0].npz",
                time=data_arrays[0],
                flux=data_arrays[1],
                trend_flux=data_arrays[2],
                quality=data_arrays[3],
            )
        except:
            np.savez(
                f"injection_recovery_data_arrays/{i}.npz",
                time=data_arrays[0],
                flux=data_arrays[1],
                quality=data_arrays[2],
            )
        results = results.split()
        recovered_time = float(results[2])
        new_depth = float(results[7])
        recovered_depth.append(float(results[7]))
        results_for_binning.append(results)

        data = data.to_pandas()
        recovered_range = data.time[
            data.time.loc[data.time == injected_time].index[0]
            - 5 : data.time.loc[data.time == injected_time].index[0]
            + 5
        ].reset_index(drop=True)

        try:
            percentage_change = (
                (abs(new_depth) - depth) / depth
            ) * 100  # 0-depth to consider normalisation
        except ZeroDivisionError:
            percentage_change = 0

        if (
            recovered_range.values[0] <= recovered_time <= recovered_range.values[-1]
        ) & (abs(percentage_change) <= 25):
            recovered = 1
            recovered_or_not.append(recovered)
        else:
            recovered = 0
            recovered_or_not.append(recovered)

    df = pd.DataFrame(results_for_binning)
    df.insert(0, "file", filename)
    df["recovered"] = recovered_or_not
    df["magnitude"] = mags
    df["injected_depths"] = injected_depths
    df["injected_time"] = injected_times

    cols = [
        "file",
        "path"
        "signal",
        "snr",
        "time",
        "asym_score",
        "width1",
        "width2",
        "duration",
        "depth",
        "peak_lspower",
        "mstat",
        "transit_prob",
        "recovered",
        "mag",
        "injected_depth",
        "injected_times",
        "",
    ]

    df.columns = cols
    df.depth = [float(i) for i in df.depth]

    if args.save_csv:
        try:
            os.makedirs("injection_recovery2")
            print("created directory injection_recovery2")
        except FileExistsError:
            pass

        try:
            os.makedirs(f"injection_recovery2/sector_{args.sector[0]}")
            print(f"created directory injection_recovery2/sector_{args.sector[0]}")
        except FileExistsError:
            pass

        try:
            df.to_csv(
                f"injection_recovery2/sector_{args.sector[0]}/tmag_{args.mag_lower}_tmag_{args.mag_higher}.csv"
            )
        except FileExistsError:
            pass

    return df


def run_injection2(path, save_csv=args.save_csv,save_arrays=False):
    """returns lightcurves as a dataframe"""

    ## grab the random lightcurves
    # lc_paths = select_lightcurves(path)
    print(os.path.basename(path))

    data, lc_info = import_XRPlightcurve(path, sector=args.sector, return_type="pandas")

    depth = 10 ** np.random.uniform(-4, -2, 1)[0]
    time_range = data["time"][
        data["time"].between(
            data["time"].min() + 1, data["time"].max() - 1, inclusive=False
        )
    ].reset_index(
        drop=True
    )  # resets index so consistency is kept when working with indices
    injected_time_index, injected_time = random.choice(list(enumerate(time_range)))

    ## comet model
    data["injected_dip_flux"] = data["corrected flux"] * (
        1
        - comet_curve(
            data["time"], depth, injected_time, 3.02715600e-01, 3.40346173e-01
        )
    )

    data = Table.from_pandas(data)
    data = data[["time", "injected_dip_flux", "quality", "flux error"]]
    results, data_arrays = processing(data, path, lc_info, method="median")
    if save_arrays:
        try:
            os.makedirs("injection_recovery_data_arrays/")
        except FileExistsError:
            pass
        try:
            np.savez(
                f"injection_recovery_data_arrays/lc_info[0].npz",
                time=data_arrays[0],
                flux=data_arrays[1],
                trend_flux=data_arrays[2],
                quality=data_arrays[3],
            )
        except:
            np.savez(
                f"injection_recovery_data_arrays/{i}.npz",
                time=data_arrays[0],
                flux=data_arrays[1],
                quality=data_arrays[2],
            )
    results = results.split()
    recovered_time = float(results[2])
    new_depth = float(results[7])

    data = data.to_pandas()
    recovered_range = data.time[
        data.time.loc[data.time == injected_time].index[0]
        - 5 : data.time.loc[data.time == injected_time].index[0]
        + 5
    ].reset_index(drop=True)

    try:
        percentage_change = (
            (abs(new_depth) - depth) / depth
        ) * 100  # 0-depth to consider normalisation
    except ZeroDivisionError:
        percentage_change = 0

    if (recovered_range.values[0] <= recovered_time <= recovered_range.values[-1]) & (
        abs(percentage_change) <= 25
    ):
        recovered = 1

    else:
        recovered = 0

    try:
        os.makedirs("injection_recovery_reformatted")
        print("created directory injection_recovery_reformatted/")
    except FileExistsError:
        pass

    lock.acquire()
    with open(
        os.path.join("injection_recovery_reformatted/", f"sector_{args.sector}_tmag_{args.mag_lower}_{args.mag_higher}.txt"), "a"
    ) as out_file2:
        ### output format is file, "signal", "snr", "time", "asym_score", "width1", "width2", "duration", "depth", "peak_lspower", "mstat", "transit_prob", "recovered", "injected_time", "injected_depth", "magnitude"
        out_file2.write(
            results
            + " "
            + " "
            + recovered
            + injected_time
            + " "
            + depth
            + " "
            + lc_info[3]
            + "\n"
        )
    lock.release()


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=args.threads)
    sample = new_select_lightcurves(args.mag_lower, args.mag_higher)
    for path in paths:
    #     print("path is", path)
        
    #     new_list = [path + '/' + x for x in sample]
    #     pool.map(run_injection2, new_list)

        if not os.path.isdir(path):
            if os.path.isfile(path):
                print(f"running search for file {path}")
                results = run_injection(path)

        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the injected search...")

            results = run_injection(path)

        else:
            print("path is ", path)

            run_injection(path)
            
