import argparse
import os
import multiprocessing
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
    default=2000,
    type=int,
)
parser.add_argument(
    "-percentage",
    help="Percentage of directory to randomise. input as decimal (i.e: 10% == 0.1)",
    dest="percentage",
    default=0.2,
)
parser.add_argument("-sector", help="lightcurve sector", dest="sector")
parser.add_argument("-save_csv", dest="save_csv", action="store_false")
parser.add_argument("-mag_lower", dest="mag_lower", help="lower magnitude", type=int)
parser.add_argument("-mag_higher", dest="mag_higher", help="higher magnitude", type=int)
parser.add_argument("-use_noiseless",dest = "use_noiseless",help="convert lightcurves to noiseless ones", action="store_true")
parser.add_argument("-drop_bad_points",dest = "drop_bad_points",help="cut out flagged data points", action="store_true")
parser.add_argument("-method", dest = "method", help="select smoothing method. Default is median",default='median')
parser.add_argument("-percentage_threshold", dest='percentage_threshold',default=100,type=int,help='specify the percentage change in depth for a "successful" recovery. Default is 100%')

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
    """uses lookup data provided by the XRP to select lightcurves within a certain magnitude range.""" 

    return (
        lookup.Filename[
            (lookup.Magnitude >= mag_lower) & (lookup.Magnitude <= mag_higher)
        ]
        .sample(args.number,replace=True)
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


def run_injection(path, save_csv=args.save_csv,use_noiseless_lightcurves=args.use_noiseless,drop_bad_points=args.drop_bad_points):
    """returns lightcurves as a dataframe
    
    :use_noiseless_lightcurves: converts all injected lightcurves to "noiseless" ones by resetting them to 1. Only done for diagnostic purposes.
    """

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
    percentage_changes = []
    injected_asym_score = []
    for i in lc_paths:
        print(i)
        filename.append(i)
        data, lc_info = import_XRPlightcurve(
            os.path.join(path, i), sector=args.sector, return_type="pandas", drop_bad_points=drop_bad_points
        )

        depth = 10 ** np.random.uniform(-4, -2, 1)[0]
        #tail = np.random.uniform(0.2,1,1)[0] ## if tail lengths want to be tested
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
        comet = 1 - comet_curve(data["time"], depth, injected_time, 3.02715600e-01, 3.40346173e-01)
        data["injected_dip_flux"] = data["corrected flux"] * comet

        if use_noiseless_lightcurves:
            if not drop_bad_points:
                ### pure noiseless lightcurves (no data gaps and ramps considered)
                time = np.linspace(1469,1490,1000)
                flux = (np.ones(1000))
                quality = np.zeros(len(flux))
                flux_error = np.zeros(len(flux))
                data = pd.DataFrame([time,flux,quality,flux_error]).T
                
                columns = ['time','flux','quality','flux error']
                data.columns = columns
                time_range = data["time"][
                data["time"].between(
                    data["time"].min() + 1, data["time"].max() - 1, inclusive=False
                )
                ].reset_index(
                    drop=True
                )  # resets index so consistency is kept when working with indices
                _ , injected_time = random.choice(list(enumerate(time_range)))
                comet = 1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)
                data["injected_dip_flux"] = (data["flux"] * comet) - 1
                #data['injected_dip_flux'] = (np.ones(len(data['corrected flux'])) * comet) - 1 ## needs to be normalised to zero for T-statistic

            else:
                ### noiseless lightcurves with realistic time coverage (MAD cuts and non-zero quality flags)
                time = data['time']
                comet = 1 - comet_curve(time, depth, injected_time, 3.02715600e-01, 3.40346173e-01)

                ## this is done because Wotan normalises flux if a smoothing filter is specified, so for ease, scaled up flat (noiseless) flux array to median flux of the lightcurve.
                if args.method == None:  
                    data["injected_dip_flux"] = (np.ones(len(data['corrected flux'])) * comet) - 1
                else:
                    data["injected_dip_flux"] = ((np.ones(len(data['corrected flux'])) * comet) * np.median(data['corrected flux'])) 

        ## converted to astropy table 
        data_to_process = Table.from_pandas(data)
        data_to_process = data_to_process[["time", "injected_dip_flux", "quality", "flux error"]]

        ## the meat of the search is done by this function
        results, data_arrays = processing(data_to_process, i, lc_info, method=args.method,noiseless=use_noiseless_lightcurves)

        ## save the lightcurve data of the search to this directory
        #try:
        #    os.makedirs(f"/storage/astro2/phrdhx/tesslcs/injection_recovery_data_arrays_10000perbin_noiseless/")
        #except FileExistsError:
        #    pass

        #np.savez(
        #    f"/storage/astro2/phrdhx/tesslcs/injection_recovery_data_arrays_10000perbin_noiseless/{lc_info[0]}.npz",
        #    original_time = data_to_process[data_to_process.colnames[0]],
        #    original_flux = data_to_process[data_to_process.colnames[1]],
        #    time=data_arrays[0],
        #    flux=data_arrays[1],
        #    #trend_flux=data_arrays[2],
        #    quality=data_arrays[2],
        #)

        ## making sense of the output from `processing`
        results = results.split()
        recovered_time = float(results[3])
        new_depth = float(results[8])
        recovered_depth.append(new_depth)
        results_for_binning.append(results)

        ## convert back to pandas to make advantage of pandas functions
        ### create a time range that is +- 2.5 hours from the injected time (total 5 hour window)
        data = data_to_process.to_pandas()
        recovered_range = data.time[
            data.time.loc[data.time == injected_time].index[0]
            - 5 : data.time.loc[data.time == injected_time].index[0]
            + 5
        ].reset_index(drop=True)

        ## defining criteria for successful recovery: within 100% change in depth, and within the timeframe
        try:
            percentage_change = (
                (abs(new_depth) - depth) / depth
            ) * 100  # 0-depth to consider normalisation
        except ZeroDivisionError:
            percentage_change = 0

        if (
            recovered_range.values[0] <= recovered_time <= recovered_range.values[-1]
        ) & (abs(percentage_change) <= args.percentage_threshold):
            recovered = 1
            recovered_or_not.append(recovered)
            percentage_changes.append(percentage_change)
        else:
            recovered = 0
            recovered_or_not.append(recovered)
            percentage_changes.append(percentage_change)
            
    try:
        df = pd.DataFrame(results_for_binning)
        df.insert(0, "file", filename)
        df["recovered"] = recovered_or_not
        df["magnitude"] = mags
        df["injected_depths"] = injected_depths
        df["injected_time"] = injected_times
        df["percentage_change"] = percentage_changes
        df.depth = [float(i) for i in df.depth]
    except:
        pass

    if args.save_csv:
       
        try:
            os.makedirs(f"injection_recovery_{args.percentage_threshold}percent_1halfday")
            print(f"created directory injection_{args.percentage_threshold}percent_1halfday")
        except FileExistsError:
            pass

        try:
            os.makedirs(f"injection_recovery_{args.percentage_threshold}percent_1halfday/sector_{args.sector[0]}")
            print(f"created directory injection_{args.percentage_threshold}percent_1halfday/sector_{args.sector[0]}")
        except FileExistsError:
            
            pass

        try:
            df.to_csv(
                f"injection_recovery_{args.percentage_threshold}percent_1halfday/sector_{args.sector[0]}/tmag_{args.mag_lower}_tmag_{args.mag_higher}.csv"
            )
            print(f"file tmag_{args.mag_lower}_tmag_{args.mag_higher}.csv saved.")
        except FileExistsError:
            pass

    return df

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
            try:
                run_injection(path)
            except:
                ### this is only because of some files that got deleted (so the lookup file has lightcurves that don't "exist" in our folders)
                lookup = pd.read_csv(f"/storage/astro2/phrdhx/tesslcs/sector{args.sector}lookupv2.csv")
                run_injection(path)
                print("imported lookupv2")

            
