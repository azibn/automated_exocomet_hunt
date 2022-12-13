import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import random 
import matplotlib.pyplot as plt
import traceback
from analysis_tools_cython import *
from astropy.table import Table

parser = argparse.ArgumentParser(description="Injection Testing")

parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")
parser.add_argument("-s",help="sector", dest="sector",default=6,type=int)#,default=f'{args.path[0].split('sector')[1].split('')}')
parser.add_argument("-tmag_low",help="brighter tmag", dest="tmag_low")
parser.add_argument("-tmag_high",help="fainter tmag", dest="tmag_high")
parser.add_argument(
    "-t", help="number of threads to use", default=1, dest="threads", type=int
)

#parser.add_argument("-o", default=f"injected_output.txt", dest="of", help="output file")
parser.add_argument("-number", help="How many lightcurves to randomise. default is 100", dest="number",default=100,type=int)
parser.add_argument("-sector",help="lightcurve sector", dest = "sector",default="6")
#parser.add_argument("-savefigs",dest = "savefigs",action="store_true")

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

def select_lightcurves(path):
    return random.sample(os.listdir(path),args.number)

def run_search(path):
    """returns lightcurve as a dataframe"""

    lc_paths = select_lightcurves(path)

    recovered_or_not = []
    for i in lc_paths:
        data, lc_info = import_XRPlightcurve(os.path.join(path,i), sector=args.sector,return_type='pandas')

        ## scale factor
        factor = round(random.uniform(0.1,10),1)

        ## want the range of transits to be within 1 day from beginning of lightcurve, and one day from the end of the lightcurve
        time_range = data['time'][data['time'].between(data['time'].min()+1,data['time'].max()-1, inclusive=False)].reset_index(drop=True) # resets index so consistency is kept when working with indices
        injected_time_index, injected_time = random.choice(list(enumerate(time_range)))

        ## multiplying comet depth by scale factor
        depth = factor * 1.86164653e-03 

        ## comet model (based of Beta Pictoris in Zieba et. al 2019)
        model = data['corrected flux']*(1-comet_curve(data['time'],depth, injected_time, 3.02715600e-01, 3.40346173e-01))

        ## setting up plotting
        plt.figure(figsize=(13,5))
        plt.plot(data['time'],data['corrected flux'],label='original lightcurve (no model)')
        plt.plot(data['time'],normalise_flux(model),label='injected lightcurve (w/ model, pre-processed')

        data.drop('corrected flux',axis=1,inplace=True)
        data.insert(1, "corrected flux", model) # re-injects the flux, but with the model
        data = data[['time','corrected flux','quality']]

        ## if data is pandas dataframe convert to astropy table
        try:
            data = Table.from_pandas(data)
        except:
            pass
        result_str, data_arrays = processing(data,path,lc_info=lc_info,method='median')

        results = result_str.split() 

        ## processed lightcurve arrays
        new_time_array = data_arrays[0]
        new_flux_array = data_arrays[1]

        ## plot new lightcurves
        plt.plot(new_time_array, new_flux_array,label='recovered lightcurve (w/ model, post-processed)')
        plt.title(f"TIC {lc_info[0]}")

        try:
            os.makedirs("injection_testing_plots")
        except FileExistsError:
            pass

        try:
            os.makedirs(f"injection_testing_plots/tmag_{path.split('_')[-2]}_tmag{path.split('_')[-1]}")
        except FileExistsError:
            pass

        plt.savefig(f"injection_testing_plots/tmag_{path.split('_')[-2]}_tmag{path.split('_')[-1]}/TIC{lc_info[0]}",dpi=350)
        plt.legend()
        plt.close()

        ## the chosen range to recover the transit signal is 5 hours (2.5 hours either side).
        data = data.to_pandas()
        recovered_range = data.time[data.time.loc[data.time == injected_time].index[0]-5:data.time.loc[data.time == injected_time].index[0]+5].reset_index(drop=True)

        ## preparing pre-processed lightcurves for plotting
        normalised_flux = normalise_flux(data['corrected flux'])

        ## accepted percentage change not more than 50%
        percentage_change = (((0-depth) - float(results[7]))/depth) * 100 # 0-depth to consider normalisation

        ## if recovered value is in range save this variable
        if (recovered_range.values[0] <= float(results[2]) <= recovered_range.values[-1]) & (abs(percentage_change) <= 50.99):
            print(f"TIC {lc_info[0]} success, time was {float(results[2])}, percentage change was {round(percentage_change,2)}%")
            recovered = 1
            recovered_or_not.append(recovered)
        else:
            print(f" TIC {lc_info[0]} injection failed. recovered time was {float(results[2])}, injected time was {injected_time}. percentage change was {round(percentage_change,2)}%")
            recovered = 0
            recovered_or_not.append(recovered)
    print(recovered_or_not.count(1), f"out of {len(recovered_or_not)} ligthcurves have recovered transits in the timeframe with an acceptable change in depth.")
    fraction = round((recovered_or_not.count(1))/len(recovered_or_not) * 100,1)
    print(fraction, f"% of lightcurves recovered in between {path.split('_')[-2]} and {path.split('_')[-1].split('/')[0]} magnitudes.")


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=args.threads)
    for path in paths:

        if not os.path.isdir(path):
            if os.path.isfile(path):
                run_search(path)
                sys.exit()

        # if we are in the lowest subdirectory, perform glob this way.
        if not list(folders_in(path)):
            print("this is the lowest subdirectory. running the injected search...")
        
            run_search(path)
            
            #pkl = glob.glob(os.path.join(path, "*.pkl"))
   
            #pool.map(run_search, path)

        else:
            print("globbing subdirectories")

            pkl = glob.glob(os.path.join(path,"**/*.pkl"))
            

            print("running the search...")

            pool.map(run_search, pkl)
    
