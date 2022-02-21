#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os
#import pandas as pd
os.nice(8)
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
import multiprocessing
import sys
import traceback
import argparse
import tqdm
import data
import glob
import loaders
import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='Analyse lightcurves in target directory.')
parser.add_argument(help='target directory(s)',
                        default='.',nargs='+',dest='path')

parser.add_argument('-t', help='number of threads to use',default=1,
                        dest='threads',type=int)

parser.add_argument('-o',default=f'output.txt',dest='of',help='output file')

parser.add_argument('-q', help='Keep only points with SAP_QUALITY=0',action='store_true')

# Get directories from command line arguments.
args = parser.parse_args()

paths = []
for path in args.path:
    paths.append( os.path.expanduser(path) )

## Prepare multithreading.
multiprocessing.set_start_method("fork") # default for >=3.8 is spawn
m = multiprocessing.Manager()
lock = m.Lock()


def process_file(f_path):
    try:
        f = os.path.basename(f_path)
        print(f)
        table = import_lightcurve(f_path, args.q)

        if len(table) > 120:

            t,flux,quality,real = clean_data(table)
            timestep = calculate_timestep(table)

            factor = ((1/48)/timestep)
            flux = normalise_flux(flux)
            lombscargle_filter(t,flux,real,0.05)
            flux = flux*real
            T = test_statistic_array(flux,60 * factor)

            Ts = nonzero(T).std()
            m,n = np.unravel_index(T.argmin(),T.shape)
            Tm = T[m,n]
            Tm_time = t[n]
            Tm_duration = m*calculate_timestep(table) # <- can change this to `timestep` as it's already defined
            Tm_start = n-math.floor((m-1)/2)
            Tm_end = Tm_start + m
            Tm_depth = flux[Tm_start:Tm_end].mean()

            asym, width1, width2 = calc_shape(m,n,t,flux)
            s = classify(m,n,real,asym)

            result_str =\
                    f+' '+\
                    ' '.join([str(round(a,8)) for a in
                        [Tm, Tm/Ts, Tm_time,
                        asym,width1,width2,
                        Tm_duration,Tm_depth]])+\
                    ' '+s
        else:
            result_str = f+' 0 0 0 0 0 0 0 0 notEnoughData'

        lock.acquire()
        with open(args.of,'a') as out_file:
            out_file.write(result_str+'\n')
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting",file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file "+f_path,file=sys.stderr)
        traceback.print_exc()

def process_tess_file(f_path):
    try:
        f = os.path.basename(f_path)
        print(f_path)
        table= (import_XRPlightcurve(f_path,sector=sector_test,clip=3)[0])
        #print(table)
        if len(table) > 120:
            to_clean = table["time", "corrected flux", "quality"]
            t, flux, quality, real = clean_data(to_clean)
           
            timestep = calculate_timestep(table)
            flux = normalise_flux(flux)
            factor = ((1/48)/timestep)
            lombscargle_filter(t,flux,real,0.05)
            flux = flux*real
            T = test_statistic_array(flux,60 * factor)

            Ts = nonzero(T).std()
            m,n = np.unravel_index(T.argmin(),T.shape)
            Tm = T[m,n]
            Tm_time = t[n]
            Tm_duration = m*calculate_timestep(table)
            Tm_start = n-math.floor((m-1)/2)
            Tm_end = Tm_start + m
            Tm_depth = flux[Tm_start:Tm_end].mean()

            asym, width1, width2 = calc_shape(m,n,t,flux)
            s = classify(m,n,real,asym)

            result_str =\
                    f+' '+\
                    ' '.join([str(round(a,8)) for a in
                        [Tm, Tm/Ts, Tm_time,
                        asym,width1,width2,
                        Tm_duration,Tm_depth]])+\
                    ' '+s
        else:
            result_str = f+' 0 0 0 0 0 0 0 0 notEnoughData'

        lock.acquire()
        with open(args.of,'a') as out_file:
            out_file.write(result_str+'\n')
        lock.release()
    except (KeyboardInterrupt, SystemExit):
        print("Process terminated early, exiting",file=sys.stderr)
        raise
    except Exception as e:
        print("\nError with file "+f_path,file=sys.stderr)
        traceback.print_exc()


def folders_in(path_to_parent):
    # Identifies if directory is the lowest directory to perform search
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)

if __name__ == '__main__':
    sector_test = int(input("sector? "))
    pool = multiprocessing.Pool(processes=args.threads)


    for path in paths:
        if not os.path.isdir(path):
            print(path,'not a directory, skipping.',file=sys.stderr)
            continue
        
        # if we are in the lowest subdirectory, perform glob this way.
        if not list(folders_in(path)):
            print("this is the lowest subdirectory")
            # this should work for both Kepler and TESS fits files.
            fits_files = glob.glob(os.path.join(path,'*lc.fits'))#[f for f in os.listdir(path) if f.endswith('lc.fits')] 
            pkl_files = glob.glob(os.path.join(path,'*.pkl'))
 
            pool.map(process_file, fits_files)
            pool.map(process_tess_file,pkl_files)

        else:
            print("globbing subdirectories")
            #if "SPOC" in  os.getcwd():
            # Start at Sector directory, glob goes through `target/000x/000x/xxxx/**/*lc.fits`
            fits_files = glob.glob(os.path.join(path,'target/**/**/**/**/*lc.fits')) # 
            # Starts at sector directory. globs files in one subdirectory level below
            pkl_files = glob.glob(os.path.join(path,'**/*.pkl'))

            pool.map(process_file,fits_files)
            pool.map(process_tess_file,pkl_files)

    
        # file_paths = [os.path.join(path,f) for f in tqdm.tqdm(fits_files)]
        # pool.map(process_file,file_paths)

        #fits_test = 
        # pkl_files = [f for f in tqdm.tqdm(os.listdir(path)) if f.endswith('.pkl')]
        # file_paths_pkl = [os.path.join(path, f) for f in tqdm.tqdm(pkl_files)]
        # pool.map(process_tess_file,file_paths_pkl)
       
        