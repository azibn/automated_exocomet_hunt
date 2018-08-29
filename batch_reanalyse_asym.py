#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from astropy.table import Table
import glob
import multiprocessing
import sys
import traceback
import argparse


parser = argparse.ArgumentParser(description='Analyse lightcurves listed in output file.')
parser.add_argument(help='Output file(s)',nargs='+',dest='files')

parser.add_argument('-t', help='number of threads to use',default=1,
                        dest='threads',type=int)

parser.add_argument('-d',default='.',dest='dataroot',help='Base path to light curves')

parser.add_argument('-o',default='output.txt',dest='of',help='output file')

parser.add_argument('-q', help='Keep only points with SAP_QUALITY=0',action='store_true')

# Get directories from command line arguments.
args = parser.parse_args()

# get list of light curve files from file
files = []
for f in args.files:
    data = Table.read(f, format='ascii')
    files += list(data['col1'])

## Prepare multithreading.
m = multiprocessing.Manager()
lock = m.Lock()


def process_file(f_path):
    try:
        f = os.path.basename(f_path)
        table = import_lightcurve(f_path, args.q)

        if len(table) > 120:
            t,flux,quality,real = clean_data(table)
            flux = normalise_flux(flux)
            lombscargle_filter(t,flux,real,0.05)
            flux = flux*real
            T = test_statistic_array(flux,60)

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



pool = multiprocessing.Pool(processes=args.threads)

paths = []
for f in files:
    try:
        p = download_lightcurve(f)
        paths.append(p)
    except:
        raise FileNotFoundError('{} undownloadable'.format(f))

pool.map(process_file, paths)

