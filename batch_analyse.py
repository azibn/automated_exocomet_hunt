#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
import multiprocessing
import sys
import traceback
import argparse


parser = argparse.ArgumentParser(description='Analyse lightcurves in target directory.')
parser.add_argument(help='target directory(s)',
                        default='.',nargs='+',dest='path')

parser.add_argument('-t', help='number of threads to use',default=1,
                        dest='threads',type=int)

parser.add_argument('-o',default='output.txt',dest='of',help='output file')

parser.add_argument('-q', help='Keep only points with SAP_QUALITY=0',action='store_true')

# Get directories from command line arguments.
args = parser.parse_args()

paths = []
for path in args.path:
    paths.append( os.path.expanduser(path) )

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

for path in paths:
    if not os.path.isdir(path):
        print(path,'not a directory, skipping.',file=sys.stderr)
        continue

    fits_files = [f for f in os.listdir(path) if f.endswith('.fits')]
    file_paths = [os.path.join(path,f) for f in fits_files]

    pool.map(process_file,file_paths)

