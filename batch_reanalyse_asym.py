#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from astropy.table import Table
import glob
import multiprocessing
import subprocess
import sys
import traceback
import argparse


parser = argparse.ArgumentParser(description='Analyse lightcurves listed in output file.')
parser.add_argument(help='Output file(s)',nargs='+',dest='files')

parser.add_argument('-t', help='number of threads to use',default=1,
                        dest='threads',type=int)

parser.add_argument('-d',default='.',help='Base path to light curves')

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

        # ensure lightcurve long enough and not a background pixel
        if len(table) > 120 and 'kplr1' not in f:
            t,flux,quality,real = clean_data(table)

            # now throw away interpolated points (we're reprocessing
            # and trying to get the shape parameters right)
            t = t[np.array(real,dtype=bool)]
            flux = flux[np.array(real,dtype=bool)]
            quality = quality[np.array(real,dtype=bool)]
            real = real[np.array(real,dtype=bool)]

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


def get_one(file, path=args.d):
    """Get one file."""
    kic_no = file.split('-')[0].split('kplr')[1]
    kic_part = kic_no[:4]
    file_path = path+'/data/lightcurves/{}/'.format(kic_no)

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    if not os.path.exists(file_path+file):
        subprocess.run(['wget',
                        'https://archive.stsci.edu/missions/kepler/lightcurves/{}/{}/{}'.format(kic_part,kic_no,file),
                        '-O',path+'/data/lightcurves/{}/{}'.format(kic_no,file),
                        '-nc'], check=True, stderr=subprocess.PIPE)

    return file_path+file


# get all files
pool = multiprocessing.Pool(processes=args.threads)
paths = pool.map(get_one,files)
pool.close()

# process them
pool = multiprocessing.Pool(processes=args.threads)
pool.map(process_file, paths)
pool.close()
