#!/usr/bin/env python3
# First have to disable inbuilt multithreading for performance reasons.
import os
os.environ['OMP_NUM_THREADS']='1'
from analysis_tools_cython import *
from tess_tools import *
import multiprocessing
import sys
import traceback
import argparse
import tqdm
import data
import loaders
import warnings
warnings.filterwarnings("ignore")


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

# def process_tess_file(f_path):
#     try:
#         f = os.path.basename(f_path)
#         table,lc_info = import_XRPlightcurve(f_path)[0],import_XRPlightcurve(f_path)[1]
#         if len(table) > 120:

#             table['normalised PCA'] = normalise_lc(table['PCA flux'])
#             bad_times = data.load_bad_times()
#             bad_times = bad_times - 2457000
#             mad_df = data.load_mad()

#             #sec = 6 #int(input("Sector? ")) # int(os.path.basename(args.path[0]).split('_')[2])  # eleanor lightcurve gives sector number in filename
#             cam = lc_info[4]
#             mad_arr = mad_df.loc[:len(table) - 1, f"{sec}-{cam}"]
#             sig_clip = sigma_clip(mad_arr,sigma=3,masked=False)
#             med_sig_clip = np.nanmedian(sig_clip)
#             rms_sig_clip = np.std(sig_clip)
#             mad_cut = mad_arr.values<(med_sig_clip + 4*(np.std(sig_clip))) # using 4 sigma threshold
#             mask = np.ones_like(table['time'], dtype=bool)
#             for i in bad_times:
#                 newchunk = (table['time'] < i[0]) | (table['time'] > i[1])
#                 mask = mask & newchunk

#             new_lc = table[(table['quality'] == 0) & mask & mad_cut]  # applying mad cut to lightcurve
#             to_clean = remove_zeros(new_lc)
#             to_clean = to_clean['time', 'PCA flux', 'quality']
#             t, flux, quality, real = clean_data(to_clean)
#             #t,flux,quality,real = clean_data(table)
#             flux = normalise_flux(flux)
#             lombscargle_filter(t,flux,real,0.05)
#             flux = flux*real
#             T = test_statistic_array(flux,60)

#             Ts = nonzero(T).std()
#             m,n = np.unravel_index(T.argmin(),T.shape)
#             Tm = T[m,n]
#             Tm_time = t[n]
#             Tm_duration = m*calculate_timestep(table)
#             Tm_start = n-math.floor((m-1)/2)
#             Tm_end = Tm_start + m
#             Tm_depth = flux[Tm_start:Tm_end].mean()

#             asym, width1, width2 = calc_shape(m,n,t,flux)
#             s = classify(m,n,real,asym)

#             result_str =\
#                     f+' '+\
#                     ' '.join([str(round(a,8)) for a in
#                         [Tm, Tm/Ts, Tm_time,
#                         asym,width1,width2,
#                         Tm_duration,Tm_depth]])+\
#                     ' '+s
#         else:
#             result_str = f+' 0 0 0 0 0 0 0 0 notEnoughData'

#         lock.acquire()
#         with open(args.of,'a') as out_file:
#             out_file.write(result_str+'\n')
#         lock.release()
#     except (KeyboardInterrupt, SystemExit):
#         print("Process terminated early, exiting",file=sys.stderr)
#         raise
#     except Exception as e:
#         print("\nError with file "+f_path,file=sys.stderr)
#         traceback.print_exc()


if __name__ == '__main__':
    sec = int(input("Sector? "))
    pool = multiprocessing.Pool(processes=args.threads)


    for path in tqdm.tqdm(paths):
        if not os.path.isdir(path):
            print(path,'not a directory, skipping.',file=sys.stderr)
            continue

        fits_files = [f for f in tqdm.tqdm(os.listdir(path)) if f.endswith('.fits')]
        file_paths = [os.path.join(path,f) for f in tqdm.tqdm(fits_files)]
        pool.map(process_file,file_paths)


        # pkl_files = [f for f in tqdm.tqdm(os.listdir(path)) if f.endswith('.pkl')]
        # file_paths_pkl = [os.path.join(path, f) for f in tqdm.tqdm(pkl_files)]
        # pool.map(process_tess_file,file_paths_pkl)

       ## for i in glob.glob('tesslcs_sectors_*_104/)
