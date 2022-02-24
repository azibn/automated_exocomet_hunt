from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from analysis_tools_cython import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Return TIC's with RA and DEC")
parser.add_argument(help="output file from batch_analyse.py", nargs=1, dest="txt_file")

args = parser.parse_args()

def get_output(file_path):
    """Imports batch_analyse output file as pandas dataframe."""
    with open(file_path) as f:
        lines = f.readlines()
    lc_lists = [word for line in lines for word in line.split()]
    lc_lists = [lc_lists[i:i+10] for i in range(0, len(lc_lists), 10)]
    cols = ['file','signal','signal/noise','time','asym_score','width1','width2','duration','depth','transit_prob']
    df = pd.DataFrame(data=lc_lists,columns=cols)
    df[cols[1:-1]] = df[cols[1:-1]].astype('float32')
    return df

def filter_df(df,min_asym_score=1.0,max_asym_score=2.0,duration=0.5,signal=-5.0):
    """filters df for given parameter range.
    Default settings:
    - `signal/noise` greater than 5.
        - Minimum test statistic is always negative. We flip the sign in plots for convenience.
    - `duration` set to greater than 0.5 days.
    - `asym_score` between 1.00 to 2.0.
    """
    return df[(df.duration >= duration) & (df['signal/noise'] <= signal) & (df['asym_score'] >= min_asym_score) & (df['asym_score'] <= max_asym_score)]

def distribution(x,y):
    """plots asymmetry score vs signal/noise over a signal of 5"""
    fig,ax = plt.subplots(figsize=(10,7))
    ax.scatter(x,y,s=1)
    ax.set_xlim(-1,1.9)
    ax.set_ylim(5,30)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$S$')
    fig.tight_layout()


if __name__ == '__main__':
    sector = 6
    path = '/storage/astro2/phrdhx/tesslcs'
    df =  get_output(args.txt_file[0])

    for i in tqdm(df.file):
        file_paths = glob.glob(os.path.join(path,f'**/**/{i}'))[0]
        ref = pd.read_pickle(glob.glob(os.path.join(path,f'**/**/{i}'))[0])
        table = import_XRPlightcurve(file_paths,sector=sector,drop_bad_points=True)[0] # drop_bad_points is True
        store = import_XRPlightcurve(file_paths,sector=sector,drop_bad_points=True)[1]
        tic = store[0]
        ra = (store[1]) # this needs to be multiplied by u * degree if you want to use SkyCoord
        dec = (store[2]) # this also needs to be multiplied by u * degree if you want to use SkyCoord

        result_str = [tic,ra,dec]
        with open('tic_info.txt','a') as output:
            output.write(str(result_str)+"\n")