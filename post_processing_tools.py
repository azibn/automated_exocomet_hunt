import pandas as pd
import numpy as np
import glob
import os
from analysis_tools_cython import *

def get_output(file_path):
    """Imports batch_analyse output file as pandas dataframe.
    
    file_path: ouptut file (.txt format)

    Returns:
    - df: DataFrame of output file.

    """
    with open(file_path) as f:
        lines = f.readlines()
    lc_lists = [word for line in lines for word in line.split()]
    lc_lists = [lc_lists[i:i+10] for i in range(0, len(lc_lists), 10)]
    cols = ['file','signal','signal/noise','time','asym_score','width1','width2','duration','depth','transit_prob']
    df = pd.DataFrame(data=lc_lists,columns=cols)
    df[cols[1:-1]] = df[cols[1:-1]].astype('float32')
    return df


def get_lightcurves(data, storage_path,sec,mad_df,plots=False,clip=4):
    """Uses dataframe obtained from `get_output` to retrieve the xrp lightcurves of desired TIC object.
    
    Notes: `mad_df` must be called separately in the script/notebook.

    data: Pandas DataFrame 
    storage_path: path to directory where lightcurve pkl files are saved.

    Returns: 
    - Plots in 2x2 format:  
        - Top left: the MAD array 
    """
    for i in data.file:
        file_paths = glob.glob(os.path.join(storage_path,f'**/**/{i}'))[0]
        table, info = import_XRPlightcurve(file_paths,sector=sec,clip=clip,drop_bad_points=True)
        raw_table, _ = import_XRPlightcurve(file_paths,sector=sec,clip=clip,drop_bad_points=False) # want to plot raw lightcurve to see if the MAD has really worked
        if plots:
            tic = info[0]
            camera = info[4]
            mad_df = pd.read_json("./data/Sectors_MAD.json")
            mad_arr = mad_df.loc[:len(table)-1, f"{sec}-{camera}"]
            sig_clip = sigma_clip(mad_arr,sigma=clip,masked=False)
            med_sig_clip = np.nanmedian(sig_clip)
            rms_sig_clip = np.nanstd(sig_clip)
            
            #fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,figsize=(15,8))
            fig = plt.figure(figsize=(10,4))
            ax1 = fig.add_subplot(221)            
            ax1.scatter(range(0,len(table['time'])), mad_arr, s=2)
            ax1.axhline(np.nanmedian(mad_arr), c='r',label='median line')
            ax1.axhline(np.nanmedian(mad_arr)+10*np.std(mad_arr[900:950]),c='blue',label='visualised MAD') # 10 sigma threshold
            ax1.axhline(med_sig_clip + clip*(rms_sig_clip), c='black',label='sigma clipped MAD')   
            ax1.set_title(f"{sec}-{camera} at {clip} sigma")
            ax1.set_ylim([0.5*np.nanmedian(mad_arr),4*np.nanmedian(mad_arr)])
            ax1.legend()
            ax2 = fig.add_subplot(223,sharex=ax1)          
            ax2.scatter(range(0,len(raw_table['time'])), raw_table['quality'], s=5)
            ax2.set_yscale('log')
            ax2.set_title('Cadence vs bit')
            ax3 = fig.add_subplot(222)          
            ax3.scatter(raw_table['time'],normalise_lc(raw_table['corrected flux']),s=0.4)
            ax3.set_title(f'Raw lightcurve for TIC {tic}')
            ax4 = fig.add_subplot(224,sharex=ax3)          
            ax4.scatter(table['time'],normalise_lc(table['corrected flux']),s=0.4)
            ax4.set_title(f'MAD corrected lightcurve for TIC {tic}')
        

            fig.tight_layout()
            fig.suptitle(f"TIC {tic}",fontsize=16,y=1.05)

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
    ax.set_xlim(-0,1.9)
    ax.set_ylim(5,30)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$S$')
    fig.tight_layout()