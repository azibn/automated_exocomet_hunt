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
from numba import jit
import sys
import glob

parser = argparse.ArgumentParser(description="Binning data and creating heatmap")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")
parser.add_argument("-o",help="output image",default="recovery_map.png",dest="o")
args = parser.parse_args()

def plot_heatmap(data):
    """
    args:
        - data: pandas dataframe of output data from injection_tests.py
    returns:
        - heatmap plot

    """
    ## the parameters you want to plot
    mag = data.mag  # x
    depth = data.depth + 1  # y

    ## dataset of recovered transits
    recovered_data = data.loc[data.recovered == 1]

    ## setting axes and bins
    ### log depths
    depth_edges = np.logspace(0.995,1,num=20)/10 ### 20 bins
    #depth_edges = np.array([-0.012, -0.01, -0.008, -0.006, -0.004, -0.002, 0]) + 1
    # depth_edges = np.array([-0.013,-0.012,-0.011,-0.01,-0.009,-0.008,-0.007,-0.006,-0.005,-0.004,-0.003,-0.002,-0.001,0])
    nbins_depth = len(depth_edges) - 1
    mag_edges = np.arange(6, 16, 1)
    nbins_mag = len(mag_edges) - 1

    ## all objects
    full_hist, full_magvals, full_depthvals = np.histogram2d(
        mag, depth, bins=[mag_edges, depth_edges]
    )

    ## recovered objects
    hist_recovered, magvals_recovered, depthvals_recovered = np.histogram2d(
        recovered_data.mag, recovered_data.depth + 1, bins=[mag_edges, depth_edges]
    )
    # hist_vet, pervals_vet, rplvals_vet = np.histogram2d(
    #    per[vet], rpl[vet], bins=[per_edges, rpl_edges])

    ## stacking arrays vertically in prep for injection-recovery plot
    pi = np.vstack(([np.array([p for r in full_depthvals]) for p in full_magvals]))
    ri = np.vstack(([np.array([r for r in full_depthvals]) for p in full_magvals]))

    ## histogram percentage
    frac_hist = hist_recovered / full_hist

    ## preparing plot
    fig, ax1 = plt.subplots(1, figsize=(16, 10), sharey=True, sharex=True)
    im = ax1.pcolormesh(pi, ri, frac_hist, vmin=0.25, vmax=0.85)
    #  fig.colorbar(im)
    for i in range(len(mag_edges) - 1):
        for j in range(len(depth_edges) - 1):
            pval = np.mean(mag_edges[[i, i + 1]])
            rval = np.mean(depth_edges[[j, j + 1]])
            hval = frac_hist[i, j]
            ax1.annotate(
                f"{hval:.2f}",
                xy=(pval, rval),  # multiply hval by 100 here for percentage
                ha="center",
                va="center",
            )

    ax1.set_ylabel("Depth (in normalised flux)", fontsize=14)
    ax1.set_yticks(depth_edges)
    ax1.set_xlabel("Magnitudes", fontsize=14)
    ax1.set_title("Injection Recovery of 500 injected Exocomet transits", fontsize=20)
    fig.colorbar(im, ax=ax1)
    plt.savefig(args.o, dpi=400)
    plt.close()

if __name__ == '__main__':
    
    cols = [
    "file",
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
    "mag"]

    li = []
    files = glob.glob("injection_test_tmag_*")
    for filename in files:
        df = pd.read_csv(filename, index_col=0)
        df.columns = cols
        li.append(df)

    data = pd.concat(li, axis=0, ignore_index=True)
    try:
        os.makedirs("injection_recovery_heatmaps")
    except FileExistsError:
        pass
    plot_heatmap(data)
    print("heatmap created.")

