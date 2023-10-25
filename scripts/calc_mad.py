from astropy.stats import median_absolute_deviation
from loaders import load_lc, load_scd_subset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from astropy.io import fits



def load_fluxes(f):
    hdul = fits.open(f)
    flux = hdul[1].data['CORR_FLUX']
    quality = hdul[1].data['QUALITY']
    flux[quality != 0] = np.NaN
    flux = flux/np.nanmedian(flux)
    return flux


def calc_mad(
    lc_set_filenames,
    sector=1,
    camera=1,
    ccd=1,
    mag_min=0,
    mag_max=15,
    output_file="cadence_mad.csv"
):

    # =========================================
    # COLLECTING LIGHT CURVES INTO A SINGLE DATAFRAME
    # =========================================
    lcs_df = pd.DataFrame()
    times_df = pd.DataFrame()
    from multiprocessing import Pool
    with Pool(40) as p:
        fluxes = p.map(load_fluxes, lc_set_filenames)
    fluxes = np.array(fluxes)

    cadence_mad = median_absolute_deviation(fluxes, axis=0)
    cadence_mad.to_csv(output_file, index=False) 

    if __name__ == "__main__":
        plt.scatter(times_df.median(), cadence_mad)
        plt.xlabel("Time (days BJD)")
        plt.ylabel("MAD")
        plt.title(
            f"Sector {sector} Camera {camera} Detector {ccd} - {mag_min} to {mag_max}")
        wandb.log({"MAD Plot": wandb.Image(plt)})
        data = [[x, y] for (x, y) in zip(times_df.T.median(), cadence_mad)]
        table = wandb.Table(data=data, columns=["Time (days BJD)", "MAD"])
        wandb.log({"MAD Data": table})
        pass

    else:
        return cadence_mad
