import os
import sys
import argparse
import warnings
import matplotlib
matplotlib.use('Agg') 
warnings.filterwarnings("ignore")
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from som.selfsom import SimpleSOMMapper

from som.TransitSOM_release import CreateSOM, LoadSOM


def get_lightcurves(ids,mapped_tuples,pixel, directory,output_save):
    """
    This function retrieves lightcurves in the SOM pixels. 
    
    :ids: TIC IDs from `stack_npz_files`
    :mapped_tuples: Obtained from the SOM process, where this is the coordinates of the lightcurve
    :pixel: desired pixel to retrieve lightcurves
    :dir: Directory of where original `.npz` files are.
    
    outputs:
        lightcurve plots.
    
    pixel has to be in the form of (x,y) coordinates"""
    df = pd.DataFrame(data=[ids,mapped_tuples]).T
    df.columns = ['TIC','coords']
    lightcurves = df.groupby('coords').get_group(pixel).reset_index(drop=True)
    
    pdf = plt.PdfPages(output_save)

    for i in lightcurves.TIC:
        #file_pattern = os.path.join(directory, '**', f'*{number_}*')
        lc = np.load("som_cutouts_snr6/{}.npz".format(i))
        plt.subplot(1, 2, 1)
        plt.title("TIC {}".format(i))
        median = np.median(lc['flux'])
        abs_depth = median - np.min(lc['flux'])  # Assuming the minimum of the lightcurve is the minimum point
        depth_normalised_lightcurve = (lc['flux'] - median) / abs_depth + 1
        
        plt.scatter(lc['time'],depth_normalised_lightcurve,s=5)
        plt.subplot(1, 2, 2)
        plt.title("TIC {} - Original processed lightcurve".format(i))
        plt.scatter(lc['time'], lc['flux']/np.nanmedian(lc['flux']), s=5)
        plot_counter += 1

        # Save the current page with two plots when plot_counter is a multiple of 2
        if plot_counter % 2 == 0:
            pdf.savefig(plt.gcf())
            plt.close()  # Close the current figure

    # If there's an odd number of plots, save the last page
    if plot_counter % 2 != 0:
        pdf.savefig(plt.gcf())

    # Close the PDF file
    pdf.close()




som = LoadSOM(file,x,y)




