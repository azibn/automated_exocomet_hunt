import os
import sys
import re
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

from som.TransitSOM_release import CreateSOM


"""
This script runs a Kohonen Self-Organising Map (SOM) for a given set of lightcurve candidates.

By default, the script saves two plots: one of the Kohonen layer, and one of the lightcurves mapped onto each pixel of the Kohonen layer. It also saves the Kohonen layer as a file, 
so that you can return to your favourite Kohonen layer later.


Usage:
- The way the script works is that it first reads in a directory of lightcurve candidates created from `batch_analyse`. It is assumed that you have run the `som_cutouts=True` argument
function to create the cutouts of lightcurve transits. These cutouts are then stacked into a 2D array, and the SOM is run on this array.
- As the SOM requires equally sized data points, and depending on how large of a cutout you specified when creating these `.npz` files, some lightcurves may be interpolated.
- `batch_analyse.py` saves these cutouts at their original flux level. There are currently two options to normalise the data: by median, or by depth. The default is median.
- The SOM is initialised with 1000 iterations. This can be changed with the `-i` argument.

"""



parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument("-ps", "--plots-save", help="save plots. False by default.", action="store_true", dest="save") ### true by default
parser.add_argument("-kname", "--kohonen-name", help="Name of output kohonen data file.", dest="outfile",default=None) ### true by default


parser.add_argument(
    "-shape",
    "--somshape",
    help="som grid size. Enter the somshape in the form '(x,y)'",
    dest="shape",
)

parser.add_argument(
    "-i", "--iterations", help="number of iterations", default=1000, dest="iterations",type=int
)
parser.add_argument("-as", "--array-save", help="save input arrays", action="store_true", dest="array_save") ### true by default
parser.add_argument("-array-save-name", "--array-save-name", help="name to give SOM input arrays. Please save as `.npz` file, otherwise will not save.", dest="array_save_name") ### true by default

# Get directories from command line arguments.
args = parser.parse_args()


def stack_npz_files(directory):
    
    """normalisation method: median, depth, depth and width"""
    
    files = os.listdir(directory)
    npz_files = [f for f in files if f.endswith('.npz')]
    if not npz_files:
        print("No .npz files found in the directory.")
        return None

    normalised_by_median = []
    normalised_by_depth = [] 
    ids = []
    
    for npz_file in tqdm(npz_files):
        file_path = os.path.join(directory, npz_file)
        data = np.load(file_path)   

        
        ## background subtraction
        x1 = np.median(data['flux'][0:12])
        x2 = np.median(data['flux'][-13:-1]) # the last 12 points

        y1 = np.median(data['time'][0:24])
        y2 = np.mean(data['time'][-25:-1])
        grad = (x2-x1)/(y2-y1)
        background_level = x1 + grad * (data['time'] - y1)
        original_flux = data['flux'].copy()
        
        flux = original_flux - background_level

        obj_id = data['id']
        
        ## normalisation method
        
        #depth_normalised_flux = remove_depth_and_normalize(flux)
        depth_normalised_flux = normalise_depth(flux)

        
        
        if len(depth_normalised_flux) == 121:
            normalised_by_depth.append(depth_normalised_flux)
            ids.append(obj_id.item())


    stacked_depth_lcs = np.vstack(normalised_by_depth)
    #try:
        #stacked_ids = [id.encode('utf-8') for id in ids]
    #except:
    stacked_ids = ids

    # Create a dictionary to map arrays to IDs
    id_map = {id: tuple(array) for array, id in zip(stacked_depth_lcs, stacked_ids)}

    ## save array
    if args.array_save:
        np.savez('{}'.format(args.array_save_name),array=stacked_depth_lcs,ids=ids)
    else:
        np.savez('som_input_arrays_{}x{}-{}.npz'.format(somshape[0],somshape[1],stacked_depth_lcs.shape[0]),array=stacked_depth_lcs,ids=ids)
    print("input arrays saved as 'som_input_arrays_{}x{}-{}.npz'".format(somshape[0],somshape[1],stacked_depth_lcs.shape[0]))

    return stacked_depth_lcs, stacked_ids, id_map

def remove_depth_and_normalize(flux):
    min_flux = np.min(flux)
    max_flux = np.max(flux)
    
    # Remove depth and normalize
    depth_removed_normalized_lightcurve = (flux - min_flux) / (max_flux - min_flux)
    
    return depth_removed_normalized_lightcurve


def normalise_depth(flux):
    median = np.median(flux)
    #depth_normalised_lightcurve = (flux - median) / median
    abs_depth = median - np.min(flux)  # Assuming the minimum of the lightcurve is the minimum point
    depth_normalised_lightcurve = ((flux - median) / abs_depth + 1)
    return depth_normalised_lightcurve

def plot_all_arrays(som, length,bins=np.arange(121),save=args.save):
    fig, axes = plt.subplots(somshape[0], somshape[1], figsize=(50, 50))
    fig.subplots_adjust(
        wspace=0.4, hspace=0.4
    )  # Adjust the width and height spacing between subplots

    for x_pixel in range(somshape[0]):
        for y_pixel in range(somshape[1]):
            ax = axes[x_pixel, y_pixel]
            ax.scatter(bins, som[x_pixel, y_pixel], c="g", s=10)
            ax.set_title("Kohonen pixel at [{},{}]".format(x_pixel, y_pixel))
            # ax.text(0, 0.92, '[{},{}]'.format(x_pixel, y_pixel))

    if args.save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/{}_{}_{}-iters_pixelview-{}.png".format(somshape[0],somshape[1],args.iterations,length))
        print("pixel view saved as som_plots/{}_{}_{}-iters_pixelview-{}.png".format(somshape[0],somshape[1],args.iterations,length))
        plt.close()        
    
###### SOM Process ######

print("stacking data")
somshape = args.shape
somshape = tuple(map(int, somshape.strip('()').split(',')))
som_array, ids, id_map = stack_npz_files(args.path[0])


print("files loaded")
print("starting som. SOM will produce a {}x{} grid, with {} iterations".format(somshape[0],somshape[1],args.iterations))

#np.savez('candidate_lightcurves.npz',**id_map)

trained_data = CreateSOM(som_array, somshape=somshape, niter=args.iterations, outfile=args.outfile)

print("training done")

if args.outfile:
    print("Kohonen layer saved as {}".format(args.outfile))

print("mapping data")
mapped = trained_data(som_array)
mapped_tuples = [tuple(point) for point in mapped]
counts = Counter(mapped_tuples)
count_list = [counts[item] for item in mapped_tuples]

som_x = somshape[0]
som_y = somshape[1]

####### Save Kohonen grid
print("saving kohonen grid")
colour = "count"  # can be 'count', 'VarGroup', False
if colour == "count":
    x_pos = mapped[:, 0]
    y_pos = mapped[:, 1]
    count_list = [counts[item] for item in mapped_tuples]
    plt.figure(figsize=(10, 7))
    plt.xlim([-1, somshape[0]])
    plt.ylim([-1, somshape[1]])
    plt.xlabel("Kohonen X axis")
    plt.xlabel("Kohonen Y axis")
    plt.scatter(x_pos, y_pos, c=count_list,s=100)
    plt.colorbar(label="# of lightcurves at pixel")
    if args.save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/{}_{}_{}-iters_map-{}.png".format(somshape[0],somshape[1],args.iterations,som_array.shape[0]))
        print("SOM grid saved as som_plots/{}_{}_{}-iters_map-{}.png".format(somshape[0],somshape[1],args.iterations,som_array.shape[0]))
    plt.close()

array = trained_data.K
plot_all_arrays(array, length=som_array.shape[0], bins=np.arange(121))
#df = pd.DataFrame(data=[ids,mapped_tuples]).T
#df.columns = ['TIC_ID','coords']
#df['TIC'] = df['TIC_ID'].apply(lambda cell: ' '.join(re.findall(r'\d+', str(cell))))
#df.to_csv("{}x{}_{}_iters_data.csv".format(somshape[0],somshape[1],args.iterations),index=False)
print("done")
