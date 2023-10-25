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

parser.add_argument("-s", "--save", help="save plots. False by default.", action="store_true", dest="save") ### true by default
parser.add_argument("-kname", "--kohonen-name", help="Name of output kohonen data file.", dest="outfile",default=None) ### true by default


parser.add_argument(
    "-somshape",
    "--somshape",
    help="som grid size. Enter the somshape in the form '(x,y)'",
    dest="grid",
)

parser.add_argument(
    "-i", "--iterations", help="number of iterations", default=500, dest="iterations",type=int
)

parser.add_argument("-msname", "--mapsave-name",help="name of output SOM map.", dest = "output_mapsave")

parser.add_argument("-psname", "--pixelsave-name",help="name of output SOM map.", dest = "output_pixelsave")

parser.add_argument("-o", "--outfile",help="name of output Kohonen file.", dest = "outfile")

parser.add_argument("-norm", "--normalisation_method",help="the normalisation method for the cutouts. Default is depth normalisation.", dest = "norm", default = "depth")


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
        
               
        normalised_lightcurve = (flux)/np.median(flux)
        median = np.median(flux)
        #depth_normalised_lightcurve = (data['flux'] - median) / median
        abs_depth = median - np.min(flux)  # Assuming the minimum of the lightcurve is the minimum point
        depth_normalised_lightcurve = ((flux - median) / abs_depth + 1)
        #elif normalisation_method == 'depth and width':
        
        abs_depth_unsubtracted = np.median(original_flux) - np.min(original_flux)
        depth_normalised_lightcurve_unsubtracted = ((original_flux - np.median(original_flux)) / abs_depth_unsubtracted + 1)
        
        #fig, ax = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns of subplots
        
        ##ax[0].scatter(data['time'],depth_normalised_lightcurve,s=5)
        #ax[0].set_title("TIC {} - unsubtracted background".format(data['id']))
        #ax[1].scatter(data['time'],depth_normalised_lightcurve_unsubtracted,s=5)
        #ax[1].set_title("TIC {} - subtracted background".format(data['id']))
        #plt.title(data['id'])
        #plt.show()    
        if len(normalised_lightcurve) == 121:
            normalised_by_median.append(normalised_lightcurve)
            normalised_by_depth.append(depth_normalised_lightcurve)
            ids.append(obj_id.item())


    stacked_median_lcs = np.vstack(normalised_by_median)
    stacked_depth_lcs = np.vstack(normalised_by_depth)
    stacked_ids = ids

    # Create a dictionary to map arrays to IDs
    id_map = {tuple(array): id for array, id in zip(stacked_median_lcs, stacked_ids)}
    return stacked_median_lcs, stacked_depth_lcs, stacked_ids, id_map

def plot_all_arrays(som, bins=np.arange(121),save=args.save):
    fig, axes = plt.subplots(somshape[0], somshape[1], figsize=(50, 50))
    fig.subplots_adjust(
        wspace=0.4, hspace=0.4
    )  # Adjust the width and height spacing between subplots

    for x_pixel in range(somshape[0]):
        for y_pixel in range(somshape[1]):
            ax = axes[x_pixel, y_pixel]
            ax.scatter(bins, som[x_pixel, y_pixel], c="g", s=3)
            ax.set_title("Kohonen pixel at [{},{}]".format(x_pixel, y_pixel))
            # ax.text(0, 0.92, '[{},{}]'.format(x_pixel, y_pixel))

    if args.save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/{}".format(args.output_pixelsave))
        plt.close()


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
        
    
###### SOM Process ######

print("stacking data")

som_array, som_array_depth, ids, id_map = stack_npz_files(args.path[0])
somshape = args.grid
somshape = tuple(map(int, somshape.strip('()').split(',')))

print("files loaded")
print("starting som")


array_options = {"median":som_array, "depth":som_array_depth}

if args.norm in array_options.keys():
    trained_data = CreateSOM(array_options[args.norm], somshape=somshape, niter=args.iterations, outfile=args.outfile)
else:
    print(args.norm.values())
    print(array_options[args.norm])
    print("Invalid method for args.norm {}. Please select one from the options: 'median', or 'depth'".format(args.norm))
    sys.exit()
# if args.norm == 'median':
#     trained_data = CreateSOM(som_array, somshape=somshape, niter=args.iterations, outfile=args.outfile)
# elif args.norm == 'depth':
#trained_data = CreateSOM(som_array_depth,somshape=somshape, niter=args.iterations, outfile=args.outfile)

print("training done")

if args.outfile:
    print("Kohonen layer saved as {}".format(args.outfile))

print("mapping data")
mapped = trained_data(som_array)
mapped_tuples = [tuple(point) for point in mapped]
counts = Counter(mapped_tuples)
count_list = [counts[item] for item in mapped_tuples]

save = args.save


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
    plt.scatter(x_pos, y_pos, c=count_list)
    plt.colorbar(label="# of lightcurves at pixel")
    if save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/{}".format(args.output_mapsave))
    plt.close()

array = trained_data.K
plot_all_arrays(array, bins=np.arange(241))
print("done")
response = input("Do you want to view the lightcurves in a specific pixel? y/n")

if response == 'y':
    pixel = input("which pixel? Please write '(x,y)' with quotation marks included.")
    pixel = tuple(map(int, pixel.strip('()').split(',')))
    get_lightcurves(ids,mapped_tuples,pixel,args.path[0],args.output_pixelsave)
    print("saved as pdf file")
else:
    sys.exit()
