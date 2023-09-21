from selfsom import SimpleSOMMapper
from collections import Counter
from TransitSOM_release import CreateSOM
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import somtools
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument("-s", "--save", help="save plots. False by default.", action="store_true", dest="save") ### true by default

parser.add_argument(
    "-somshape",
    "--somshape",
    help="som grid size. Enter the somshape in the form '(x,y)'",
    dest="grid",
)

parser.add_argument(
    "-i", "--iterations", help="number of iterations", default=1000, dest="iterations",type=int
)

parser.add_argument("-output_kohonen", "--kohonensave", help="save the output kohonen layer.", dest="outfile")
parser.add_argument("-output_ms", "--output_mapsave",help="name of output SOM map.", dest = "output_mapsave")

parser.add_argument("-output_ps", "--output_pixelsave",help="name of output SOM map.", dest = "output_pixelsave")

parser.add_argument("-norm", "--normalisation_method",help="the normalisation method for the cutouts. Default is median normalisation.", dest = "norm", default = "median")


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
        normalised_lightcurve = (data['flux'])/np.median(data['flux'])
        median = np.median(data['flux'])
        abs_depth = median - np.min(data['flux'])  # Assuming the minimum of the lightcurve is the minimum point
        depth_normalised_lightcurve = ((data['flux'] - median) / abs_depth + 1)

        obj_id = data['id']
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

def plot_all_arrays(som, bins=np.arange(241)):
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


    if not os.path.exists("som_plots/"):
        os.mkdir("som_plots/")
    else:
        print("Directory already exists, skipping creation.")
    plt.savefig("som_plots/{}".format(args.output_pixelsave))
    plt.close()


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
    trained_data = CreateSOM(som_array_depth,somshape=somshape, niter=args.iterations, outfile=args.outfile)

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