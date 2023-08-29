from selfsom import SimpleSOMMapper
from collections import Counter
from TransitSOM_release import CreateSOM
import os
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

parser.add_argument("-s", "--save", help="save plots", action="store_false", dest="save") ### true by default

parser.add_argument(
    "-grid",
    "--grid",
    help="som grid size. Default is (5,5)",
    default=(5, 5),
    dest="grid",
)

parser.add_argument(
    "-i", "--iterations", help="number of iterations", default=1000, dest="iterations"
)

parser.add_argument("-sector", "--sector", help="TESS sector", dest="sector", type=int)
parser.add_argument("-outfile", "--outfile", help="save the output kohonen layer. default true", dest="outfile")

# Get directories from command line arguments.
args = parser.parse_args()


def stack_npz_files(directory):
    files = os.listdir(directory)
    npz_files = [f for f in files if f.endswith('.npz')]
    if not npz_files:
        print("No .npz files found in the directory.")
        return None

    arrays = []
    stacked_ids = []
    for npz_file in tqdm(npz_files, desc="stacking lightcurves"):
        file_path = os.path.join(directory, npz_file)
        data = np.load(file_path)
        array = data['flux']  # Assuming the array in each .npz file is named 'arr_0'
        obj_id = data['id']
        if len(array) == 241:
            arrays.append(array)
            stacked_ids.append(obj_id.item())

    stacked_array = np.vstack(arrays)

    # Create a dictionary to map arrays to IDs
    id_map = {tuple(array): id for array, id in zip(arrays, stacked_ids)}

    return stacked_array, stacked_ids, id_map

def plot_all_arrays(som, bins=np.arange(241), save=True):
    fig, axes = plt.subplots(5, 5, figsize=(50, 50))
    fig.subplots_adjust(
        wspace=0.4, hspace=0.4
    )  # Adjust the width and height spacing between subplots

    for x_pixel in range(5):
        for y_pixel in range(5):
            ax = axes[x_pixel, y_pixel]
            ax.scatter(bins, som[x_pixel, y_pixel], c="g", s=3)
            ax.set_title("Kohonen pixel at [{},{}]".format(x_pixel, y_pixel))
            # ax.text(0, 0.92, '[{},{}]'.format(x_pixel, y_pixel))

    if save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/som_s{}.png".format(args.sector))
    plt.close()


###### SOM Process ######

print("stacking data")
som_array, lc_ids, _ = stack_npz_files(args.path[0])
somshape = args.grid

print("files loaded")
print("starting som")
trained_data = CreateSOM(som_array, somshape=somshape, niter=args.iterations, outfile=args.outfile)
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
    plt.xlim([-1, 5])
    plt.ylim([-1, 5])
    plt.xlabel("Kohonen X axis")
    plt.xlabel("Kohonen Y axis")
    plt.scatter(x_pos, y_pos, c=count_list)
    plt.colorbar(label="# of lightcurves at pixel")
    if save:
        if not os.path.exists("som_plots/"):
            os.mkdir("som_plots/")
        else:
            print("Directory already exists, skipping creation.")
        plt.savefig("som_plots/Kohonen_s{}.png".format(args.sector))
    plt.close()

array = trained_data.K
plot_all_arrays(array, bins=np.arange(241), save=True)
print("done")