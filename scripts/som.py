from som.selfsom import SimpleSOMMapper
from collections import Counter
from som.TransitSOM_release import CreateSOM
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import som.somtools


parser = argparse.ArgumentParser(description="Analyse lightcurves in target directory.")
parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")

parser.add_argument("-s", "--save", help="save plots", action="store_true", dest="save")

parser.add_argument(
    "-grid",
    "--grid",
    help="som grid size. Default is (15,15)",
    default=(15, 15),
    dest="grid",
)

parser.add_argument(
    "-i", "--iterations", help="number of iterations", default=200, dest="iterations"
)

parser.add_argument("-sector", "--sector", help="TESS sector", dest="sector", type=int)

# Get directories from command line arguments.
args = parser.parse_args()


def stack_npz_files(directory):
    files = os.listdir(directory)
    npz_files = [f for f in files if f.endswith(".npz")]
    if not npz_files:
        print("No .npz files found in the directory.")
        return None

    arrays = []
    obj_id = []
    for npz_file in npz_files:
        file_path = os.path.join(directory, npz_file)
        data = np.load(file_path)
        array = data["flux"]  # Assuming the array in each .npz file is named 'arr_0'
        obj_id.append(data["id"])
        if len(array) == 241:
            arrays.append(array)
            obj_id.append(data["id"].item())

    stacked_array = np.vstack(arrays)
    return stacked_array, obj_id


def plot_all_arrays(som, bins=np.arange(241), save=True):
    fig, axes = plt.subplots(10, 10, figsize=(50, 50))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust the width and height spacing between subplots

    for x_pixel in range(10):
        for y_pixel in range(10):
            ax = axes[x_pixel, y_pixel]
            ax.scatter(bins, som[x_pixel, y_pixel], c='g',s=3)
            ax.set_title('Kohonen pixel at [{},{}]'.format(x_pixel, y_pixel))
            #ax.text(0, 0.92, '[{},{}]'.format(x_pixel, y_pixel))


    if save:
        plt.savefig('../s{}_som_pixel_by_pixel.png'.format(args.sector)) 

        try:
            os.mkdir('../som_plots')
        except FileExistsError:
            pass
        plt.savefig('pixel_by_pixel_s{}.png'.format(args.sector),dpi=400)  # Save the figure if save is True
    
    plt.show()


print('starting')
##########################################
som_array, lc_ids = stack_npz_files(args.path[0])
#lc_ids = stack_npz_files(args.path[0])[1]
somshape = args.grid

print('files loaded')
print("starting som")
trained_data = CreateSOM(som_array, somshape=somshape, niter=args.iterations)
print("training done")

mapped = trained_data(som_array)
mapped_tuples = [tuple(point) for point in mapped]
counts = Counter(mapped_tuples)
count_list = [counts[item] for item in mapped_tuples]

save = args.save


som_x = somshape[0]
som_y = somshape[1]

####### Save Kohonen grid
colour = "count"  # can be 'count', 'VarGroup', False
if colour == "count":
    x_pos = mapped[:, 0]
    y_pos = mapped[:, 1]
    count_list = [counts[item] for item in mapped_tuples]
    plt.figure(figsize=(10, 7))
    plt.xlim([-1, 20])
    plt.ylim([-1, 20])
    plt.xlabel("Kohonen X axis")
    plt.xlabel("Kohonen Y axis")
    plt.scatter(x_pos, y_pos, c=count_list, s=count_list)
    plt.colorbar(label="# of lightcurves at pixel")
    if save:
        plt.savefig('Kohonen_s{}.png'.format(args.sector))


array = trained_data.K
plot_all_arrays(array, bins=np.arange(241), save=True)