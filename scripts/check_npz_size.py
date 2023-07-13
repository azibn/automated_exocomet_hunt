import numpy as np
import glob

####### Quick check to see if all SOM extracted lightcurves have the same size #######


def check_array_lengths():
    for filename in os.listdir("../som_cutouts/"):
        file = np.load(os.path.join("../som_cutouts/", filename))
        print(f"length time {filename}", len(file["time"]))


def check_npz_array_sizes(file_paths):
    # Load the first .npz file to get the array size
    with np.load(file_paths[0]) as data:
        first_array = data["flux"]
        first_array_size = first_array.shape
        first_array_length = len(first_array)

    # Iterate through the remaining .npz files
    for file_path in file_paths[1:]:
        with np.load(file_path) as data:
            current_array = data["flux"]
            current_array_size = current_array.shape
            current_array_length = len(current_array)

        # Compare the array sizes
        if current_array_size != first_array_size:
            return False, None

        # Compare the array lengths
        if current_array_length != first_array_length:
            return False, None

    return True, first_array_length


# Directory containing the .npz files
directory = "../som_cutouts_s6/"

# Get a list of all .npz files in the directory
file_paths = glob.glob(directory + "*.npz")
# Check if all the .npz files have the same array size
same_array_size, array_length = check_npz_array_sizes(file_paths)

if same_array_size:
    print("All .npz files have the same array size.")
    print("Array length: ", array_length)
else:
    print("The .npz files have different array sizes.")
