import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse 
import pickle
import multiprocessing
from analysis_tools_cython import import_XRPlightcurve, skewed_gaussian


parser = argparse.ArgumentParser(description="Generate synthetic lightcurves using skewed Gaussuian model.")

parser.add_argument(
    "-n", help="number of lightcurves to make", dest="number", type=int
)

parser.add_argument(
    "-dir", help="name of directory to store generated lightcurves. Default is 'synthetic_lightcurves'.", dest="directory", default='synthetic_lightcurves'
)

args = parser.parse_args()


# Define your inject_lightcurve function here
def inject_lightcurve(time, flux, depth, injected_time, sigma, skewness):
    return flux * (
        1 - skewed_gaussian(time, depth, injected_time, sigma, skewness)
    )

# Load the real lightcurve data from a file (assuming it's in CSV format)
# Replace 'real_lightcurve.csv' with the actual filename.
data, lc_info = import_XRPlightcurve('/storage/astro2/phrdhx/tesslcs/tesslcs_sector_6_104/tesslcs_tmag_4_5/tesslc_234957922.pkl',sector=6) # quiet lightcurve
time = data['time']
flux = data['corrected flux']
quality = data['quality']
flux_error = data['flux error']


# Define parameters for generating synthetic lightcurves
no_of_lightcurves = args.number

# Create a directory to store synthetic lightcurves if it doesn't exist
output_directory = args.directory
os.makedirs(output_directory, exist_ok=True)

# Function to generate a single synthetic lightcurve and save it
def generate_and_save_synthetic_lightcurve(i,lc_info=lc_info):
    depth = 10 ** np.random.uniform(0.3, 0.4)
    injected_time = np.random.uniform(min(time), max(time))
    sigma = np.random.uniform(0.01, 0.1)  # Example range for sigma
    skewness = np.random.uniform(-20,20)   # Example range for skewness

    synthetic_flux = inject_lightcurve(time, flux, depth, injected_time, sigma, skewness)
    

    lightcurve = pd.DataFrame(data=[time, synthetic_flux,quality,flux_error]).T

    with open(os.path.join(output_directory,f'synthetic_lc_{i}.pkl'), 'wb') as file:
        pickle.dump((lightcurve, lc_info), file)

    #output_filename = os.path.join(output_directory, f'synthetic_lc_{i}.pkl')
    #np.savetxt(output_filename, np.column_stack((time, synthetic_flux)), delimiter=',')

    # Optionally, you can plot and visualize the synthetic lightcurve
    # plt.plot(real_time, synthetic_flux)
    # plt.xlabel('Time')
    # plt.ylabel('Flux')
    # plt.title(f'Synthetic Lightcurve {i+1}')
    # plt.savefig(f'synthetic_lc_{i}.png')
    # plt.close()

if __name__ == '__main__':
    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the pool to generate synthetic lightcurves in parallel
    pool.map(generate_and_save_synthetic_lightcurve, range(1,no_of_lightcurves))

    # Close the pool to release resources
    pool.close()
    pool.join()

    print(f'{no_of_lightcurves} synthetic lightcurves generated and saved in the {output_directory} directory.')
