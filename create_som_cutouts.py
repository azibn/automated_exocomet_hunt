import pandas as pd
import multiprocessing
from scripts.analysis_tools_cython import import_XRPlightcurve, processing

def process_file(filepath):
    lc, lc_info = import_XRPlightcurve(filepath, sector=6)
    lc = lc[['time', 'corrected flux', 'quality', 'flux error']]
    results, _ = processing(lc, lc_info=lc_info, method='median', som_cutouts=True)
    #plt.scatter(data['TIME'], data['PDCSAP_FLUX'], s=2)

if __name__ == '__main__':
    filename = 'paths.csv'  # Name of the input CSV file

    # Read the CSV file and extract the filepaths using pandas
    df = pd.read_csv(filename)

    # Create a multiprocessing pool with the desired number of processes
    #num_processes = multiprocessing.cpu_count()  # Use the number of available CPU cores
    pool = multiprocessing.Pool(processes=35)

    # Map the filepaths from the DataFrame to the process_file function in parallel
    pool.map(process_file, df['path'])

    # Close the pool to free resources
    pool.close()
    pool.join()