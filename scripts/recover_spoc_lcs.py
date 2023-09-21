from multiprocessing import Pool
from tqdm import tqdm
import os

os.nice(15)

def search_sector(sector):
    search_path = f'/storage/astro2/phsqzm/TESS/SPOC_30min/{sector}/target/'
    try:
        files = []
        for root, _, filenames in os.walk(search_path):
            print(f"Searching in {root}")
            subdirectories = sorted(filenames, reverse=True)
            for filename in subdirectories:
                if str(159670453) in filename:
                    files.append(os.path.join(root, filename))
                    print("File found!")
                    break
        if files:
            return id, sector, files
    except Exception as e:
        print(f"Error searching for files in sector {sector}: {str(e)}")
    return None

sectors = [f'S{i:02}' for i in range(1, 27)]  # Creates S01 to S19
id = 159670453  # Replace with your TIC ID

# Number of processes for parallel processing
num_processes = 100  # Adjust as needed based on your system's capabilities

# Initialize an empty list to store results
results = []

# Create a Pool for parallel processing
with Pool(num_processes) as pool, tqdm(total=len(sectors), desc="Searching Sectors") as pbar:
    results = list(filter(None, pool.map(search_sector, sectors)))

