import os
import re
import multiprocessing

# Regular expression pattern to match a 16-digit string
pattern = r'\d{16}'

# Function to extract the 16-digit string from a file path
def extract_16_digit_string(file_path):
    match = re.search(pattern, file_path)
    if match:
        return match.group(0)
    else:
        return None

# Function to process all files in a directory and its subdirectories
def process_directory(directory, output_file):
    print("Processing directory", directory)
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.fits'):
                file_path = os.path.join(dirpath, filename)
                result = extract_16_digit_string(file_path)
                if result:
                    print(result)
                    with open(output_file, 'a') as f:
                        f.write(result + '\n')
                else:
                    print("No match found for " + file_path)

if __name__ == "__main__":
    # Directory containing your files
    directory = "/storage/astro2/phrdhx/eleanor-lite/s0019"
    # Name of the text file to which 16-digit strings will be appended
    output_file = "s19-eleanor-lite.txt"

    # Check if the output file already exists, if not, create it
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            pass

    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(processes=40)

    # Use the Pool to process directories in parallel
    print("Starting multiprocessing pool")
    pool.starmap(process_directory, [(os.path.join(directory, d), output_file) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

    # Close the Pool and wait for all processes to finish
    pool.close()
    pool.join()