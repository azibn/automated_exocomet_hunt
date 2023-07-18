import os
import glob
import argparse

def find_file_by_id(directory, search_id):
    search_pattern = os.path.join(directory, "**", f"*{search_id}*.pkl")
    matching_files = glob.glob(search_pattern, recursive=True)
    
    if matching_files:
        return matching_files
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find files by ID in a directory.")
    parser.add_argument("directory", type=str, help="Directory to search in")
    parser.add_argument("search_id", type=str, help="ID to search for")

    args = parser.parse_args()

    directory = args.directory
    search_id = args.search_id

    result = find_file_by_id(directory, search_id)

    if result:
        print("Matching files found:")
        for file_path in result:
            print(file_path)
    else:
        print("No matching files found.")